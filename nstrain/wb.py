"""W&B logging with one continuous curve per training line, taken from the server only (no local metric log).

* Every line (trunk, each branch, later stages) has one W&B run; its id is stored in every checkpoint.  The step is the
  update number, the x axis of the charts is ``tokens`` (``define_metric``).
* Resume after a crash / stop (checkpoint at update S, the run may hold updates beyond S): the same run is resumed
  (``resume="must"``).  W&B ignores steps at or below the last step it holds, so the replayed updates S+1.. are not
  logged twice; before that, the replayed updates are compared with the server's copy (:meth:`check_replay`).  A
  replay that differs (not the same computation) switches to a new run as for a rollback.
* Rollback (continuing from an earlier checkpoint on purpose, with different results): a new run of the same name,
  backfilled from the old run's server history up to S; the old run is renamed ``<name> (superseded @S)``, tagged
  ``superseded`` and moved to the group ``superseded`` (nothing is deleted).
* Branch: a new run, backfilled from the parent run's server history up to the fork update; ``parent_run`` /
  ``parent_step`` in its config.
* The run of a line is recorded in ``<ckpt_root>/wandb_run.json`` (:class:`RunRecord`) as soon as it is created, with
  the backfill still to do; a restart (before or after the line's first checkpoint, during a backfill) resumes that
  run and completes the backfill (W&B drops steps at or below the last one it holds), so a line never gets a second
  run by accident.
* Every call to the W&B service is retried with backoff for up to ``retry_s`` seconds (an outage stalls the start of
  a run instead of failing it); logging during the run is asynchronous in the W&B client.
* Rows the server never received (the client delivers asynchronously; a crash loses what it had not sent) are sent
  again at the resume from the rows stored in the checkpoints (nstrain.trainer): the line's curve has no hole as long
  as the missing updates come after the server's last one (the server drops steps at or below it).
* The client's local files are kept in ``<local_dir>/wandb``; those of a process whose run finished cleanly (every row
  delivered) are removed at the next start (:meth:`Tracker.clean_local`), those of a crashed process stay.
"""
from __future__ import annotations

import json
import math
import os
import random
import secrets
import shutil
import string
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import wandb

AXIS = "tokens"


class RunRecord:
    """``<ckpt_root>/wandb_run.json``: the line's W&B run (``run_id``), how it was created (``kind``: start, branch,
    rollback), the run and step its history is backfilled from (``source``, ``source_step``) and whether that backfill
    is complete (``ready``).  Written atomically."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> Optional[dict]:
        return json.loads(self.path.read_text()) if self.path.exists() else None

    def save(self, rec: dict) -> dict:
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(rec, indent=1))
        os.replace(tmp, self.path)
        return rec


class Tracker:
    def __init__(self, project: str, name: str, config: dict, entity: Optional[str] = None, retry_s: float = 7200.0,
                 on_wait: Optional[Callable[[], None]] = None, local_dir: Optional[Path] = None):
        self.project, self.name, self.config, self.entity = project, name, config, entity
        self.local_dir = Path(local_dir) if local_dir is not None else None
        self.retry_s = float(retry_s)
        self.on_wait = on_wait                                       # called before every retry wait (heartbeat)
        self.run = None
        self.replay: Dict[int, dict] = {}                            # server rows above the resumed checkpoint

    # ------------------------------------------------------------------ helpers
    def _retry(self, what: str, fn: Callable):
        """``fn()``, retried with backoff (10 s doubling to 5 min, jittered) for up to ``retry_s`` seconds."""
        t0, wait = time.time(), 10.0
        while True:
            try:
                return fn()
            except Exception as e:                                   # network / service errors of the W&B client
                if time.time() - t0 + wait > self.retry_s:
                    raise
                print(f"wandb: {what} failed ({type(e).__name__}: {str(e)[:200]}); retrying in {wait:.0f} s",
                      flush=True)
                if self.on_wait is not None:
                    self.on_wait()
                time.sleep(wait * random.uniform(0.8, 1.2))
                wait = min(2 * wait, 300.0)

    def _init(self, **kw):
        def go():
            if self.local_dir is not None:
                kw["dir"] = str(self.local_dir)
            run = wandb.init(project=self.project, entity=self.entity, **kw)
            run.define_metric(AXIS)
            run.define_metric("*", step_metric=AXIS)
            return run
        return self._retry("init", go)

    def _path(self, run_id: str) -> str:
        ent = self.entity or self._retry("api", lambda: wandb.Api().default_entity)
        return f"{ent}/{self.project}/{run_id}"

    def _tick(self, n: int) -> None:
        if n % 1000 == 0 and self.on_wait is not None:              # long scans / backfills keep the heartbeat
            self.on_wait()

    def history(self, path: str, upto: Optional[int] = None, above: Optional[int] = None) -> List[dict]:
        def go():
            rows = []
            for n, row in enumerate(wandb.Api().run(path).scan_history(), 1):
                self._tick(n)
                s = row.get("_step")
                if s is None or (upto is not None and s > upto) or (above is not None and s <= above):
                    continue
                rows.append({k: v for k, v in row.items() if not k.startswith("_") or k == "_step"})
            rows.sort(key=lambda r: r["_step"])
            return rows
        return self._retry(f"history of {path}", go)

    # ------------------------------------------------------------------ runs
    @staticmethod
    def new_id() -> str:
        """A W&B run id (8 lowercase letters and digits), chosen before the run is created."""
        return "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(8))

    def create(self, run_id: str, extra_config: Optional[dict] = None) -> str:
        """The run ``run_id`` of this line's name (``extra_config`` added to its config): created, or continued when a
        crash came after its creation (the id is recorded before this call, so a run is never created twice)."""
        self.run = self._init(id=run_id, resume="allow", name=self.name,
                              config=dict(self.config, **(extra_config or {})))
        return self.run.id

    def resume(self, run_id: str, ckpt_step: int) -> str:
        """Continue ``run_id``; the server's rows above ``ckpt_step`` are kept for :meth:`check_replay`."""
        self.run = self._init(id=run_id, resume="must")
        self.replay = {r["_step"]: r for r in self.history(self._path(run_id), above=ckpt_step)}
        return self.run.id

    def steps_from(self, run_id: str, lo: int) -> set:
        """The steps the server holds for ``run_id`` from ``lo`` on."""
        def go():
            return {r["_step"] for r in wandb.Api().run(self._path(run_id)).scan_history(min_step=lo)
                    if r.get("_step") is not None}
        return self._retry(f"steps of {run_id}", go)

    def backfill(self, src_run_id: str, upto: int) -> int:
        """Log the source run's server rows up to ``upto`` into the current run (rows at or below the current run's
        last step are dropped by W&B, so a backfill interrupted by a restart is completed by running it again)."""
        n = 0
        for row in self.history(self._path(src_run_id), upto=upto):
            step = row.pop("_step")
            self.run.log({k: v for k, v in row.items() if v is not None}, step=step)
            n += 1
            self._tick(n)
        return n

    def supersede(self, run_id: str, at: int) -> None:
        """Rename ``run_id`` to ``<name> (superseded @at)``, tag it and move it to the group ``superseded``."""
        def go():
            old = wandb.Api().run(self._path(run_id))
            if "superseded" in old.tags:
                return
            old.name = f"{old.name} (superseded @{at})"
            old.tags = sorted(set(old.tags) | {"superseded"})
            old.group = "superseded"
            old.update()
        self._retry(f"supersede {run_id}", go)

    # ------------------------------------------------------------------ use
    def check_replay(self, step: int, metrics: dict, keys=("train/loss", "train/grad_norm"),
                     rtol: float = 1e-3) -> bool:
        """True if ``step`` is not on the server or matches it; consumed entries are dropped."""
        row = self.replay.pop(step, None)
        if row is None:
            return True
        for k in keys:
            a, b = row.get(k), metrics.get(k)
            if a is None or b is None:
                continue
            try:
                a, b = float(a), float(b)                            # the server may hold NaN as a string
            except (TypeError, ValueError):
                return False
            if not math.isfinite(a) or not math.isfinite(b):
                if math.isfinite(a) or math.isfinite(b):             # a skipped update replays as skipped
                    return False
                continue
            if abs(a - b) > rtol * max(abs(a), abs(b), 1e-12):
                return False
        return True

    def log(self, step: int, metrics: dict) -> None:
        self.run.log(metrics, step=step)

    def finish(self, exit_code: int = 0) -> None:
        if self.run is not None:
            files = Path(self.run.dir).parent                      # <local_dir>/wandb/run-<time>-<id>
            self.run.finish(exit_code=exit_code)
            self.run = None
            if self.local_dir is not None:
                with open(self.local_dir / "wandb" / "finished.txt", "a") as f:
                    f.write(str(files) + "\n")

    def clean_local(self) -> None:
        """Remove the local files of the runs that finished cleanly (listed by :meth:`finish`)."""
        if self.local_dir is None:
            return
        done = self.local_dir / "wandb" / "finished.txt"
        if not done.exists():
            return
        for line in done.read_text().split():
            d = Path(line)
            if d.parent == self.local_dir / "wandb" and d.is_dir():
                shutil.rmtree(d)
        done.unlink()
