"""Pre-training of a NeuronSpark model on the prepared cycles of a data line: one trainer process (``nstrain train
RUN``; ``nstrain run RUN`` runs it under the supervisor, see nstrain.supervise).

The line's run directory (see nstrain.config for run.json)::

    run.json            the configuration
    checkpoints/        u<update>/ (nstrain.ckpt), latest; stale-*/, rolled-back-*/ (moved aside, never read)
    run.log             the trainer's and the supervisor's output (written by the supervisor)
    heartbeat           {time, update, stage, pid}: after every update and every step that can take long (loading,
                        calibration, CUDA-graph capture, each validation source, waiting for data, W&B retries)
    wandb_run.json      the line's W&B run (nstrain.wb.RunRecord)
    trainer.lock        one trainer per line
    STOP                stop with a checkpoint: empty = at the next update boundary, {"at_cycle_end": k} = after the
                        last update of cycle k (``nstrain stop``); renamed STOP.honored-<update> when honored
                        (nstrain.proc.honor_stop)
    wandb/              the W&B client's local files (those of runs that finished cleanly are removed at the next
                        start)
    rollback.json       written by `nstrain rollback`: this start continues as a new W&B run (removed once it exists)

Starts: a fresh line writes a kept checkpoint of its launch state (update 0) before its first update; a branch
(``fork_from``) starts from the parent's kept cycle-end checkpoint, forked at that checkpoint's cycle (checked
against the branch's data state), with the parent's model configuration and schedule up to the fork; a continued
line keeps the FROZEN settings of its checkpoints and its schedule up to the checkpoint (append-only), unless listed in
``allow_changes`` (see :func:`start_problems`; ``nstrain check`` runs the same checks without a GPU).

Stopping: SIGTERM / SIGINT / SIGHUP or the STOP file end the run at an update boundary with a checkpoint (also while
waiting for data).  Each cycle end and each change of sequence length (a stage end: the state after the last update
of the old length, also when that update's checkpoint was a rolling one or the run resumed from it) is a kept
checkpoint; the data preparer is told a cycle is consumed only after that checkpoint is on disk, and is told the cycle
in training when the trainer starts.  Exit codes (nstrain.exits): 0 done / stopped / the line's data is exhausted, 3
another process uses the GPU (beside other processes, at the start: the shapes do not fit beside them), 4 another
trainer holds the line, 5 a condition a relaunch cannot fix (not enough disk space for the next checkpoints, checked
after every update -- a checkpoint is written while there is room --; a failed start check or preflight; a rollback
marker that does not match ``latest``; another trainer reading the data line; the data read off their plan), 1 failure
(also when nvidia-smi fails: at start, or at three checks in a row, then after a checkpoint).  A stop by a signal
(exit 0) writes "stopped by a signal" as the last heartbeat stage (the supervisor relaunches a trainer it did not
stop).  The GPU is checked every ``guards.gpu_check_every`` updates.

The GPU to itself, the line trains with each length's own shape; its arena (nstrain.gpu.Arena) holds all of the
GPU's memory but a capture's slack, in large blocks (``_hold_all``), and every allocation of the engine's warm-up and
capture and of the validation rows of any length is served from them.  With ``guards.share_reserve_gb`` set it also
trains beside other processes, and changes mode inside the process, at a check (``_switch``; nothing else moves:
weights, optimizer state, the data position and the W&B run stay as they are): when they appear (in the slack, at
first) it releases its engine and arena, holds at once all of the GPU but what they may still take -- the reserve
less what they use (nstrain.gpu.hold_memory, adjusted at every check: they cannot grow past the reserve) -- and
captures the length's ``shared`` shape (measured for that reserve by ``nstrain tune --shared``; a start without one
is refused) in the new arena; when they have left for ALONE_CHECKS checks it gives the arena back, holds all of the
GPU again and captures the length's own shape.  The memory is free for anything to take only between the close of
the old arena and the hold of the new one (one round of cudaFree / cudaMalloc).  When they use more than the reserve
and the shapes' margin -- at a start, at a check before the switch, or taken in that gap so that the capture beside
them runs out of memory -- the line checkpoints and exits 3: the supervisor relaunches it once they fit or the GPU is
free.  Without a reserve another process appearing ends the run with a checkpoint (exit 3).  Started by a supervisor
(``NSTRAIN_SUPERVISOR_PID``), the trainer gets SIGTERM when the supervisor dies.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

import torch

from nstrain import config as C
from nstrain import exits
from nstrain import model as M
from nstrain import schedule as sched
from nstrain.checks import preflight, start_problems
from nstrain.ckpt import Checkpoints, snapshot
from nstrain.data.batches import Line, Prefetch, val_rows
from nstrain.data.stream import ORDER_FORMAT
from nstrain.engine import StepEngine
from nstrain.flat_adamw import FlatAdamW
from nstrain.gpu import (CAPTURE_SLACK_GIB, GpuBusy, GpuCheckFailed, beside_limit_gib, cap_memory, fits_beside,
                        reclaim_dead,
                         hold_memory, lease, other_processes, others_gb, room_beside)
from nstrain.probes import dendritic_line, dendritic_metrics, hippocampal_line, hippocampal_metrics
from nstrain.proc import honor_stop
from nstrain.wb import RunRecord, Tracker
from neuronspark.utils.probe import DendriticProbe, HippocampalProbe, probing


class LineBusy(RuntimeError):
    pass


class NeedsOperator(RuntimeError):
    pass


def _version(root: Path) -> str:
    """The commit of a source tree: its git HEAD, or the ``.commit`` file a copy without .git carries."""
    if (root / ".git").exists():
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    c = root / ".commit"
    return c.read_text().strip() if c.exists() else "unknown"


def _tree_sha256(*pkgs: Path) -> str:
    """sha256 over the packages' Python sources (relative path and content): the code itself, with or without git."""
    h = hashlib.sha256()
    for pkg in pkgs:
        for f in sorted(pkg.rglob("*.py")):
            h.update(f"{pkg.name}/{f.relative_to(pkg)}".encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def code_versions(tokenizer_sha256: str) -> dict:
    import neuronspark
    ns_pkg, here = Path(neuronspark.__file__).resolve().parent, Path(__file__).resolve().parent
    v = {"neuronspark": _version(ns_pkg.parent), "neuronspark_tree_sha256": _tree_sha256(ns_pkg),
         "nstrain": _version(here.parent), "nstrain_tree_sha256": _tree_sha256(here),
         "tokenizer_sha256": tokenizer_sha256}
    for k in ("neuronspark", "nstrain"):
        if v[k] == "unknown":
            print(f"warning: the {k} commit is unknown (no .git / .commit); checkpoints record its source sha256 "
                  "only", flush=True)
    return v


MIX_ALARM_Z = 8.0                  # the read mix this many standard deviations off the plan: an alert
ALONE_CHECKS = 3                   # checks without other processes before a line beside them takes the GPU back


class Trainer:
    def __init__(self, run):
        self.stop = False
        self.stop_honored = False
        self.rows = []                                               # (update, row) logged since the last checkpoint
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, self._on_signal)
        self.cfg = c = C.load(run)
        self.root = c.run_dir
        self.dev = torch.device("cuda")
        self.update = 0
        self.finished = False
        self._lock = open(self.root / "trainer.lock", "w")           # one trainer per line
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LineBusy(f"another trainer holds {self.root / 'trainer.lock'}") from None
        self.reserve = c.guards.share_reserve_gb                    # GB left to other processes, else none allowed
        self.shared = False                                          # beside other processes now (see the module
        self.alone_checks = 0                                        # documentation; _switch); checks without them
        self.others_last = 0.0                                       # their GB when last measured (nstrain.gpu)
        self.stopped_waiting = False                                 # a stop came while waiting for data
        self._beat("start")                                          # (its mode: the supervisor's crash count)
        if not c.guards.allow_shared_gpu:                           # before any CUDA initialisation
            self._lease = lease(str(self.root))
            others = other_processes(exclude_pid=os.getpid())
            if others and self.reserve is None:
                raise GpuBusy(f"other processes on the GPU: {others}; training needs the GPU to itself "
                              "(guards.allow_shared_gpu overrides)")
            if others:
                self.others_last = others_gb(exclude_pid=os.getpid())
                if not fits_beside(self.others_last, self.reserve):
                    raise GpuBusy(f"the shapes do not fit beside the other processes on the GPU (more than the "
                                  f"{self.reserve:g} GB reserved for them and the tune's margin): {others}")
                self.shared = True
        problems = preflight(c)
        if problems:                                                 # a configuration to fix: no relaunch helps
            raise NeedsOperator("cannot start: " + "; ".join(problems))
        if self.shared:                                              # out of memory before the reserve is touched
            cap_memory(beside_limit_gib(self._total_gib(), self.reserve))
        self.model_cfg = c.model
        self.probe_len = M.longest_context(c.model)                  # the long-row probe's length (_probe_long)
        self.ckpts = Checkpoints(c.ckpt_dir, keep_last=c.checkpoints.keep_last)
        self.line = Line(c.data_work, c.tokens_per_update, seed=c.data_seed)
        self.data_state = json.loads((Path(c.data_work) / "state.json").read_text())
        self.eos_id = int(self.data_state["eos_id"])                 # ends every document of the rows (the probes)
        self.code = code_versions(self.data_state["tokenizer_sha256"])
        latest = self.ckpts.recover(self.root)
        self.rollback = self._rollback_marker(latest)
        src = latest if latest is not None else (Path(c.fork_from) if c.fork_from else None)
        self.last_ckpt = None
        self.prev_shape = None                                       # shape of the last update (changes: a stage end)
        self.in_a_row = 0                                            # non-finite updates in a row before this one
        if src is not None:
            meta, ms, opt_state = Checkpoints.load(src)
            self._beat("checkpoint loaded")
            problems = start_problems(c, meta, self.data_state, forking=latest is None)
            if problems:
                raise NeedsOperator("cannot start: " + "; ".join(problems) + "  (allow_changes in run.json lists "
                                    "keys that may change on purpose)")
            self.model = M.build(self.model_cfg, self.dev, init="empty")
            self.model.load_state_dict({k: v.to(self.dev) for k, v in ms.items()}, strict=True)
            self.update, self.tokens = int(meta["update"]), int(meta["tokens"])
            self.skipped = int(meta.get("skipped", 0))
            self.in_a_row = int(meta.get("skipped_in_a_row", 0))
            if latest is not None:                                   # continue this line
                self.progress = (int(meta["data"]["cycle"]), int(meta["data"]["sample"]))
                self.expect_sha = meta["data"]["manifest_sha256"]
                self.expect_order = meta["data"].get("order_digest", "")
                self.last_ckpt = self.update
                self.prev_shape = meta.get("shape")
            else:                                                    # a branch starts at its own cycle 0
                self.progress, self.expect_sha, self.expect_order = (0, 0), "", ""
        else:
            meta, opt_state = None, None
            torch.manual_seed(c.seed)
            self.model = M.build(self.model_cfg, self.dev, init="launch")
            self.update = self.tokens = self.skipped = 0
            self.progress, self.expect_sha, self.expect_order = (0, 0), "", ""
        o = c.optimizer
        groups = M.param_groups(self.model, o.weight_decay, o.lexical_lr_mult)
        self.opt = FlatAdamW(self.model, groups, tuple(o.betas), o.eps)
        if opt_state is not None:
            self.opt.load_state({n: {k: v.to(self.dev) for k, v in st.items()} for n, st in opt_state.items()})
        self.ckpt_bytes = (sum(t.numel() * t.element_size() for t in self.model.state_dict().values()) +
                           sum(t.numel() * t.element_size() for st in self.opt.state().values() for t in st.values()))
        if not self._room(1):
            raise NeedsOperator(self._space_message(1))
        self.engine = None
        torch.cuda.empty_cache()                                     # the build's and the load's freed blocks (the
        self.mix: Dict[tuple, tuple] = {}                            # arena cannot use them; before any hold)
        self.mix_alarm = None
        from nstrain.gpu import Arena
        self.arena = Arena()                                         # the engines', calibration's, validation's memory
        self.slack = self.slack_pool = None                          # (_slack_in)
        if self.shared:                                              # from here on -- the wait for data, W&B -- they
            self._slack_in(self._others(None))                       # get the reserve, no more
        if c.stop_at_tokens is not None and self.tokens >= c.stop_at_tokens:
            print(f"line {c.name}: stop_at_tokens {c.stop_at_tokens:.0f} reached (update {self.update}); nothing to "
                  "train", flush=True)                               # (not waiting for the next cycle's data)
            self.finished = True
            return
        try:
            self.line.consume(self.root)
        except RuntimeError as e:
            raise NeedsOperator(str(e)) from None
        self.line.mark(self.progress[0])
        self.updates = Prefetch(self.line.updates(*self.progress, expect_sha256=self.expect_sha,
                                                  expect_order=self.expect_order))
        self.first = self._next_update(self.progress[0])
        if self.first is None:
            if self.stopped_waiting:
                print(f"line {c.name}: stopped before its first update here (update {self.update})", flush=True)
                self.finished = True
                return
            if latest is not None:
                print(f"line {c.name}: its data is exhausted (update {self.update}); nothing to train", flush=True)
                self.finished = True
                return
            raise RuntimeError("no data on this line")
        if src is None:                                              # the process's first forward: memory outside
            self._slack_out()                                        # the allocator, as a capture's (_engine_for)
            with self.arena.use():
                M.calibrate(self.model, self.first.inputs, c.length(self.first.seq_len).micro_rows, self.dev)
            if self.shared:
                self._slack_in(self._others(None))
            self._beat("calibrated")
            self._save(self.first.cycle, self.first.first, self.first.manifest_sha256, self.first.order_digest,
                       keep=True, background=False, shape=None, run_id=None)     # the launch state of the line
            self.last_ckpt = 0

        wcfg = dict(c.to_dict(), run_dir=str(self.root), model_config=self.model_cfg, code=self.code)
        self.wb = Tracker(c.wandb.project, c.name, wcfg, c.wandb.entity, retry_s=c.wandb.retry_s,
                          on_wait=lambda: self._beat("wandb"), local_dir=self.root, key_file=c.wandb.key_file)
        self.wb.clean_local()
        self.rec = RunRecord(self.root / "wandb_run.json")
        self.run_id = self._open_run(meta, forking=src is not None and latest is None)
        if latest is not None:
            self._resend()
        self.val = {}

    # ------------------------------------------------------------------ small pieces
    def _on_signal(self, *_):
        self.stop = True

    def _stop_file(self):
        p = self.root / "STOP"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text() or "{}")
        except json.JSONDecodeError:
            return {}                                                # unreadable: an unconditional stop

    def _stop_requested(self, u=None, before: bool = False, at: Optional[int] = None) -> bool:
        """A stop by signal, or by the STOP file: unconditional, or {"at_cycle_end": k}, due after the last update
        of cycle k (``before``: before an update of a later cycle; ``at``: waiting for the data of cycle ``at``)."""
        s = self._stop_file()
        due = False
        if s is not None:
            k = s.get("at_cycle_end")
            due = (k is None or (at is not None and at > k) or
                   (u is not None and (u.cycle > k if before else (u.cycle > k or (u.cycle == k and u.cycle_end)))))
            self.stop_honored = self.stop_honored or due
        return self.stop or due

    def _room(self, n: int) -> bool:
        return shutil.disk_usage(self.root).free >= n * 1.1 * self.ckpt_bytes + 10e9

    def _space_message(self, n: int) -> str:
        return (f"{shutil.disk_usage(self.root).free / 1e9:.0f} GB free on {self.root}, {n} checkpoint(s) need "
                f"{(n * 1.1 * self.ckpt_bytes + 10e9) / 1e9:.0f} GB: free space, then continue the line")

    def _beat(self, stage: str) -> None:
        tmp = self.root / "heartbeat.tmp"
        tmp.write_text(json.dumps({"time": time.time(), "update": self.update, "stage": stage, "pid": os.getpid(),
                                   "others": getattr(self, "others_seen", 0),
                                   "mode": "shared" if getattr(self, "shared", False) else "exclusive"}))
        os.replace(tmp, self.root / "heartbeat")

    def _rollback_marker(self, latest):
        """rollback.json (``nstrain rollback``): {"to": update, "source_run": the W&B run it supersedes}."""
        p = self.root / "rollback.json"
        if not p.exists():
            return None
        m = json.loads(p.read_text())
        if latest is None or latest.name != f"u{int(m['to']):08d}":
            raise NeedsOperator(f"{p}: a rollback to update {m['to']}, but `latest` is {latest}; re-run "
                                "`nstrain rollback`")
        return m

    def _resend(self) -> None:
        """Rows of updates up to this checkpoint that the W&B server never received (a crash before the client
        delivered them) are logged again from the rows the checkpoints carry."""
        rows = {}
        for d in self.ckpts.complete():
            if int(d.name[1:]) <= self.update:
                for step, row in json.loads((d / "trainer.json").read_text()).get("rows", []):
                    rows[int(step)] = row
        if not rows:
            return
        have = self.wb.steps_from(self.run_id, min(rows))
        top = max(have) if have else min(rows) - 1
        late = [k for k in sorted(rows) if k not in have and k > top]
        holes = [k for k in sorted(rows) if k not in have and k <= top]
        for k in late:
            self.wb.log(k, rows[k])
        if late:
            print(f"W&B: re-sent updates {late[0]}..{late[-1]} ({len(late)}) the server never received", flush=True)
        if holes:
            print(f"warning: the W&B run lacks updates {holes[:20]} below its last one ({top}); they cannot be sent "
                  "again", flush=True)

    def _next_update(self, at: int):
        """The next update (of cycle ``at`` or later); None at the end of the line or when a stop is requested while
        the preparer is behind (a STOP at the end of a cycle before ``at`` is due then)."""
        waited = 0
        while True:
            u = self.updates.get(timeout=60.0)
            if u is not Prefetch.WAITING:
                return u
            if self._stop_requested(at=at):
                print(f"update {self.update}: stop requested while waiting for prepared data", flush=True)
                self.stopped_waiting = True
                return None
            waited += 1
            self._beat("waiting for data")
            if waited % 10 == 0:
                print(f"update {self.update}: waiting for prepared data ({waited} min)", flush=True)

    def _open_run(self, meta, forking: bool) -> str:
        """The line's W&B run (see nstrain.wb), continued from wandb_run.json or created; the id is recorded before
        the run is created and marked ready once the server holds its backfill, so a restart at any point completes
        the same run.  A
        rollback marker makes a new run (once: a restart finds the rollback's run in the record and continues it)."""
        rec = self.rec.load()
        if rec is not None and not rec["ready"]:                     # a creation / backfill cut short: finish it
            self.wb.create(rec["run_id"], self._run_config(rec["kind"], rec["source"], rec["source_step"]))
            if rec["source"] is not None:
                self.wb.backfill(rec["source"], rec["source_step"])
                self.wb.confirm(rec["run_id"], rec["source_step"])
            if rec["kind"] == "rollback":
                self.wb.supersede(rec["source"], rec["source_step"])
            rec = self.rec.save(dict(rec, ready=True))
        m = self.rollback
        if m is not None:
            if not (rec is not None and rec["kind"] == "rollback" and rec["source"] == m["source_run"]
                    and rec["source_step"] == m["to"]):
                if self.wb.run is not None:
                    self.wb.finish()
                rec = {"run_id": self._new_run("rollback", m["source_run"], int(m["to"]))}
            (self.root / "rollback.json").unlink()
            print(f"rollback to update {m['to']}: W&B run {rec['run_id']} (supersedes {m['source_run']})", flush=True)
            if self.wb.run is None:
                self.wb.resume(rec["run_id"], self.update)
            return rec["run_id"]
        if rec is None:
            if forking:                                              # a branch: the parent's checkpoint
                return self._new_run("branch", meta["wandb_run"], self.update)
            if meta is not None and meta.get("wandb_run"):           # a line checkpointed before the record existed
                rec = self.rec.save({"run_id": meta["wandb_run"], "kind": "start", "source": None,
                                     "source_step": None, "ready": True})
            else:
                return self._new_run("start", None, None)
        if self.wb.run is None:
            self.wb.resume(rec["run_id"], self.update)
        return rec["run_id"]

    def _new_run(self, kind: str, source, step) -> str:
        rid = self.wb.new_id()
        rec = self.rec.save({"run_id": rid, "kind": kind, "source": source, "source_step": step, "ready": False})
        self.wb.create(rid, self._run_config(kind, source, step))
        if source is not None:
            self.wb.backfill(source, step)
            self.wb.confirm(rid, step)                               # on the server before the line's own rows
        if kind == "rollback":
            self.wb.supersede(source, step)
        self.rec.save(dict(rec, ready=True))
        return rid

    @staticmethod
    def _run_config(kind: str, source, step) -> dict:
        """The W&B config entries of a run's origin."""
        return {"rollback": {"rollback_of": source, "rollback_step": step},
                "branch": {"parent_run": source, "parent_step": step}}.get(kind, {})

    @staticmethod
    def _total_gib() -> float:
        return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 2**30

    def _switch(self, shared: bool, others: float) -> None:
        """The GPU mode changed at a check (``guards.share_reserve_gb`` set): beside other processes that appeared, or
        with the GPU to itself again once they left.  The engine is released and the arena closed -- its memory goes
        back to the driver -- and a new arena is made: beside them it holds at once all of the GPU but what they may
        still take (the reserve less their ``others`` GB) and keeps the capture slack (``_slack_in``); alone, the
        next update's capture holds all of the GPU but the slack (``_hold_all``).  The next update captures the
        length's shape of the new mode (``_engine_for``).  Weights, optimizer state, the data position and the W&B
        run do not move.  Nothing is printed between the close and the hold: that gap is the only time the memory is
        free for anything to take."""
        import gc
        from nstrain.gpu import Arena
        if self.engine is not None:
            self.engine.release()
            self.engine = None
        self._slack_out()
        gc.collect()                                                 # (the dead engine's graphs, before the close:
        self.arena.close()                                           # nothing left for its own collection)
        self.arena = Arena()
        self.shared = shared
        print(f"update {self.update}: the arena closed: {torch.cuda.memory_reserved() / 2**30:.1f} GiB reserved by "
              f"the allocator, {torch.cuda.mem_get_info()[0] / 2**30:.1f} GiB free on the GPU", flush=True)
        if shared:
            cap_memory(beside_limit_gib(self._total_gib(), self.reserve))
            self._slack_in(others, quick=True)                       # their memory as the check read it: no
                                                                     # nvidia-smi in the gap
            print(f"update {self.update}: other processes on the GPU use {others:.1f} GB ({self.reserve:g} GB "
                  "reserved for them): beside them from here, with the shapes measured for the reserve", flush=True)
        else:
            cap_memory(self._total_gib())
            dead = 0
            while (got := reclaim_dead()) >= 16 << 20:               # what the driver kept from the processes that
                dead += got                                          # left: reclaimed, so the GPU is whole again
            print(f"update {self.update}: the other processes left the GPU: with the GPU to itself from here, with "
                  f"the lengths' own shapes ({dead / 2**30:.1f} GiB of theirs reclaimed from the driver)", flush=True)
        self.alone_checks = 0

    def _engine_for(self, L: int, others: Optional[float] = None) -> StepEngine:
        """The engine of length L, warmed up and captured in the trainer's arena (nstrain.gpu.Arena): a new one
        reuses the memory the arena holds.  Beside other processes the slack (``_slack_in``) goes back to the driver
        for the capture -- the memory outside the allocator that a warm-up and a capture create (kernel modules,
        local memory, graph data) -- while the arena holds all the rest but what they may still take (``others``:
        their GB, else measured); the slack's unused rest is held again after the capture.  Alone, the arena holds
        all of the GPU but that slack first (``_hold_all``), and the slack stays free.  One validation micro-batch
        and one long row run before the capture (their first-call caches then live outside the graphs' scratch)."""
        if self.engine is not None and self.engine.L == L:
            return self.engine
        if self.engine is not None:
            self.engine.release()                                    # its memory stays in the arena (beside others:
            self.engine = None                                       # held, the slack aside); alone on the GPU it
            if not self.shared:                                      # goes back to the driver, so that the new
                from nstrain.gpu import Arena                        # length's engine fits as in `nstrain tune` (a
                self.arena.close()                                   # process of its own)
                self.arena = Arena()
        c = self.cfg
        if self.shared:
            self._slack_out()
            self._hold(self._others(others), slack_gib=CAPTURE_SLACK_GIB, reclaim=True)
        else:
            self._hold_all()
        self._beat(f"capturing the {L} engine")
        s, accum = c.length(L, self.shared), c.accum(L, self.shared)
        M.recompute(self.model, s.activation_checkpointing)
        t0 = time.time()
        self.engine = StepEngine(self.model, self.opt, s.micro_rows, L, accum, c.optimizer.grad_clip,
                                 arena=self.arena, before_capture=lambda: self._warm_validation(L))
        print(f"engine {accum} x {s.micro_rows} x {L}{' (beside other processes)' if self.shared else ''} captured "
              f"in {time.time() - t0:.0f} s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
        torch.cuda.empty_cache()                                     # blocks freed outside the arena since the last
        if self.shared:                                              # release (a closed arena's, a pool's) back to
            self._slack_in(self._others(None))                       # the driver: free alone, held beside others
        self._beat(f"{L} engine captured")
        return self.engine

    def _slack_in(self, others: float, quick: bool = False) -> None:
        """Beside other processes: all of the GPU held but what they may still take -- in the arena all but
        CAPTURE_SLACK_GIB, the slack less 0.25 GiB in a block of its own pool outside the arena (``self.slack``),
        which the next capture gives back to the driver (``_slack_out``): memory only a capture needs, never free for
        the others to take in between (nstrain.gpu.room_beside: what they free meanwhile stays theirs).  ``quick``
        (the switch to this mode, ``_switch``): their memory is ``others`` as the check read it, not read again, and
        the dead memory is not probed -- the shortest hold, the next check does both."""
        self._hold(others, slack_gib=CAPTURE_SLACK_GIB, reclaim=not quick, quick=quick)

        # the slack's second reading: when nvidia-smi fails, the first one (no slack would be worse: the next capture
        # would take its memory from what the others may still use)
        def again() -> float:
            if quick:
                return others
            try:
                return others_gb(exclude_pid=os.getpid())
            except GpuCheckFailed:
                return others
        n = min(int((CAPTURE_SLACK_GIB - 0.25) * 2**30), room_beside(self.cfg.guards.share_reserve_gb, others, again))
        n = n // (2 << 20) * (2 << 20)
        if self.slack is None and n >= 64 << 20:
            cap_memory((torch.cuda.memory_reserved() + n) / 2**30 + 1.0)            # (room for the block; the
                                                                                    # hold below caps it again)
            pool = torch.cuda.MemPool()
            try:
                with torch.cuda.use_mem_pool(pool):
                    self.slack = torch.empty(n, dtype=torch.uint8, device=self.dev)
                self.slack_pool = pool
            except torch.cuda.OutOfMemoryError:                      # another process took it meanwhile
                self.slack = None
        self._hold(others, quick=quick)                              # (the allocator's cap: what it holds, again)

    def _slack_out(self) -> None:
        """The slack block back to the driver (before a capture): its pool goes, and with it that block only -- the
        arena's blocks and any other cached ones stay held."""
        if self.slack is not None:
            self.slack = None
            torch.cuda.synchronize()
            self.slack_pool = None

    def _others(self, others: Optional[float]) -> float:
        """The other processes' GB: ``others`` if given, else measured -- else (nvidia-smi failed) the last value
        seen."""
        if others is None:
            try:
                others = others_gb(exclude_pid=os.getpid())
            except GpuCheckFailed as e:
                print(f"{e}; the other processes' memory taken as last seen ({self.others_last:.1f} GB)", flush=True)
                return self.others_last
        self.others_last = others
        return others

    def _save(self, cycle: int, sample: int, sha: str, order: str, keep: bool, background: bool, shape,
              run_id) -> None:
        """A checkpoint of the current state whose data progress is (cycle, sample)."""
        if not self._room(1):
            raise NeedsOperator(self._space_message(1))
        ms, os_ = snapshot(self.model, self.opt.state())
        c = self.cfg
        rows, self.rows = [r for r in self.rows if r[0] <= self.update], []
        meta = {"update": self.update, "tokens": self.tokens, "skipped": self.skipped,
                "skipped_in_a_row": self.in_a_row, "line": c.name, "run_dir": str(self.root), "rows": rows,
                "data": {"work": c.data_work, "cycle": cycle, "sample": sample, "manifest_sha256": sha,
                         "seed": c.data_seed, "order": ORDER_FORMAT, "order_digest": order},
                "shape": shape, "run": c.to_dict(), "model_config": self.model_cfg, "code": self.code,
                "wandb_run": run_id, "time": time.time()}
        self.ckpts.save(self.update, ms, os_, meta, keep=keep, background=background)
        self.last_ckpt = self.update

    def _checkpoint(self, u, keep: bool, background: bool = True) -> None:
        """A checkpoint after update ``u``."""
        shape = {"rows": self.engine.rows, "seq_len": self.engine.L, "accum": self.engine.accum}
        if u.cycle_end:
            self._save(u.cycle + 1, 0, "", "", keep, background, shape, self.run_id)
        else:
            self._save(u.cycle, u.next, u.manifest_sha256, u.order_digest, keep, background, shape, self.run_id)

    def _val_rows(self, L: int, unit: Optional[int] = None):
        """The validation rows of length L (nstrain.data.batches.val_rows), each stratum's a multiple of L's own
        micro_rows: the same rows whatever the reserve (a shared shape is re-tuned with it)."""
        if L not in self.val:
            self.val[L] = val_rows(self.cfg.val_work, L, self.cfg.validation.rows,
                                   self.cfg.length(L, False).micro_rows if unit is None else unit)
        return self.val[L]

    def _val_batch(self, L: int) -> int:
        """Validation micro-batch rows: the training shape's micro_rows when it divides L's own (the strata's row
        counts are multiples of that; the kernels' per-shape caches are then the training ones), else their greatest
        common divisor (its first call is warmed before the capture, ``_warm_validation``)."""
        return math.gcd(self.cfg.length(L, False).micro_rows, self.cfg.length(L, self.shared).micro_rows)

    @torch.no_grad()
    def _warm_validation(self, L: int) -> None:
        """One validation micro-batch, before an engine's capture (StepEngine's ``before_capture``): its kernels are
        loaded and its first-call caches are created while the arena's free blocks are not yet graph scratch."""
        mb = self._val_batch(L)
        for strata in self._val_rows(L).values():
            for _, _, arr in strata:
                probe, hprobe = DendriticProbe(self.eos_id), HippocampalProbe(self.eos_id, L)
                with self.arena.use(), probing(self.model, probe, hprobe):
                    w = torch.from_numpy(arr[:mb]).to(self.dev)
                    probe.row(w[:, :-1])
                    hprobe.row(w[:, :-1])
                    self.model.loss(w[:, :-1].contiguous(), w[:, 1:].contiguous())
                    del w
                print(f"{dendritic_line(probe)}; {hippocampal_line(hprobe)} (the warm-up micro-batch of length {L})",
                      flush=True)
                if self.probe_len > L:                                # the long-row probe's first call too
                    self._warm_long(L)
                return

    @torch.no_grad()
    def _warm_long(self, seg: int) -> None:
        """One long row (``_probe_long``'s length, segments of ``seg``) before a capture: its kernels' first-call
        caches and its memory while the arena's free blocks are not yet graph scratch, and a reading of the model at
        this start."""
        P = self.probe_len
        if P % seg:
            raise RuntimeError(f"long probe: the length {P} is not a multiple of the training length {seg}")
        arr = next(iter(self._val_rows(P, unit=1).values()))[0][2]
        probe, hprobe = DendriticProbe(self.eos_id, horizon=P, segments=P // seg), HippocampalProbe(self.eos_id, seg)
        with self.arena.use(), probing(self.model, probe, hprobe):
            w = torch.from_numpy(arr[:1]).to(self.dev)
            probe.row(w[:, :-1])
            hprobe.row(w[:, :-1])
            per = self.model.per_token_ce(w[:, :-1].contiguous(), w[:, 1:].contiguous())
            segs = [round(float(x), 3) for x in per.view(-1, seg).mean(dim=1)]
            del w, per
        print(f"long probe: one row of {P}, ce by segment of {seg} {segs}; {dendritic_line(probe)}; "
              f"{hippocampal_line(hprobe)}", flush=True)

    def _hold_all(self) -> None:
        """The GPU to itself: the arena holds all of the GPU's free memory but a capture's slack (CAPTURE_SLACK_GIB,
        for the memory outside the allocator that the capture creates), in large blocks (nstrain.gpu.hold_memory
        with nothing left to others; what the driver keeps from dead processes is reclaimed too), so that every
        allocation of the engine's warm-up and capture and of the validation rows of any length is served from them.
        An arena that took its blocks from the driver as they were asked for holds none larger than the largest
        allocation so far: once it held the GPU, a larger one -- the long rows' (a 32K row's copy scores are 4 GiB,
        a 4K micro-batch's a few hundred MiB) -- failed."""
        torch.cuda.empty_cache()                                     # blocks cached outside the arena (a build's, a
        taken = hold_memory(0.0, 0.0, lambda: 0.0, slack_gib=CAPTURE_SLACK_GIB, arena=self.arena, reclaim=True)
                                                                     # load's): the arena cannot use them, it takes
                                                                     # them from the driver as its own blocks
        print(f"update {self.update}: holding {taken:.1f} GiB ({torch.cuda.memory_reserved() / 2**30:.1f} GiB "
              f"reserved, {torch.cuda.mem_get_info()[0] / 2**30:.1f} GiB free): the GPU to itself", flush=True)

    def _hold(self, others: float, slack_gib: float = 0.25, reclaim: bool = False, quick: bool = False) -> None:
        """Beside other processes: the arena holds the GPU's memory but what they may still take, and ``slack_gib``
        for this process's own memory outside the allocator (nstrain.gpu.hold_memory: their memory read again after
        the free memory, so that what they free meanwhile is not taken -- not with ``quick``, the switch's hold, which
        takes ``others`` as read; with ``reclaim``, at a start or a capture, what the driver keeps from dead
        processes is taken too)."""
        taken = hold_memory(self.cfg.guards.share_reserve_gb, others,
                            (lambda: others) if quick else (lambda: others_gb(exclude_pid=os.getpid())),
                            slack_gib=slack_gib, arena=self.arena, reclaim=reclaim)
        if taken >= 0.5:
            print(f"update {self.update}: holding {taken:.1f} GiB more ({torch.cuda.memory_reserved() / 2**30:.1f} GiB "
                  f"reserved); the other processes use {others:.1f} of the {self.cfg.guards.share_reserve_gb:g} GB "
                  "reserved for them", flush=True)

    @torch.no_grad()
    def _validate(self, L: int, recipe: dict) -> dict:
        """The held-out loss of every source at length L (``val_<L>/<source>``: its strata's losses weighted by their
        shares of the source), and over the sources of the update's recipe ``recipe`` ({source: weight}): weighted
        by the recipe (``val_<L>/mix``, the held-out estimate of the training loss) and unweighted (``.../mean``).
        The dendritic neuron layers are probed on the same rows (``val_<L>/dendritic/...``, nstrain.probes)."""
        key = f"val_{L}"
        out, got = {}, {}
        mb = self._val_batch(L)
        probe, hprobe = DendriticProbe(self.eos_id), HippocampalProbe(self.eos_id, L)
        with probing(self.model, probe, hprobe):
            for name, strata in self._val_rows(L).items():
                self._beat(f"validation {name}")
                acc = tot = 0.0
                for _, share, arr in strata:
                    s = 0.0
                    with self.arena.use():                           # (in the arena: memory the trainer holds)
                        for i in range(0, arr.shape[0], mb):
                            w = torch.from_numpy(arr[i:i + mb]).to(self.dev)
                            probe.row(w[:, :-1])
                            hprobe.row(w[:, :-1])
                            s += float(self.model.loss(w[:, :-1].contiguous(), w[:, 1:].contiguous()).ce)
                            del w
                    acc += share * s / (arr.shape[0] // mb)
                    tot += share
                got[name] = out[f"{key}/{name}"] = acc / tot
        mine = {n: float(w) for n, w in recipe.items() if w > 0}
        missing = sorted(n for n in mine if n not in got)
        if missing:
            raise RuntimeError(f"validation: no held-out rows of the recipe's sources {missing} at length {L}")
        out[f"{key}/mix"] = sum(got[n] * w for n, w in mine.items()) / sum(mine.values())
        out[f"{key}/mean"] = sum(got[n] for n in mine) / len(mine)
        out.update(dendritic_metrics(key, probe))
        out.update(hippocampal_metrics(key, hprobe))
        return out

    @torch.no_grad()
    def _probe_long(self, seg: int, recipe: dict) -> dict:
        """The long-row probe: every source's held-out rows of the model's longest context ``self.probe_len``
        (``val_<P>/<source>``, ``/mix``, ``/mean`` as ``_validate``, from the per-token cross entropy, one row at a
        time), the loss by position in segments of ``seg`` tokens (the training length): ``val_<P>/pos/seg<k>``
        weighted by the recipe and ``.../last_over_first``; the dendritic probe over the long rows, with the firing
        rate by segment (``val_<P>/dendritic/...``, nstrain.probes)."""
        P, key = self.probe_len, f"val_{self.probe_len}"
        if P % seg:
            raise RuntimeError(f"long probe: the length {P} is not a multiple of the training length {seg}")
        n_seg = P // seg
        probe, hprobe = DendriticProbe(self.eos_id, horizon=P, segments=n_seg), HippocampalProbe(self.eos_id, seg)
        out, got, pos = {}, {}, {}
        with probing(self.model, probe, hprobe):
            for name, strata in self._val_rows(P, unit=1).items():
                self._beat(f"long probe {name}")
                acc, tot = torch.zeros(n_seg, dtype=torch.float64), 0.0
                for _, share, arr in strata:
                    s = torch.zeros(n_seg, dtype=torch.float64)
                    with self.arena.use():                           # (in the arena: memory the trainer holds)
                        for i in range(arr.shape[0]):
                            w = torch.from_numpy(arr[i:i + 1]).to(self.dev)
                            probe.row(w[:, :-1])
                            hprobe.row(w[:, :-1])
                            per = self.model.per_token_ce(w[:, :-1].contiguous(), w[:, 1:].contiguous())
                            s += per.view(n_seg, seg).mean(dim=1).double().cpu()
                            del w, per
                    acc += share * s / arr.shape[0]
                    tot += share
                pos[name] = acc / tot                                # (n_seg,) the source's loss by segment
                got[name] = out[f"{key}/{name}"] = float(pos[name].mean())
        mine = {n: float(w) for n, w in recipe.items() if w > 0}
        missing = sorted(n for n in mine if n not in got)
        if missing:
            raise RuntimeError(f"long probe: no held-out rows of the recipe's sources {missing} at length {P}")
        wsum = sum(mine.values())
        out[f"{key}/mix"] = sum(got[n] * w for n, w in mine.items()) / wsum
        out[f"{key}/mean"] = sum(got[n] for n in mine) / len(mine)
        mix = sum(pos[n] * w for n, w in mine.items()) / wsum
        for k in range(n_seg):
            out[f"{key}/pos/seg{k}"] = float(mix[k])
        out[f"{key}/pos/last_over_first"] = float(mix[-1] / mix[0])
        out.update(dendritic_metrics(key, probe))
        out.update(hippocampal_metrics(key, hprobe))
        return out

    # ------------------------------------------------------------------ the loop
    def _stage_end(self, u) -> None:
        """The sequence length changes before update ``u``: the state after the last update of the old length is a
        kept checkpoint (the one written at that update promoted, or a new one)."""
        if self.last_ckpt == self.update:
            self.ckpts.wait()
            self.ckpts.keep(self.update)
        else:
            self._save(u.cycle, u.first, u.manifest_sha256, u.order_digest, keep=True, background=False,
                       shape=self.prev_shape, run_id=self.run_id)
        print(f"update {self.update}: stage end ({self.prev_shape['seq_len']} -> {u.seq_len}), kept checkpoint",
              flush=True)

    def _mix_check(self) -> dict:
        """The strata of the samples read since the last check against the cycles' plans: the largest deviation in
        standard deviations (``data/mix_z``; binomial, a random order keeps it under ~5).  Past MIX_ALARM_Z the data
        are not mixed as planned: a W&B alert, and the line stops with a checkpoint for the operator (a defect in
        the data's order costs every update trained on it)."""
        worst, name = 0.0, None
        tot: Dict[str, float] = {}
        for (src, _), (_, e) in self.mix.items():
            tot[src] = tot.get(src, 0.0) + e
        for key, (o, e) in self.mix.items():
            p = e / tot[key[0]] if tot.get(key[0]) else 0.0
            sd = math.sqrt(max(e * (1 - p), 1e-9))
            z = abs(o - e) / sd if e >= 25 else 0.0                  # (too few samples to judge)
            if z > worst:
                worst, name = z, key
        self.mix = {}
        if worst > MIX_ALARM_Z:
            msg = (f"update {self.update}: the samples of {name[0]}/{name[1] or '(flat)'} deviate {worst:.1f} standard "
                   "deviations from the cycle's plan since the last check: the data are not mixed as planned")
            print(msg, flush=True)
            self.wb.alert("data mix", msg)
            self.mix_alarm = msg
        return {"data/mix_z": worst}

    def _gpu_check(self, row: dict):
        """(other processes on the GPU, their memory in GB beside them, else 0), or None when the check failed; a
        failed check counts, three in a row raise."""
        try:
            busy = other_processes(exclude_pid=os.getpid())
            gb = others_gb(exclude_pid=os.getpid()) if busy and self.reserve is not None else 0.0
            if self.shared:
                self.others_last = gb
        except GpuCheckFailed as e:
            self.gpu_check_fails += 1
            row["sys/gpu_check_failed"] = self.gpu_check_fails
            print(f"update {self.update}: {e} ({self.gpu_check_fails} in a row)", flush=True)
            if self.gpu_check_fails >= 3:
                raise
            return None
        self.gpu_check_fails = 0
        self.others_seen = len(busy)                                 # (in the heartbeat: the supervisor's crash call)
        row["sys/other_gpu_processes"] = len(busy)
        if busy and self.shared:
            row["sys/other_gpu_gb"] = gb
        return busy, gb

    def run(self) -> int:
        if self.finished:
            self._honor_stop()
            return 0
        c = self.cfg
        ck_every, val_every = c.checkpoints.every_tokens, c.validation.every_tokens
        stop_at = c.stop_at_tokens
        next_ck = (math.floor(self.tokens / ck_every) + 1) * ck_every
        next_val = (math.floor(self.tokens / val_every) + 1) * val_every
        diverged = False
        self.gpu_check_fails = 0
        u = self.first
        last_u = recipe = None
        last_t = time.time()
        rc = 0
        while u is not None:
            if self.prev_shape is not None and self.prev_shape["seq_len"] != u.seq_len:
                self._stage_end(u)
            if stop_at is not None and self.tokens >= stop_at:
                print(f"update {self.update}: stop_at_tokens {stop_at:.0f} reached; no further update", flush=True)
                break
            if self._stop_requested(u, before=True):
                print(f"update {self.update}: stop requested; no further update", flush=True)
                break
            if u.recipe is not recipe:                               # a new segment: its sources validate (else
                missing = sorted(set(u.recipe) - set(self._val_rows(u.seq_len)))       # an error before training)
                if missing:
                    raise RuntimeError(f"update {self.update + 1}: the recipe's sources {missing} have no held-out "
                                       f"rows of {u.seq_len} in {c.val_work} (`nstrain data val`)")
                recipe = u.recipe
            try:
                eng = self._engine_for(u.seq_len)
            except torch.cuda.OutOfMemoryError:
                if not self.shared:
                    raise
                print(f"update {self.update}: out of memory capturing the {u.seq_len} engine beside the other "
                      "processes (they took more than the reserve meanwhile); checkpoint and stop until they fit",
                      flush=True)
                rc = exits.GPU_BUSY
                break
            lr = sched.lr_at(c.schedule, c.optimizer.peak_lr, self.tokens)
            metrics, applied = eng.update(u.inputs, u.targets, lr)
            self.update += 1
            self.tokens += u.inputs.numel()
            self.skipped += 0 if applied else 1
            self.in_a_row = 0 if applied else self.in_a_row + 1
            self.prev_shape = {"rows": eng.rows, "seq_len": eng.L, "accum": eng.accum}
            eng = None                                               # (no reference left on the engine: a switch of
            last_u = u                                               # mode below frees it with the arena)
            now = time.time()
            row = {"tokens": self.tokens, "train/loss": metrics["loss"], "train/ce": metrics["ce"],
                   "train/aux": metrics["aux"], "train/grad_norm": metrics["grad_norm"], "train/lr": lr,
                   **{f"train/grad_norm/{g}": v for g, v in metrics["grad_norms"].items()},
                   "train/seq_len": u.seq_len, "train/skipped": self.skipped, "data/cycle": u.cycle,
                   "data/sample": u.next, "perf/update_s": now - last_t,
                   "perf/tok_per_s": u.inputs.numel() / max(now - last_t, 1e-9)}
            last_t = now
            if not diverged and not self.wb.check_replay(self.update, row, shared=int(self.shared)):
                diverged = True                                      # the line stays in its run (nstrain.wb)
                print(f"update {self.update}: replay differs from the server's copy (recomputed with another "
                      "micro-batch shape, or not the same computation); the server keeps its rows and the run "
                      "continues after its last step", flush=True)
            for name, (o, e) in u.mix.items():                      # the samples read against the plan
                a, b = self.mix.get(name, (0, 0.0))
                self.mix[name] = (a + o, b + e)
            validated = self.tokens >= next_val
            if validated:
                row.update(self._validate(u.seq_len, u.recipe))
                if self.probe_len > u.seq_len:
                    row.update(self._probe_long(u.seq_len, u.recipe))
                row.update(self._mix_check())
                next_val += val_every
            seen, check_failed = None, False                         # (other processes, their GB) when checked
            if self.update % c.guards.gpu_check_every == 0 and not c.guards.allow_shared_gpu:
                try:
                    seen = self._gpu_check(row)
                except GpuCheckFailed:
                    check_failed = True
            busy = seen[0] if seen else []
            evict = bool(busy) and not self.shared and self.reserve is None
            if seen is not None and self.reserve is not None:
                if busy and not self.shared:                         # they appeared: beside them from here, if
                    if fits_beside(seen[1], self.reserve):           # their memory leaves room for the shapes
                        self._switch(True, seen[1])
                    else:
                        evict = True
                elif self.shared and not busy:                       # they left: alone again after ALONE_CHECKS
                    self.alone_checks += 1                           # checks without them (not for a gap of one)
                    if self.alone_checks >= ALONE_CHECKS:
                        self._switch(False, 0.0)
                else:
                    self.alone_checks = 0
                if self.shared and busy:                             # the others get the reserve, no more (and
                    self._hold(seen[1])                              # what they free is held again)
            row["sys/gpu_shared"] = int(self.shared)
            self.wb.log(self.update, row)
            self.rows.append((self.update, row))
            self._beat("update")
            self.ckpts.poll()                                        # a failed background checkpoint stops here
            if validated or self.update % c.print_every == 0:          # a validation's row always reaches run.log
                print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}),
                      flush=True)
            if self.in_a_row > c.guards.max_skipped_in_a_row:
                raise RuntimeError(f"{self.in_a_row} non-finite updates in a row (last: update {self.update}); the "
                                   f"line stops without a checkpoint past update {self.last_ckpt}")
            stop = (stop_at is not None and self.tokens >= stop_at) or self._stop_requested(u)
            if check_failed and not stop:
                print(f"update {self.update}: the GPU cannot be checked; checkpoint and stop (failure)", flush=True)
                rc = exits.FAILURE
            elif evict and not stop:
                why = ("no guards.share_reserve_gb: the line trains with the GPU to itself" if self.reserve is None
                       else f"they use {seen[1]:.1f} GB, more than the {self.reserve:g} GB reserved for them and the "
                            "shapes' margin")
                print(f"update {self.update}: other processes on the GPU: {busy} ({why}); checkpoint and stop",
                      flush=True)
                rc = exits.GPU_BUSY
            elif self.mix_alarm and not stop:
                print(f"update {self.update}: {self.mix_alarm}; checkpoint and stop for the operator", flush=True)
                rc = exits.NEEDS_OPERATOR
            elif not self._room(2) and not stop:
                print(f"update {self.update}: {self._space_message(2)}; checkpoint and stop", flush=True)
                rc = exits.NEEDS_OPERATOR
            done = stop or rc != 0
            if u.cycle_end:
                self._checkpoint(u, keep=True)
                self.ckpts.wait()
                self.line.release(u.cycle)
                self._beat("cycle-end checkpoint written")
            elif self.tokens >= next_ck or done:
                self._checkpoint(u, keep=False, background=not done)
            while self.tokens >= next_ck:
                next_ck += ck_every
            if done:
                break
            u = self._next_update(u.cycle + 1 if u.cycle_end else u.cycle)
        if last_u is not None and self.last_ckpt != self.update:     # stopped while waiting for data
            self._checkpoint(last_u, keep=False, background=False)
        self.ckpts.wait()
        self.wb.finish(exit_code=rc)
        self._honor_stop()
        return rc

    def _honor_stop(self) -> None:
        """The STOP file renamed when it stopped the run; a stop by a signal said in the last heartbeat (the
        supervisor tells it from the end of the line, nstrain.supervise)."""
        stop = self.root / "STOP"
        if self.stop_honored and stop.exists():
            honor_stop(self.root, self.update)
        if self.stop:
            self._beat(exits.SIGNAL_STOP)


def main(run) -> int:
    try:
        return Trainer(run).run()
    except GpuBusy as e:
        print(f"trainer: {e}", flush=True)
        return exits.GPU_BUSY
    except LineBusy as e:
        print(f"trainer: {e}", flush=True)
        return exits.LINE_BUSY
    except NeedsOperator as e:
        print(f"trainer: {e}", flush=True)
        return exits.NEEDS_OPERATOR
