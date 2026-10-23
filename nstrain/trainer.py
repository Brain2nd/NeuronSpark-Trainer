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
another process uses the GPU (beside other processes: they use more than ``guards.share_reserve_gb``), 4 another
trainer holds the line, 5 not enough disk space for the next checkpoints (checked after every update; a checkpoint is
written while there is room), 6 beside other processes, and they left (relaunch with the GPU to itself), 1 failure
(also when nvidia-smi fails: at start, or at three checks in a row, then after a checkpoint).  The GPU is checked every
``guards.gpu_check_every`` updates; a change of the GPU's occupancy ends the run with a checkpoint.  Beside other
processes (``$NSTRAIN_GPU_MODE=shared``, set by the supervisor, see nstrain.gpu) every length trains with its
``shared`` shape.  Started by a supervisor (``NSTRAIN_SUPERVISOR_PID``), the trainer gets SIGTERM when the supervisor
dies.
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
import sys
import time
from pathlib import Path
from typing import List

import torch

from nstrain import config as C
from nstrain import exits
from nstrain import model as M
from nstrain import schedule as sched
from nstrain.checks import preflight, start_problems
from nstrain.ckpt import Checkpoints, snapshot
from nstrain.data.batches import Line, Prefetch, val_rows
from nstrain.engine import StepEngine
from nstrain.flat_adamw import FlatAdamW
from nstrain.gpu import (GPU_MODE, GpuBusy, GpuCheckFailed, GpuFree, beside_limit_gib, cap_memory, lease,
                         other_processes, others_gb)
from nstrain.wb import RunRecord, Tracker


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
        self._beat("start")
        self.shared = os.environ.get(GPU_MODE) == "shared"
        if self.shared and c.guards.share_reserve_gb is None:
            raise RuntimeError(f"${GPU_MODE}=shared needs guards.share_reserve_gb")
        if not c.guards.allow_shared_gpu:                           # before any CUDA initialisation
            self._lease = lease(str(self.root))
            others = other_processes(exclude_pid=os.getpid())
            if self.shared and not others:
                raise GpuFree("no other process on the GPU: the line trains with the GPU to itself")
            if self.shared and others_gb(exclude_pid=os.getpid()) > c.guards.share_reserve_gb:
                raise GpuBusy(f"other processes on the GPU use more than the {c.guards.share_reserve_gb:g} GB "
                              f"reserved for them: {others}")
            if others and not self.shared:
                raise GpuBusy(f"other processes on the GPU: {others}; training needs the GPU to itself "
                              "(guards.allow_shared_gpu overrides)")
        problems = preflight(c)
        if problems:
            raise RuntimeError("cannot start: " + "; ".join(problems))
        if self.shared:                                              # out of memory before the reserve is touched
            total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 2**30
            cap_memory(beside_limit_gib(total, c.guards.share_reserve_gb))
        self.model_cfg = c.model
        self.ckpts = Checkpoints(c.ckpt_dir, keep_last=c.checkpoints.keep_last)
        self.line = Line(c.data_work, c.tokens_per_update, seed=c.data_seed)
        self.data_state = json.loads((Path(c.data_work) / "state.json").read_text())
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
                raise RuntimeError("cannot start: " + "; ".join(problems) + "  (allow_changes in run.json lists "
                                   "keys that may change on purpose)")
            self.model = M.build(self.model_cfg, self.dev, init="empty")
            self.model.load_state_dict({k: v.to(self.dev) for k, v in ms.items()}, strict=True)
            self.update, self.tokens = int(meta["update"]), int(meta["tokens"])
            self.skipped = int(meta.get("skipped", 0))
            self.in_a_row = int(meta.get("skipped_in_a_row", 0))
            if latest is not None:                                   # continue this line
                self.progress = (int(meta["data"]["cycle"]), int(meta["data"]["sample"]))
                self.expect_sha = meta["data"]["manifest_sha256"]
                self.last_ckpt = self.update
                self.prev_shape = meta.get("shape")
            else:                                                    # a branch starts at its own cycle 0
                self.progress, self.expect_sha = (0, 0), ""
        else:
            meta, opt_state = None, None
            torch.manual_seed(c.seed)
            self.model = M.build(self.model_cfg, self.dev, init="launch")
            self.update = self.tokens = self.skipped = 0
            self.progress, self.expect_sha = (0, 0), ""
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
        self.line.mark(self.progress[0])
        self.updates = Prefetch(self.line.updates(*self.progress, expect_sha256=self.expect_sha))
        self.first = self._next_update()
        if self.first is None:
            if self._stop_requested():
                print(f"line {c.name}: stopped before its first update here (update {self.update})", flush=True)
                self.finished = True
                return
            if latest is not None:
                print(f"line {c.name}: its data is exhausted (update {self.update}); nothing to train", flush=True)
                self.finished = True
                return
            raise RuntimeError("no data on this line")
        if src is None:
            M.calibrate(self.model, self.first.inputs, c.length(self.first.seq_len).micro_rows, self.dev)
            self._beat("calibrated")
            self._save(self.first.cycle, self.first.first, self.first.manifest_sha256, keep=True, background=False,
                       shape=None, run_id=None)                      # the launch state of the line
            self.last_ckpt = 0

        wcfg = dict(c.to_dict(), run_dir=str(self.root), model_config=self.model_cfg, code=self.code)
        self.wb = Tracker(c.wandb.project, c.name, wcfg, c.wandb.entity, retry_s=c.wandb.retry_s,
                          on_wait=lambda: self._beat("wandb"), local_dir=self.root)
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

    def _stop_requested(self, u=None, before: bool = False) -> bool:
        """A stop by signal, or by the STOP file: unconditional, or {"at_cycle_end": k}, due after the last update
        of cycle k (``before``: before an update of a later cycle)."""
        s = self._stop_file()
        due = False
        if s is not None:
            k = s.get("at_cycle_end")
            due = k is None or (u is not None and (u.cycle > k if before else
                                                    (u.cycle > k or (u.cycle == k and u.cycle_end))))
            self.stop_honored = self.stop_honored or due
        return self.stop or due

    def _room(self, n: int) -> bool:
        return shutil.disk_usage(self.root).free >= n * 1.1 * self.ckpt_bytes + 10e9

    def _space_message(self, n: int) -> str:
        return (f"{shutil.disk_usage(self.root).free / 1e9:.0f} GB free on {self.root}, {n} checkpoint(s) need "
                f"{(n * 1.1 * self.ckpt_bytes + 10e9) / 1e9:.0f} GB: free space, then continue the line")

    def _beat(self, stage: str) -> None:
        tmp = self.root / "heartbeat.tmp"
        tmp.write_text(json.dumps({"time": time.time(), "update": self.update, "stage": stage, "pid": os.getpid()}))
        os.replace(tmp, self.root / "heartbeat")

    def _rollback_marker(self, latest):
        """rollback.json (``nstrain rollback``): {"to": update, "source_run": the W&B run it supersedes}."""
        p = self.root / "rollback.json"
        if not p.exists():
            return None
        m = json.loads(p.read_text())
        if latest is None or latest.name != f"u{int(m['to']):08d}":
            raise RuntimeError(f"{p}: a rollback to update {m['to']}, but `latest` is {latest}; re-run "
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

    def _next_update(self):
        """The next update; None at the end of the line or when a stop is requested while the preparer is behind."""
        waited = 0
        while True:
            u = self.updates.get(timeout=60.0)
            if u is not Prefetch.WAITING:
                return u
            if self._stop_requested():
                print(f"update {self.update}: stop requested while waiting for prepared data", flush=True)
                return None
            waited += 1
            self._beat("waiting for data")
            if waited % 10 == 0:
                print(f"update {self.update}: waiting for prepared data ({waited} min)", flush=True)

    def _open_run(self, meta, forking: bool) -> str:
        """The line's W&B run (see nstrain.wb), continued from wandb_run.json or created; the id is recorded before
        the run is created and marked ready after its backfill, so a restart at any point completes the same run.  A
        rollback marker makes a new run (once: a restart finds the rollback's run in the record and continues it)."""
        rec = self.rec.load()
        if rec is not None and not rec["ready"]:                     # a creation / backfill cut short: finish it
            self.wb.create(rec["run_id"], self._run_config(rec["kind"], rec["source"], rec["source_step"]))
            if rec["source"] is not None:
                self.wb.backfill(rec["source"], rec["source_step"])
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
        if kind == "rollback":
            self.wb.supersede(source, step)
        self.rec.save(dict(rec, ready=True))
        return rid

    @staticmethod
    def _run_config(kind: str, source, step) -> dict:
        """The W&B config entries of a run's origin."""
        return {"rollback": {"rollback_of": source, "rollback_step": step},
                "branch": {"parent_run": source, "parent_step": step}}.get(kind, {})

    def _engine_for(self, L: int) -> StepEngine:
        if self.engine is not None and self.engine.L == L:
            return self.engine
        if self.engine is not None:
            self.engine.release()
            self.engine = None
            torch.cuda.empty_cache()
        c = self.cfg
        self._beat(f"capturing the {L} engine")
        s, accum = c.length(L, self.shared), c.accum(L, self.shared)
        M.recompute(self.model, s.activation_checkpointing)
        t0 = time.time()
        self.engine = StepEngine(self.model, self.opt, s.micro_rows, L, accum, c.optimizer.grad_clip)
        print(f"engine {accum} x {s.micro_rows} x {L}{' (beside other processes)' if self.shared else ''} captured "
              f"in {time.time() - t0:.0f} s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
        self._beat(f"{L} engine captured")
        return self.engine

    def _save(self, cycle: int, sample: int, sha: str, keep: bool, background: bool, shape, run_id) -> None:
        """A checkpoint of the current state whose data progress is (cycle, sample)."""
        if not self._room(1):
            raise NeedsOperator(self._space_message(1))
        ms, os_ = snapshot(self.model, self.opt.state())
        c = self.cfg
        rows, self.rows = [r for r in self.rows if r[0] <= self.update], []
        meta = {"update": self.update, "tokens": self.tokens, "skipped": self.skipped,
                "skipped_in_a_row": self.in_a_row, "line": c.name, "run_dir": str(self.root), "rows": rows,
                "data": {"work": c.data_work, "cycle": cycle, "sample": sample, "manifest_sha256": sha,
                         "seed": c.data_seed},
                "shape": shape, "run": c.to_dict(), "model_config": self.model_cfg, "code": self.code,
                "wandb_run": run_id, "time": time.time()}
        self.ckpts.save(self.update, ms, os_, meta, keep=keep, background=background)
        self.last_ckpt = self.update

    def _checkpoint(self, u, keep: bool, background: bool = True) -> None:
        """A checkpoint after update ``u``."""
        shape = {"rows": self.engine.rows, "seq_len": self.engine.L, "accum": self.engine.accum}
        if u.cycle_end:
            self._save(u.cycle + 1, 0, "", keep, background, shape, self.run_id)
        else:
            self._save(u.cycle, u.next, u.manifest_sha256, keep, background, shape, self.run_id)

    @torch.no_grad()
    def _validate(self, L: int) -> dict:
        rows = self.cfg.validation.rows
        if L not in self.val:
            self.val[L] = val_rows(self.cfg.val_work, L, rows)
        out, tot, n = {}, 0.0, 0
        mb = self.cfg.length(L, self.shared).micro_rows
        for name, arr in self.val[L].items():
            self._beat(f"validation {name}")
            s, k = 0.0, 0
            for i in range(0, arr.shape[0], mb):
                w = torch.from_numpy(arr[i:i + mb]).to(self.dev)
                if w.shape[0] < mb:
                    break
                s += float(self.model.loss(w[:, :-1].contiguous(), w[:, 1:].contiguous()).ce)
                k += 1
            if k:
                out[f"val/{name}"] = s / k
                tot, n = tot + s / k, n + 1
        if n:
            out["val/mean"] = tot / n
        return out

    # ------------------------------------------------------------------ the loop
    def _stage_end(self, u) -> None:
        """The sequence length changes before update ``u``: the state after the last update of the old length is a
        kept checkpoint (the one written at that update promoted, or a new one)."""
        if self.last_ckpt == self.update:
            self.ckpts.wait()
            self.ckpts.keep(self.update)
        else:
            self._save(u.cycle, u.first, u.manifest_sha256, keep=True, background=False, shape=self.prev_shape,
                       run_id=self.run_id)
        print(f"update {self.update}: stage end ({self.prev_shape['seq_len']} -> {u.seq_len}), kept checkpoint",
              flush=True)

    def _gpu_check(self, row: dict):
        """(other processes on the GPU, their memory in GB beside them, else 0), or None when the check failed; a
        failed check counts, three in a row raise."""
        try:
            busy = other_processes(exclude_pid=os.getpid())
            gb = others_gb(exclude_pid=os.getpid()) if self.shared and busy else 0.0
        except GpuCheckFailed as e:
            self.gpu_check_fails += 1
            row["sys/gpu_check_failed"] = self.gpu_check_fails
            print(f"update {self.update}: {e} ({self.gpu_check_fails} in a row)", flush=True)
            if self.gpu_check_fails >= 3:
                raise
            return None
        self.gpu_check_fails = 0
        row["sys/other_gpu_processes"] = len(busy)
        if self.shared:
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
        last_u = None
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
            eng = self._engine_for(u.seq_len)
            lr = sched.lr_at(c.schedule, c.optimizer.peak_lr, self.tokens)
            metrics, applied = eng.update(u.inputs, u.targets, lr)
            self.update += 1
            self.tokens += u.inputs.numel()
            self.skipped += 0 if applied else 1
            self.in_a_row = 0 if applied else self.in_a_row + 1
            self.prev_shape = {"rows": eng.rows, "seq_len": eng.L, "accum": eng.accum}
            last_u = u
            now = time.time()
            row = {"tokens": self.tokens, "train/loss": metrics["loss"], "train/ce": metrics["ce"],
                   "train/aux": metrics["aux"], "train/grad_norm": metrics["grad_norm"], "train/lr": lr,
                   "train/seq_len": u.seq_len, "train/skipped": self.skipped, "data/cycle": u.cycle,
                   "data/sample": u.next, "perf/update_s": now - last_t,
                   "perf/tok_per_s": u.inputs.numel() / max(now - last_t, 1e-9)}
            last_t = now
            if not diverged and not self.wb.check_replay(self.update, row, shared=int(self.shared)):
                diverged = True                                      # resumed run is not the same computation
                print(f"update {self.update}: replay differs from the server's copy; switching to a new run",
                      flush=True)
                old = self.run_id
                self.wb.finish()
                self.run_id = self._new_run("rollback", old, self.update - 1)
            if self.tokens >= next_val:
                row.update(self._validate(u.seq_len))
                next_val += val_every
            seen, check_failed = None, False                         # (other processes, their GB) when checked
            if self.update % c.guards.gpu_check_every == 0 and not c.guards.allow_shared_gpu:
                try:
                    seen = self._gpu_check(row)
                except GpuCheckFailed:
                    check_failed = True
            busy = seen[0] if seen else []
            left = self.shared and seen is not None and not busy
            crowded = self.shared and seen is not None and seen[1] > c.guards.share_reserve_gb
            row["sys/gpu_shared"] = int(self.shared)
            self.wb.log(self.update, row)
            self.rows.append((self.update, row))
            self._beat("update")
            self.ckpts.poll()                                        # a failed background checkpoint stops here
            if self.update % c.print_every == 0:
                print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}),
                      flush=True)
            if self.in_a_row > c.guards.max_skipped_in_a_row:
                raise RuntimeError(f"{self.in_a_row} non-finite updates in a row (last: update {self.update}); the "
                                   f"line stops without a checkpoint past update {self.last_ckpt}")
            stop = (stop_at is not None and self.tokens >= stop_at) or self._stop_requested(u)
            if check_failed and not stop:
                print(f"update {self.update}: the GPU cannot be checked; checkpoint and stop (failure)", flush=True)
                rc = exits.FAILURE
            elif busy and not self.shared and not stop:
                print(f"update {self.update}: other processes on the GPU: {busy}; checkpoint and stop", flush=True)
                rc = exits.GPU_BUSY
            elif crowded and not stop:
                print(f"update {self.update}: the other processes on the GPU use {seen[1]:.1f} GB, more than the "
                      f"{c.guards.share_reserve_gb:g} GB reserved for them; checkpoint and stop", flush=True)
                rc = exits.GPU_BUSY
            elif left and not stop:
                print(f"update {self.update}: the other processes left the GPU; checkpoint and continue with the GPU "
                      "to itself", flush=True)
                rc = exits.GPU_FREE
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
            u = self._next_update()
        if last_u is not None and self.last_ckpt != self.update:     # stopped while waiting for data
            self._checkpoint(last_u, keep=False, background=False)
        self.ckpts.wait()
        self.wb.finish(exit_code=rc)
        self._honor_stop()
        return rc

    def _honor_stop(self) -> None:
        stop = self.root / "STOP"
        if self.stop_honored and stop.exists():
            os.replace(stop, self.root / f"STOP.honored-{self.update}")


def main(run) -> int:
    try:
        return Trainer(run).run()
    except GpuBusy as e:
        print(f"trainer: {e}", flush=True)
        return exits.GPU_BUSY
    except GpuFree as e:
        print(f"trainer: {e}", flush=True)
        return exits.GPU_FREE
    except LineBusy as e:
        print(f"trainer: {e}", flush=True)
        return exits.LINE_BUSY
    except NeedsOperator as e:
        print(f"trainer: {e}", flush=True)
        return exits.NEEDS_OPERATOR
