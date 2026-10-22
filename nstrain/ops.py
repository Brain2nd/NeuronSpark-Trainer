"""Operations on training lines: create (``new``, ``branch``), check, stop, roll back.

``new`` writes a trunk-type line from a model preset: every value resolved into run.json (the model configuration
too), so a later change of a preset or of a default in the code never changes the line.

``branch`` creates a branch from the parent's kept checkpoint at the end of its cycle K in one step: the run
configuration is the parent's at that checkpoint (its FROZEN settings, its schedule up to the fork, plus a decay over
the branch's data), the data line is forked from the parent's state after cycle K.  Order: the branch's run.json is
written as ``run.json.pending``, the data is forked (its state.json is written last), then run.json is put in place
and the branch recorded in the parent's ``branches.json``; running the same command again completes an interrupted
one.

``rollback`` (offline: no supervisor or trainer may hold the line): ``rollback.json`` first (which update, which W&B
run it supersedes), then ``latest`` -> the target, then the later checkpoints moved into ``rolled-back-<time>/``
(removed once the line writes a checkpoint after the rollback), then the data line's cycle in training set to the
target's (the preparer rebuilds it if it was released).  Running it again completes an interrupted rollback.  A
checkpoint that is the fork point of a branch (``branches.json``) is not moved unless ``force``.  The next start
continues from the target as a new W&B run (nstrain.trainer).
"""
from __future__ import annotations

import fcntl
import json
import os
import signal
import time
from contextlib import ExitStack
from pathlib import Path
from typing import List, Optional

from nstrain import config as C
from nstrain import model as M
from nstrain import schedule as sched
from nstrain.ckpt import Checkpoints
from nstrain.data import mixture
from nstrain.data.work import Work, _write_atomic


def _held(lock: Path) -> bool:
    """Another process holds this lock file."""
    if not lock.exists():
        return False
    with open(lock, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(f, fcntl.LOCK_UN)
        return False


def _runs_line(pid: int, cfg: C.RunConfig) -> bool:
    """Process ``pid`` is an nstrain command on this line (its arguments resolved against its working directory)."""
    try:
        args = [a.decode() for a in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if a]
        cwd = os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        return False
    if not any("nstrain" in a for a in args):
        return False
    return any(Path(cwd, a).resolve() in (cfg.run_dir, cfg.run_dir / C.RUN_FILE) for a in args)


def running(cfg: C.RunConfig) -> dict:
    """Which of the line's processes are alive (by their locks, not by recorded pids)."""
    return {"supervisor": _held(cfg.run_dir / "supervisor.lock"), "trainer": _held(cfg.run_dir / "trainer.lock")}


# ---------------------------------------------------------------------- new / branch
def new(run_dir, model: str, data_work: str, val_work: Optional[str] = None, name: Optional[str] = None,
        peak_lr: float = 3e-4, warmup_tokens: Optional[float] = None, wandb_project: str = "ns-pretrain",
        tokens_per_update: Optional[int] = None, shape_from: Optional[str] = None) -> C.RunConfig:
    """A trunk-type line (warmup, then a constant rate).  ``model``: a preset name or a model-config JSON file;
    ``shape_from``: the preset whose run defaults (lengths, tokens per update, warmup) a model file uses."""
    run_dir = Path(run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"{run_dir} is not empty")
    preset = shape_from or (model if model in M.RUN_DEFAULTS else None)
    if preset not in M.RUN_DEFAULTS:
        raise ValueError(f"no run defaults for {model!r}: give --shape-from one of {sorted(M.RUN_DEFAULTS)}")
    d = M.RUN_DEFAULTS[preset]
    w = float(warmup_tokens if warmup_tokens is not None else d["warmup_tokens"])
    cfg = C.from_dict({
        "name": name or run_dir.name, "model": M.model_config(model), "data_work": str(Path(data_work).resolve()),
        "val_work": str(Path(val_work or data_work).resolve()), "seed": 0, "data_seed": 0,
        "tokens_per_update": int(tokens_per_update or d["tokens_per_update"]), "lengths": d["lengths"],
        "optimizer": dict(M.OPTIMIZER, peak_lr=peak_lr),
        "schedule": [{"from": 0, "to": w, "kind": "warmup"}, {"from": w, "to": None, "kind": "constant"}],
        "wandb": {"project": wandb_project}}, where=str(run_dir / C.RUN_FILE))
    cfg.run_dir = run_dir
    from nstrain.checks import preflight
    problems = preflight(cfg)
    if problems:
        raise ValueError("; ".join(problems))
    C.save(cfg, run_dir)
    return cfg


def cycle_end_checkpoint(parent: C.RunConfig, k: int) -> Path:
    """The parent's kept checkpoint at the end of its data cycle k, up to the parent's latest (read only)."""
    ck = Checkpoints(parent.ckpt_dir)
    latest = ck.latest()
    if latest is None:
        raise FileNotFoundError(f"{parent.name}: no checkpoints yet")
    for d in ck.complete():
        if d.name > latest.name:                                     # not the line's (yet): retention may remove it
            continue
        m = json.loads((d / "trainer.json").read_text())
        if m.get("keep") and m["data"]["work"] == parent.data_work and int(m["data"]["cycle"]) == k + 1 \
                and int(m["data"]["sample"]) == 0:
            return d
    raise FileNotFoundError(f"{parent.name}: no kept checkpoint at the end of cycle {k} in {parent.ckpt_dir} "
                            "(the line has not finished that cycle)")


def branch(run_dir, parent_run, k: int, plan: str, data_work: str, cycle_tokens: Optional[float] = None,
           name: Optional[str] = None, shape: str = "1-sqrt", final: float = 0.05, **prep) -> C.RunConfig:
    """A branch line from the parent's cycle k.  ``prep``: dl_workers, tok_workers, rayon of the data fork."""
    parent = C.load(parent_run)
    run_dir = Path(run_dir).resolve()
    fork = cycle_end_checkpoint(parent, k)
    meta = Checkpoints.verify(fork)
    entries, total = mixture.parse_plan(plan)
    at = float(meta["tokens"])
    d = json.loads(json.dumps(meta["run"]))                          # the parent's settings at the fork
    d.update(name=name or run_dir.name, data_work=str(Path(data_work).resolve()), fork_from=str(fork),
             schedule=sched.branch(meta["run"]["schedule"], at, at + total, shape, final), stop_at_tokens=None,
             allow_changes=[])
    cfg = C.from_dict(d, where=str(run_dir / C.RUN_FILE))
    missing = sorted({int(e["seq_len"]) for e in entries} - set(cfg.lengths))
    if missing:
        raise ValueError(f"the plan uses sequence lengths {missing}; the parent's run.json has no lengths for them")
    if (run_dir / C.RUN_FILE).exists():
        if C.load(run_dir).to_dict() != cfg.to_dict():
            raise FileExistsError(f"{run_dir / C.RUN_FILE} exists and is another branch")
        _record_branch(parent, run_dir, fork, k)
        return C.load(run_dir)
    if run_dir.exists() and any(p.name != C.RUN_FILE + ".pending" for p in run_dir.iterdir()):
        raise FileExistsError(f"{run_dir} is not empty")
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_atomic(run_dir / (C.RUN_FILE + ".pending"), json.dumps(cfg.to_dict(), indent=1))
    pst = json.loads((Path(parent.data_work) / "state.json").read_text())
    c_tok = float(cycle_tokens or pst["cycle_tokens"])
    w = Work(cfg.data_work)
    with w.lock():
        if w.state_path.exists():                                    # forked by an earlier run of this command
            st = json.loads(w.state_path.read_text())
            par = st.get("parent") or {}
            if not (Path(par.get("root", "")) == Path(parent.data_work) and par.get("cycle") == k
                    and st["schedule"] == entries and st["total_tokens"] == total and st["cycle_tokens"] == c_tok
                    and st["line"] == cfg.name):
                raise FileExistsError(f"{w.state_path} exists and is not this fork (parent {par}, line {st['line']})")
        else:
            w.fork(parent.data_work, k, entries, total, c_tok, cfg.name, **prep)
    C.save(cfg, run_dir)
    (run_dir / (C.RUN_FILE + ".pending")).unlink()
    _record_branch(parent, run_dir, fork, k)
    return cfg


def _record_branch(parent: C.RunConfig, run_dir: Path, fork: Path, k: int) -> None:
    p = parent.run_dir / "branches.json"
    rows = json.loads(p.read_text()) if p.exists() else []
    if not any(r["run_dir"] == str(run_dir) for r in rows):
        rows.append({"run_dir": str(run_dir), "fork_from": str(fork), "at_cycle": k, "time": time.time()})
        _write_atomic(p, json.dumps(rows, indent=1))


# ---------------------------------------------------------------------- check
def check(run) -> List[str]:
    """The start checks of the trainer without a GPU and without touching the line (read only): the config, the data
    line and the held-out streams, the newest checkpoint (or the fork point of a branch's first start)."""
    from nstrain.checks import preflight, start_problems
    cfg = C.load(run)
    if not (Path(cfg.data_work) / "state.json").exists():
        return [f"no data line at {cfg.data_work}"]
    problems = preflight(cfg)
    st = json.loads((Path(cfg.data_work) / "state.json").read_text())
    ck = Checkpoints(cfg.ckpt_dir)
    latest = ck.latest()
    if latest is None and ck.complete():
        problems.append(f"{cfg.ckpt_dir} holds checkpoints but no `latest` (the trainer adopts a single first "
                        "checkpoint of this line; otherwise name the newest complete one in `latest`)")
    src = latest if latest is not None else (Path(cfg.fork_from) if cfg.fork_from else None)
    if src is not None:
        if not (src / "trainer.json").exists():
            return problems + [f"{src}: no checkpoint"]
        problems += start_problems(cfg, json.loads((src / "trainer.json").read_text()), st, forking=latest is None)
    if (cfg.run_dir / "rollback.json").exists():
        m = json.loads((cfg.run_dir / "rollback.json").read_text())
        if latest is None or latest.name != f"u{int(m['to']):08d}":
            problems.append(f"rollback.json names update {m['to']}, latest is {latest}: run `nstrain rollback` again")
    return problems


# ---------------------------------------------------------------------- stop
def stop(run, now: bool = False, at_cycle_end: Optional[int] = None) -> str:
    cfg = C.load(run)
    live = running(cfg)
    if now:
        if not live["supervisor"]:
            raise RuntimeError(f"{cfg.name}: no supervisor runs this line")
        pid = int(json.loads((cfg.run_dir / "supervisor.json").read_text())["pid"])
        if not _runs_line(pid, cfg):
            raise RuntimeError(f"{cfg.name}: the recorded supervisor pid {pid} is not this line's supervisor")
        os.kill(pid, signal.SIGTERM)
        return f"SIGTERM to supervisor {pid}: the trainer stops at the next update boundary, then the supervisor ends"
    body = "" if at_cycle_end is None else json.dumps({"at_cycle_end": int(at_cycle_end)})
    _write_atomic(cfg.run_dir / "STOP", body)
    when = "at the next update boundary" if at_cycle_end is None else f"after the last update of cycle {at_cycle_end}"
    note = "" if live["trainer"] else " (no trainer runs now: the next start honors it)"
    return f"{cfg.run_dir / 'STOP'}: the trainer stops {when} with a checkpoint{note}"


# ---------------------------------------------------------------------- rollback
def rollback(run, to: int, force: bool = False) -> str:
    cfg = C.load(run)
    with ExitStack() as held:
        for lock in ("supervisor.lock", "trainer.lock"):             # offline only
            f = held.enter_context(open(cfg.run_dir / lock, "a"))
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"{cfg.name}: {lock} is held (a supervisor / trainer runs this line); "
                                   "stop it first (`nstrain stop`)") from None
        ck = Checkpoints(cfg.ckpt_dir)
        target = cfg.ckpt_dir / f"u{to:08d}"
        marker = cfg.run_dir / "rollback.json"
        pending = json.loads(marker.read_text()) if marker.exists() else None
        if not (target / "trainer.json").exists():
            raise FileNotFoundError(f"{target}: no complete checkpoint of update {to} "
                                    f"(have {[d.name for d in ck.complete()]})")
        meta = Checkpoints.verify(target)
        latest = ck.latest()
        if pending is None or int(pending["to"]) != to:
            if latest is not None and latest.name < target.name:
                raise ValueError(f"update {to} is after the line's latest checkpoint {latest.name}")
            rec_p = cfg.run_dir / "wandb_run.json"
            source = json.loads(rec_p.read_text())["run_id"] if rec_p.exists() else meta.get("wandb_run")
            if not source:
                raise RuntimeError(f"{cfg.name}: no W&B run to roll back from")
            pending = {"to": to, "source_run": source, "time": time.time()}
        later = [d for d in sorted(cfg.ckpt_dir.glob("u[0-9]*")) if d.name > target.name]
        forks = {}
        for b in cfg.run_dir.glob("branches.json"):
            for r in json.loads(b.read_text()):
                forks[Path(r["fork_from"]).name] = r["run_dir"]
        hit = {d.name: forks[d.name] for d in later if d.name in forks}
        if hit and not force:
            raise RuntimeError(f"{cfg.name}: checkpoints after update {to} are fork points of branches {hit}; "
                               "--force moves them anyway (those branches can then not start from them)")
        _write_atomic(marker, json.dumps(pending))                   # 1. what this rollback is
        ck._point_latest(target.name)                                # 2. the line continues from the target
        if later:                                                    # 3. the later checkpoints set aside
            stamp = time.strftime('%Y%m%dT%H%M%S', time.gmtime())
            moved, n = cfg.ckpt_dir / f"rolled-back-{stamp}", 1
            while moved.exists():                                    # an earlier rollback in the same second
                moved, n = cfg.ckpt_dir / f"rolled-back-{stamp}-{n}", n + 1
            moved.mkdir()
            for d in later:
                os.replace(d, moved / d.name)
        w = Work(cfg.data_work)                                      # 4. the data: the target's cycle in training
        with w.lock():
            _write_atomic(w.root / "progress.json", json.dumps({"cycle": int(meta["data"]["cycle"]), "done": False}))
    return (f"{cfg.name}: latest -> {target.name}" + (f"; {len(later)} later checkpoints set aside" if later else "")
            + f"; cycle {meta['data']['cycle']} in training; the next start continues as a new W&B run superseding "
              f"{pending['source_run']}")
