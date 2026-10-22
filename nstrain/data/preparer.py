"""The data preparer of a line: keeps the cycle in training and ``lookahead`` cycles after it prepared, rebuilds a
released cycle that is needed again (a rollback), and releases the cycles the trainer has consumed.

One preparer serves a work directory: whoever serves it holds ``<work>/preparer.lock`` -- the supervisor's preparer
thread for as long as it keeps a preparer running (also across restarts), or a ``nstrain data loop`` started by hand.
The trainer waits for a released cycle only while the line is served (:func:`served`).  Each pass holds the work
directory's lock (one writer of state.json at a time).  The cycle in training comes from ``progress.json`` (written by
the trainer, :meth:`nstrain.data.work.Work.in_training`).  A restart resumes a cycle in preparation with the same files
(state.json: inflight).
"""
from __future__ import annotations

import fcntl
import os
import shutil
import time
from pathlib import Path

from nstrain.data.work import Work

HELD_BY_PARENT = "NSTRAIN_PREPARER_LOCK_HELD"


def serve(work) -> object:
    """Take ``<work>/preparer.lock`` (keep the returned file open while serving); RuntimeError when another process
    serves the line."""
    f = open(Path(work) / "preparer.lock", "a")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        raise RuntimeError(f"another preparer serves {work}") from None
    return f


def served(work) -> bool:
    p = Path(work) / "preparer.lock"
    if not p.exists():
        return False
    with open(p, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(f, fcntl.LOCK_UN)
        return False


def loop_pass(w: Work, lookahead: int = 1, min_free_gb: float = 200.0, **kw) -> float:
    """One pass: release the consumed cycles, or rebuild / prepare the next cycle needed; returns the pause before
    the next pass (s)."""
    with w.lock():
        st = w.load()
        cur = w.in_training()
        gone = sorted(int(k) for k, v in st["cycles"].items() if int(k) < cur and v["status"] in ("prepared",
                                                                                                "releasing"))
        for k in gone:
            w.release(k)
            print(f"released cycle {k}", flush=True)
        if gone:
            return 0.0
        for k, v in st["cycles"].items():
            if v["status"] == "prepared":
                w.ensure_visible(int(k))
        T, C = st["total_tokens"], st["cycle_tokens"]
        last = cur + 1 + lookahead if T is None else min(int(-(-T // C)), cur + 1 + lookahead)
        want = [k for k in range(cur, last) if st["cycles"].get(str(k), {}).get("status") != "prepared"]
        if not want:
            return 60.0
        free = shutil.disk_usage(w.root).free / 1e9
        if free < min_free_gb:
            print(f"waiting: {free:.0f} GB free < {min_free_gb:.0f}", flush=True)
            return 300.0
        k, t0 = want[0], time.time()
        if st["cycles"].get(str(k), {}).get("status") in ("released", "releasing"):
            w.rebuild(k, **kw)
            print(f"rebuilt cycle {k} in {time.time() - t0:.0f} s", flush=True)
        else:
            w.prepare(k, **kw)
            print(f"prepared cycle {k} in {time.time() - t0:.0f} s", flush=True)
        return 0.0


def release_consumed(work) -> list:
    """Release the cycles the trainer has consumed (when a line stops: no pass of the loop follows)."""
    w = Work(work)
    with w.lock():
        st = w.load()
        cur = w.in_training()
        gone = sorted(int(k) for k, v in st["cycles"].items() if int(k) < cur and v["status"] in ("prepared",
                                                                                                "releasing"))
        for k in gone:
            w.release(k)
    return gone


def loop(work: str, lookahead: int = 1, min_free_gb: float = 200.0, **kw) -> None:
    held = None if os.environ.get(HELD_BY_PARENT) else serve(work)       # noqa: F841 (held while the loop runs)
    w = Work(work)
    while True:
        time.sleep(loop_pass(w, lookahead, min_free_gb, **kw))
