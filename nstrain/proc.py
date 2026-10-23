"""Child processes that end with their parent: the trainer, the preparer and a tune with the supervisor, the
tokenizers with the preparer, a tune's measurements with the tune (Linux PR_SET_PDEATHSIG; the parent's pid comes in
an environment variable, so a child started after its parent died exits at once)."""
from __future__ import annotations

import os
import signal
import sys

SUPERVISOR = "NSTRAIN_SUPERVISOR_PID"
PREPARER = "NSTRAIN_PREPARER_PID"
TUNE = "NSTRAIN_TUNE_PID"


def die_with_parent(var: str, sig: int = signal.SIGTERM) -> None:
    parent = os.environ.get(var)
    if parent is None or not sys.platform.startswith("linux"):
        return
    import ctypes
    if ctypes.CDLL(None, use_errno=True).prctl(1, int(sig), 0, 0, 0) != 0:    # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
    if os.getppid() != int(parent):
        raise SystemExit(f"the parent process {parent} exited before this one started")
