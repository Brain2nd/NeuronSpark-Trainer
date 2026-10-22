"""The training GPU: a host-wide lease and the other compute processes on it (training must have it to itself).

When the GPU is time-sliced with another process, the driver's preemption of the training kernels can fault, so the
trainer refuses to start beside another process and checkpoints and exits (nstrain.exits.GPU_BUSY) when one appears;
the supervisor waits for the GPU to be free.  The lease (an exclusive lock on ``$NSTRAIN_GPU_LOCK_DIR`` (default
/tmp) ``/nstrain-gpu<index>.lock``, taken before any CUDA initialisation and held for the life of the process) closes
the window in which two trainers (or a trainer and ``nstrain bench``) both see a free GPU at start.
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
from pathlib import Path


class GpuBusy(RuntimeError):
    pass


class GpuCheckFailed(RuntimeError):
    pass


def device() -> str:
    """The physical index of the training GPU (the first of CUDA_VISIBLE_DEVICES)."""
    return os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"


def other_processes(exclude_pid: int = None) -> list:
    """Compute processes on the training GPU other than ``exclude_pid`` and the MPS server; raises
    :class:`GpuCheckFailed` when nvidia-smi fails (a failure, not a busy GPU)."""
    try:
        out = subprocess.run(["nvidia-smi", "-i", device(), "--query-compute-apps=pid,process_name",
                              "--format=csv,noheader"], capture_output=True, text=True, check=True, timeout=60).stdout
    except Exception as e:
        raise GpuCheckFailed(f"nvidia-smi failed: {e}") from e
    others = []
    for r in (r.strip() for r in out.splitlines()):
        if not r or "mps-server" in r:
            continue
        try:
            if exclude_pid is not None and int(r.split(",")[0]) == exclude_pid:
                continue
        except ValueError:                                           # an unreadable pid counts as another process
            pass
        others.append(r)
    return others


def _lease_path() -> Path:
    return Path(os.environ.get("NSTRAIN_GPU_LOCK_DIR", "/tmp")) / f"nstrain-gpu{device()}.lock"


def lease(what: str):
    """Take the GPU lease for this process (keep the returned file open); raises :class:`GpuBusy` with the holder."""
    p = _lease_path()
    f = open(p, "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.seek(0)
        holder = f.read().strip()
        f.close()
        raise GpuBusy(f"GPU {device()} is leased by {holder or 'another nstrain process'}") from None
    f.seek(0)
    f.truncate()
    f.write(json.dumps({"pid": os.getpid(), "what": what, "since": time.time()}))
    f.flush()
    return f


def lease_holder():
    """The holder's record when the GPU lease is held by another process, else None."""
    p = _lease_path()
    if not p.exists():
        return None
    with open(p, "a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            f.seek(0)
            try:
                return json.loads(f.read() or "{}")
            except json.JSONDecodeError:
                return {}
        fcntl.flock(f, fcntl.LOCK_UN)
        return None
