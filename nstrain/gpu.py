"""The training GPU: a host-wide lease and the other compute processes on it.

When the GPU is time-sliced with another process, the driver's preemption of the training kernels can fault, so by
default the trainer refuses to start beside another process and checkpoints and exits (nstrain.exits.GPU_BUSY) when
one appears; the supervisor waits for the GPU to be free.  With ``guards.share_reserve_gb`` the supervisor instead
relaunches the trainer beside the other processes (``$NSTRAIN_GPU_MODE=shared``) with the lengths' ``shared`` shapes,
measured to fit while the other processes use up to the reserve plus the tune's margin (:func:`fits_beside`).  The
trainer then holds all of the GPU's memory but what the other processes may still take, the reserve less what they use
(:func:`hold_memory`): they cannot grow past the reserve (an allocation past it fails in them, not in the line), and
the line does not stop for them (memory they already used past the reserve when the line started stays theirs until
they free it); when they leave, the trainer checkpoints and exits (nstrain.exits.GPU_FREE) and is relaunched with the
GPU to itself.  The lease (an
exclusive lock on ``$NSTRAIN_GPU_LOCK_DIR`` (default /tmp) ``/nstrain-gpu<index>.lock``, taken before any CUDA
initialisation and held for the life of the process) closes the window in which two trainers (or a trainer and
``nstrain bench``) both see a free GPU at start.
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


class GpuFree(RuntimeError):
    pass


GPU_MODE = "NSTRAIN_GPU_MODE"           # "shared": the trainer runs beside other processes (set by the supervisor)
CONTEXT_GIB = 2.0                       # the CUDA context and the libraries beside PyTorch's allocator
TUNE_MARGIN_GIB = 10.0                  # what `nstrain tune` leaves free beside a shape's peak (by default)


def beside_limit_gib(total_gib: float, reserve_gb: float, margin_gib: float = 0.0) -> float:
    """PyTorch memory (GiB) a process beside others may reserve: the GPU's memory less what is left to them, the CUDA
    context and a margin."""
    return total_gib - reserve_gb * 1e9 / 2**30 - CONTEXT_GIB - margin_gib


def cap_memory(limit_gib: float) -> None:
    """PyTorch's allocator in this process runs out of memory past ``limit_gib`` instead of taking memory left to other
    processes on the GPU."""
    import torch
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, max(0.0, limit_gib * 2**30 / total)))


def fits_beside(others: float, reserve_gb: float) -> bool:
    """The shapes beside other processes fit while they use ``others`` GB: at most the reserve plus the tune's
    margin (the shapes were measured to leave both free)."""
    return others * 1e9 / 2**30 <= reserve_gb * 1e9 / 2**30 + TUNE_MARGIN_GIB


def hold_memory(reserve_gb: float, others: float, sizes_gib=(4.0, 1.0), slack_gib: float = 0.25,
                headroom_gib: float = 1.0) -> float:
    """Take the GPU's free memory into this process's allocator but what the other processes may still take: the
    reserve less what they use (``others`` GB), plus ``slack_gib`` (this process's own memory outside the allocator,
    e.g. a kernel module loaded late).  Taken in blocks of ``sizes_gib`` (largest first, each as large as fits; the
    largest as large as the largest eager allocation, the readout chunk, so that a later one is served from a held
    block), then the rest in 2 MiB units; the freed blocks stay reserved by this process and serve its own allocations.
    The allocator's cap (:func:`cap_memory`) is then what it holds plus ``headroom_gib``: an allocation that no held
    block can serve takes at most that much more of the GPU; past it the allocator first gives its free blocks back
    (the next call takes the memory again).  Returns the GiB taken (negative when the allocator gave memory back)."""
    import torch
    unit, least = 2 << 20, 16 << 20                                  # requests under 10 MiB take 20 MiB segments
    leave = max(0, int((reserve_gb - others) * 1e9)) + int(slack_gib * 2**30)
    sizes, head = [int(s * 2**30) for s in sizes_gib], int(headroom_gib * 2**30)
    before, held = torch.cuda.memory_reserved(), []
    cap_memory((before + max(0, torch.cuda.mem_get_info()[0] - leave) + head + unit) / 2**30)
    try:
        while True:
            room = torch.cuda.mem_get_info()[0] - leave
            n = next((s for s in sizes if s <= room), room // unit * unit)
            if n < least:
                break
            held.append(torch.empty(n, dtype=torch.uint8, device="cuda"))
    except torch.cuda.OutOfMemoryError:                              # another process took the memory meanwhile
        pass
    del held
    reserved = torch.cuda.memory_reserved()
    cap_memory((reserved + max(0, torch.cuda.mem_get_info()[0] - leave) + head) / 2**30)
    return (reserved - before) / 2**30


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


def others_gb(exclude_pid: int = None) -> float:
    """GPU memory in GB (10^9 bytes) of the compute processes on the training GPU other than ``exclude_pid`` and the
    MPS server; raises :class:`GpuCheckFailed` when nvidia-smi fails or does not report a process's memory."""
    try:
        out = subprocess.run(["nvidia-smi", "-i", device(), "--query-compute-apps=pid,process_name,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
                             timeout=60).stdout
    except Exception as e:
        raise GpuCheckFailed(f"nvidia-smi failed: {e}") from e
    mib = 0.0
    for r in (r.strip() for r in out.splitlines()):
        if not r or "mps-server" in r:
            continue
        f = [x.strip() for x in r.split(",")]
        if exclude_pid is not None and f[0] == str(exclude_pid):
            continue
        try:
            mib += float(f[-1])
        except ValueError:
            raise GpuCheckFailed(f"nvidia-smi reports no memory for process {r!r}") from None
    return mib * 2**20 / 1e9


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
