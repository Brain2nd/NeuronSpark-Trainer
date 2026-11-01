"""The training GPU: a host-wide lease and the other compute processes on it.

When the GPU is time-sliced with another process, the driver's preemption of the training kernels can fault, so by
default the trainer refuses to start beside another process and checkpoints and exits (nstrain.exits.GPU_BUSY) when
one appears; the supervisor waits for the GPU to be free.  With ``guards.share_reserve_gb`` the trainer instead always
trains with the lengths' ``shared`` shapes, measured to fit while other processes use up to the reserve plus the
tune's margin less a capture's slack (:func:`fits_beside`), whether other processes are there or not. It holds all of
the GPU's memory but what the other processes may still take, the reserve less what they use (:func:`hold_memory`, in
the trainer's :class:`Arena`, which never gives memory back while the trainer runs, and a slack block of a few GiB
that goes back to the driver only while an engine is captured or a new line's first forward runs, for their memory
outside the allocator): they cannot grow past the reserve -- but for that slack while it is out and the 0.25 GiB
left for the trainer's own memory outside the allocator -- (an allocation past it fails in them, not in the line),
the line does not stop for them, and what they free past the reserve is held again, never what is theirs: the free
memory taken is read between two readings of theirs (:func:`room_beside`).  Memory they already used past the
reserve when the line started stays theirs until they free it.  It never switches shapes for them: a switch would
give memory back to the driver, and any process could take it then.  The lease (an exclusive lock on
``$NSTRAIN_GPU_LOCK_DIR`` (default /tmp) ``/nstrain-gpu<index>.lock``, taken before any CUDA initialisation and held
for the life of the process) closes the window in which two trainers (or a trainer and ``nstrain bench``) both see a
free GPU at start.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Callable


class GpuBusy(RuntimeError):
    pass


class GpuCheckFailed(RuntimeError):
    pass


CONTEXT_GIB = 2.0                       # the CUDA context and the libraries beside PyTorch's allocator
TUNE_MARGIN_GIB = 10.0                  # what `nstrain tune` leaves free beside a shape's peak (by default)
CAPTURE_SLACK_GIB = 4.0                 # beside others, what a trainer gives back while it captures an engine (or runs
                                        # its first forward): memory outside the allocator (kernel modules, local
                                        # memory, graph data: about 2.5 GiB for the 2B)


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
    margin (the shapes were measured to leave both free) less the slack a capture needs free (CAPTURE_SLACK_GIB)."""
    return others * 1e9 / 2**30 <= reserve_gb * 1e9 / 2**30 + TUNE_MARGIN_GIB - CAPTURE_SLACK_GIB


class Arena:
    """The trainer's GPU working memory: one CUDA memory pool (torch.cuda.MemPool) and one stream.  Every engine
    warms up and captures its graphs in it, validation runs in it, and held memory (``hold_memory``) is cached in it,
    on that stream -- so a capture or a new engine reuses what the arena holds instead of asking the driver (a block
    is reused only on the stream it was allocated on), and ``torch.cuda.empty_cache`` does not give it back (a pool's
    blocks go back only when the arena is closed)."""

    def __init__(self):
        import torch
        self.pool = torch.cuda.MemPool()
        self.stream = torch.cuda.Stream()

    @contextlib.contextmanager
    def use(self):
        """Work of the current thread on the arena's stream with its allocations in the arena; ordered after the
        caller's stream on entry and before it again on exit (the arena's free blocks are also the graphs' scratch:
        nothing in it runs beside a replay)."""
        import torch
        outer = torch.cuda.current_stream()
        self.stream.wait_stream(outer)
        dev = torch.cuda.current_device()
        with torch.cuda.stream(self.stream):
            # every allocation on the arena's stream, of any thread (backward runs in autograd's own thread, which
            # torch.cuda.use_mem_pool -- current thread only -- would miss), as torch's own graph capture routes them
            torch._C._cuda_beginAllocateCurrentStreamToPool(dev, self.pool.id)
            try:
                yield
            finally:
                torch._C._cuda_endAllocateToPool(dev, self.pool.id)
                torch._C._cuda_releasePool(dev, self.pool.id)
        outer.wait_stream(self.stream)

    def close(self) -> None:
        """The arena's memory back to the driver (after every graph captured in it is released).  The cuBLAS
        workspaces are dropped first: torch keeps one per stream in its allocator, so the arena's stream got one
        inside the arena -- a few MiB that would pin a whole held segment (32 GiB) of the closed pool for good."""
        import gc
        import torch
        torch.cuda.synchronize()
        gc.collect()                                                 # graphs of a dead engine
        if self.pool.use_count() != 1:                               # (torch aborts the process otherwise)
            raise RuntimeError(f"the arena still has {self.pool.use_count() - 1} users (graphs captured in it)")
        torch._C._cuda_clearCublasWorkspaces()
        self.pool = None
        torch.cuda.empty_cache()


def room_beside(reserve_gb: float, others: float, recheck: Callable[[], float], slack_gib: float = 0.25,
                least: int = 16 << 20) -> int:
    """Bytes this process may take now beside the other processes: the GPU's free memory but the reserve less what
    they use and ``slack_gib``.  The free memory is read between two readings of theirs (``others`` GB before it,
    ``recheck()`` after it; the lower one counts), so memory they free meanwhile is not counted: it stays theirs, up
    to the reserve.  0 when ``recheck`` fails (GpuCheckFailed) or there is less than ``least`` to take (then
    ``recheck`` is not called)."""
    import torch

    def leave(gb: float) -> int:
        return max(0, int((reserve_gb - gb) * 1e9)) + int(slack_gib * 2**30)
    free = torch.cuda.mem_get_info()[0]
    if free - leave(others) < least:
        return 0
    try:
        return max(0, free - leave(min(others, recheck())))
    except GpuCheckFailed:
        return 0


def reclaim_dead(least: int = 16 << 20) -> int:
    """GPU memory the driver keeps although no process holds it (seen three times on 2026-10-31 on the B300 after
    processes were killed: 17-49 GB used by no process, for an hour or more; a killed process holding 248 GiB, busy or
    idle, left none in a direct trial, so what leaves it is not known) is reclaimed when an allocation runs the free
    memory out -- by whoever allocates, so a neighbour could take it beyond its reserve.  An allocation past the
    reported free memory, in a pool of its own that goes at once, reclaims it for this process's next hold: returns
    the bytes freed (0: none, the allocation fails).  For the milliseconds of the probe the other processes' free
    memory is taken too (an allocation of theirs then fails), so a line probes only when it starts or captures an
    engine, not at every check."""
    import torch
    free = torch.cuda.mem_get_info()[0]
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    torch.cuda.set_per_process_memory_fraction(1.0)                 # (the hold caps the allocator again)
    pool = torch.cuda.MemPool()
    try:
        with torch.cuda.use_mem_pool(pool):
            probe = torch.empty(min(total, free + least), dtype=torch.uint8, device="cuda")
    except torch.cuda.OutOfMemoryError:
        return 0
    del probe
    torch.cuda.synchronize()
    del pool
    torch.cuda.empty_cache()                                         # (its block back to the driver now, not at the
    return max(0, torch.cuda.mem_get_info()[0] - free)               # allocator's next release)


def hold_memory(reserve_gb: float, others: float, recheck: Callable[[], float], sizes_gib=(32.0, 4.0, 1.0),
                slack_gib: float = 0.25, headroom_gib: float = 1.0, arena: "Arena" = None,
                reclaim: bool = False) -> float:
    """Take the GPU's free memory into this process's allocator but what the other processes may still take: the
    reserve less what they use (``others`` GB), plus ``slack_gib`` (this process's own memory outside the allocator,
    e.g. a kernel module loaded late) -- the bytes :func:`room_beside` gives, from one reading of the free memory
    between two of theirs, taken from the driver: blocks are allocated (and kept until the end) until this process
    reserves that much more, so free blocks it already caches serve the first ones and are not counted.  Blocks of
    ``sizes_gib`` (largest first, each as large as fits -- few, large blocks: fewer allocations, a shorter hold; the
    allocator splits them for smaller requests, so a later eager allocation is served from a held block), then the
    rest in one block; the freed blocks stay reserved by this process and serve its own allocations.  With ``reclaim`` the memory the driver keeps from dead
    processes is taken too (:func:`reclaim_dead`, then held again, until the probe finds none).  The
    allocator's cap (:func:`cap_memory`) is then what it holds plus ``headroom_gib``: an allocation that no held
    block can serve takes at most that much more of the GPU; past it the allocator gives back the free blocks outside
    the arena (the arena's stay).  With ``arena`` the blocks are cached in it, on its stream (they serve the arena's
    engines and validation).  Returns the GiB taken."""
    import torch
    unit, least = 2 << 20, 16 << 20                                  # requests under 10 MiB take 20 MiB segments
    sizes, head = [int(s * 2**30) for s in sizes_gib], int(headroom_gib * 2**30)
    before = torch.cuda.memory_reserved()
    while True:
        start = torch.cuda.memory_reserved()
        budget = room_beside(reserve_gb, others, recheck, slack_gib, least) // unit * unit
        held = []
        if budget >= least:
            cap_memory((start + budget + head + unit) / 2**30)
            with arena.use() if arena is not None else contextlib.nullcontext():
                try:
                    while (left := budget - (torch.cuda.memory_reserved() - start)) >= least:
                        n = next((s for s in sizes if s <= left), left // unit * unit)
                        held.append(torch.empty(n, dtype=torch.uint8, device="cuda"))
                except torch.cuda.OutOfMemoryError:                  # another process took the memory meanwhile
                    pass
                del held
            if arena is not None:
                arena.stream.synchronize()
        if not reclaim or reclaim_dead(least) < least:
            break
    reserved = torch.cuda.memory_reserved()
    cap_memory((reserved + head) / 2**30)
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
    from nstrain.proc import flock_held
    p = _lease_path()
    if not flock_held(p):
        return None
    try:
        return json.loads(p.read_text() or "{}")
    except (OSError, json.JSONDecodeError):
        return {}
