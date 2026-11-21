"""The training step as CUDA graphs, for one shape (micro-batch rows x sequence length).

Three graphs share one memory pool:

* ``micro``: forward + loss + backward of one micro-batch from the static input buffers; the loss is scaled by
  ``1 / accum`` (a device scalar: an update may have another number of micro-batches than the last, a batch-size
  ramp, nstrain.batch, without a new capture) and the gradients accumulate in the flat gradient buffers of
  :class:`FlatAdamW` (the parameters' ``.grad`` are views into them); the loss terms accumulate in static sums.
* ``norm``: the gradient norms (fp32) into static tensors: the groups' and the buckets' (the model's layers,
  nstrain.flat_adamw) and the global one.  The host reads them once per update: a non-finite global norm skips the
  update (the gradients are zeroed, the optimizer and the weights are untouched).
* ``step``: clip by that norm and the AdamW kernels (which also zero the gradients and move the parameters' moving
  average by the device scalar ``ema_c``, set per update); the sums are zeroed.

Data-parallel replicas (``data_parallel``: nstrain.data_parallel, more than one process): the loss scale is
``1 / (accum * world)`` and a fourth graph, ``micro_last``, replaces ``micro`` for the update's last micro-batch: its
backward sums the flat gradient buffers across the replicas bucket by bucket as the layers' gradients are written,
then the rest of the buffers and the loss sums; the earlier micro-batches reduce nothing.  The loss sums are then
divided by ``world``: the metrics are the global batch's, and every replica takes the same decision on a non-finite
norm.  The weights read through their bf16 values are updated in parts (the optimizer split by
:meth:`nstrain.data_parallel.DataParallel.attach`): the ``norm`` graph gathers the replicas' sums of squares, the
``step`` graph updates this replica's parts and gathers the bf16 values of the others.  An engine runs only with the
split it was captured for (another split of the optimizer, or none, needs a new engine).  With one process
(``data_parallel`` None or of world 1) the engine is the one-GPU step.

The graphs are launched with the priorities of the streams their kernels were captured from (:class:`Graph`): the
replicas' collectives, captured from side streams of the highest priority, are dispatched ahead of the pending blocks
of the backward's and the update's kernels they overlap.

The optimizer state is created at construction; the warm-up before capture runs at learning rate 0 (weights unchanged)
and the optimizer state is restored afterwards (with the bf16 values of the linear operator's weights the optimizer
keeps, nstrain.flat_adamw, refreshed before the warm-up and after the restore); buffers the forward may write are
snapshotted and restored as well
(the snapshot lives on the host: the weights, the optimizer's moments and the buffers of a large model are tens of
GiB that the warm-up and the capture would otherwise have to fit beside).  A new shape (a sequence-length change) gets
a new engine over the same model and optimizer, in the same ``arena`` (nstrain.gpu.Arena): it warms up and captures
in the memory the arena holds (the old engine's, once it is released, or memory held beside other processes), never
asking the driver for more when the arena has room.
"""
from __future__ import annotations

import ctypes
import math
import weakref
from typing import Dict, Tuple

import torch

from nstrain.flat_adamw import FlatAdamW

_CU = None                                                       # the CUDA driver, loaded on first use
AUTO_FREE_ON_LAUNCH, USE_NODE_PRIORITY = 1, 8                    # CUgraphInstantiate_flags


def _driver():
    global _CU
    if _CU is None:
        cu = ctypes.CDLL("libcuda.so.1")
        cu.cuGraphInstantiateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                                   ctypes.c_ulonglong]
        cu.cuGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        cu.cuGraphExecDestroy.argtypes = [ctypes.c_void_p]
        for f in (cu.cuGraphInstantiateWithFlags, cu.cuGraphLaunch, cu.cuGraphExecDestroy):
            f.restype = ctypes.c_int
        _CU = cu
    return _CU


_LIVE: "weakref.WeakSet" = weakref.WeakSet()                     # every Graph of the process not yet collected


def release_graphs() -> None:
    """Release every graph of the process that is still alive: before the process group is destroyed
    (nstrain.data_parallel), as NCCL's teardown waits for the graphs that hold its kernels (an engine that was not
    released, or the graphs of an engine whose construction failed after a capture)."""
    for g in list(_LIVE):
        g.release()


class Graph:
    """A captured CUDA graph whose kernels run at the priority of the stream each was captured from.  torch's
    CUDAGraph instantiates without the driver's node-priority flag, so a replayed kernel runs at the priority of the
    stream the graph is launched into, and a collective captured from a high-priority side stream (the backward's
    reductions, nstrain.data_parallel; the step's gathers, nstrain.flat_adamw) is dispatched only after every block of
    the kernel it was meant to overlap.  torch keeps the captured graph (``keep_graph``) and the memory pool its
    buffers live in; the executable is instantiated here with torch's flag and the node priority, launched as torch
    launches a graph (on the current stream of the device it was captured on), and destroyed by :meth:`release` before
    torch's graph (whose reset gives the pool back), once the device has finished the launches in flight."""

    def __init__(self):
        self.g = torch.cuda.CUDAGraph(keep_graph=True)
        self.exec = None
        self.device = torch.cuda.current_device()
        _LIVE.add(self)

    def capture_begin(self, pool, capture_error_mode: str) -> None:
        self.device = torch.cuda.current_device()
        self.g.capture_begin(pool=pool, capture_error_mode=capture_error_mode)

    def capture_end(self) -> None:
        self.g.capture_end()

    def instantiate(self) -> None:
        ex = ctypes.c_void_p()
        rc = _driver().cuGraphInstantiateWithFlags(ctypes.byref(ex), ctypes.c_void_p(self.g.raw_cuda_graph()),
                                                   AUTO_FREE_ON_LAUNCH | USE_NODE_PRIORITY)
        if rc != 0:
            raise RuntimeError(f"nstrain.engine: cuGraphInstantiateWithFlags failed (CUresult {rc})")
        self.exec = ex

    def replay(self) -> None:
        with torch.cuda.device(self.device):
            rc = _driver().cuGraphLaunch(self.exec, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        if rc != 0:
            raise RuntimeError(f"nstrain.engine: cuGraphLaunch failed (CUresult {rc})")

    def release(self) -> None:
        if self.exec is not None or self.g is not None:
            torch.cuda.synchronize(self.device)                  # (a launch in flight still uses the pool's memory)
        if self.exec is not None:
            _driver().cuGraphExecDestroy(self.exec)
            self.exec = None
        if self.g is not None:
            self.g.reset()
            self.g = None

    def __del__(self):
        try:
            self.release()
        except Exception:                                        # (the interpreter's exit: the driver may be gone)
            pass


class StepEngine:
    def __init__(self, model, optimizer: FlatAdamW, rows: int, seq_len: int, accum: int,
                 grad_clip: float = 1.0, warmup_iters: int = 3, compile: bool = False, arena=None,
                 before_capture=None, data_parallel=None):
        self.model, self.opt = model, optimizer
        self.rep = data_parallel if data_parallel is not None and data_parallel.world > 1 else None
        if self.rep is None and optimizer.sharding is not None:
            raise RuntimeError("StepEngine: the optimizer is split among data-parallel replicas (FlatAdamW.shard); "
                               "a one-GPU engine needs it whole: call unshard() on every replica first")
        self.world = self.rep.world if self.rep is not None else 1
        self.loss_fn = torch.compile(model.loss, dynamic=False) if compile else model.loss
        self.rows, self.L, self.accum, self.clip = int(rows), int(seq_len), int(accum), float(grad_clip)
        self.dev = next(model.parameters()).device
        self.x = torch.zeros((self.rows, self.L), dtype=torch.int64, device=self.dev)
        self.y = torch.zeros((self.rows, self.L), dtype=torch.int64, device=self.dev)
        f32 = dict(dtype=torch.float32, device=self.dev)
        self.inv_accum = torch.full((), 1.0 / (self.accum * self.world), **f32)   # the micro-batches' loss scale
        self.sums = torch.zeros(3, **f32)                            # (set per update); the loss terms' sums
        self.loss_sum, self.ce_sum, self.aux_sum = self.sums[0], self.sums[1], self.sums[2]
        self.gnorm = torch.zeros((), **f32)
        self.gnorms = torch.zeros(len(optimizer.groups), **f32)      # per parameter group (the diagnostic of a
        self.group_names = [g["name"] for g in optimizer.groups]     # spike: nstrain.trainer logs them by name)
        self.gnorm_buckets = torch.zeros(len(optimizer.buckets), **f32)   # per bucket (the model's layers)
        self.graphs: Dict[str, Graph] = {}
        from nstrain.gpu import Arena
        self.arena = arena if arena is not None else Arena()          # the trainer's: memory it already holds
        if self.rep is not None:
            self.rep.attach(model, optimizer, self.arena.stream)
        self.sharding = optimizer.sharding                           # the split the graphs are captured for
        self.bf16_at = optimizer.bf16_buffers()                      # and the bf16 buffers they read and write
        self._capture(warmup_iters, before_capture)

    # ------------------------------------------------------------------ bodies (eager and captured)
    def _micro(self) -> None:
        out = self.loss_fn(self.x, self.y)
        (out.loss * self.inv_accum).backward()
        self.loss_sum.add_(out.loss.detach())
        self.ce_sum.add_(out.ce.detach())
        self.aux_sum.add_(out.aux.detach())

    def _micro_last(self) -> None:
        self.rep.begin()
        out = self.loss_fn(self.x, self.y)
        self.rep.forward_done()
        (out.loss * self.inv_accum).backward()
        self.loss_sum.add_(out.loss.detach())
        self.ce_sum.add_(out.ce.detach())
        self.aux_sum.add_(out.aux.detach())
        self.rep.finish(self.sums)
        self.sums.mul_(1.0 / self.world)

    def _norm(self) -> None:
        groups, buckets = self.opt.grad_norms()
        self.gnorms.copy_(groups)
        self.gnorm_buckets.copy_(buckets)
        self.gnorm.copy_(torch.linalg.vector_norm(self.gnorms))

    def _step(self) -> None:
        coef = torch.clamp(self.clip / (self.gnorm + 1e-6), max=1.0)
        self.opt.step(coef)
        self._zero_sums()

    def _zero_sums(self) -> None:
        self.sums.zero_()

    # ------------------------------------------------------------------ capture
    def _capture(self, warmup_iters: int, before_capture=None) -> None:
        # the warm-up runs real steps at learning rate 0 and moving-average step 0; a non-finite warm-up gradient would
        # still reach the weights, the moments and the moving average (0 x NaN), so all of them are restored
        # everything in the arena (the copies, the warm-up, the captures): memory it holds, no new driver memory;
        # captured without torch.cuda.graph, whose entry empties the allocator's cache (it would give the memory
        # back to the driver, for any process to take), thread-local (the prefetch thread may pin host memory)
        a = self.arena
        a.stream.wait_stream(torch.cuda.current_stream())
        saved = self._snapshot()
        with a.use():
            self.opt.refresh_bf16()                                  # (weights written since the optimizer was built:
            self.opt.lr.zero_()                                      # the launch calibration); warm-up steps leave the
            self.opt.ema_c.zero_()                                   # weights unchanged, and the moving average
            for _ in range(warmup_iters):
                if self.rep is None:
                    for _ in range(min(2, self.accum)):
                        self._micro()
                else:
                    self._micro()
                    self._micro_last()
                self._norm()
                self._step()
        torch.cuda.synchronize()
        if before_capture is not None:                               # eager work whose first-call caches must be
            self.opt.gather_()                                       # (its reads of the fp32 weights: whole)
            before_capture()                                         # live before the arena's free blocks become
        bodies = [("micro", self._micro)] + ([("micro_last", self._micro_last)] if self.rep is not None else []) + \
            [("norm", self._norm), ("step", self._step)]
        for name, body in bodies:                                    # graph scratch
            g = Graph()
            with torch.cuda.stream(a.stream):
                g.capture_begin(pool=a.pool.id, capture_error_mode="thread_local")
                try:
                    body()
                finally:                                             # (as torch.cuda.graph's exit: a failed body
                    g.capture_end()                                  # still ends the capture)
            g.instantiate()
            self.graphs[name] = g
        torch.cuda.synchronize()
        self._restore(saved)                                         # undo the warm-up
        with a.use():
            self.opt.zero_grad()
            self._zero_sums()
        torch.cuda.synchronize()

    def _snapshot(self) -> dict:
        """The state the warm-up (and the validation before the capture) may change, copied to the host: the
        optimizer's flat buffers (the trainable weights ``p``, the moments ``m``, ``v`` and, with ``ema``, the moving
        average ``e``, one contiguous copy per group), the parameters outside them (none trainable), the model's
        buffers, the step count and the learning rate."""
        torch.cuda.synchronize()                                     # (the last engine's work on them: done)
        keys = ("p", "m", "v") + (("e",) if self.opt.ema else ())
        flat = [{k: g["flat"][k].to("cpu") for k in keys} for g in self.opt.groups]
        others = {n: p.detach().to("cpu") for n, p in self.model.named_parameters() if n not in self.opt.slots}
        bufs = {n: b.detach().to("cpu") for n, b in self.model.named_buffers()}
        torch.cuda.synchronize()
        return {"flat": flat, "others": others, "bufs": bufs, "t": self.opt.t.to("cpu"), "bc": self.opt.bc.to("cpu"),
                "lr": self.opt.lr.to("cpu"), "ema_c": self.opt.ema_c.to("cpu"), "stale": self.opt.stale}

    @torch.no_grad()
    def _restore(self, saved: dict) -> None:
        for g, h in zip(self.opt.groups, saved["flat"]):
            for k, v in h.items():
                g["flat"][k].copy_(v)
        params = dict(self.model.named_parameters())
        for n, q in saved["others"].items():
            params[n].copy_(q)
        bufs = dict(self.model.named_buffers())
        for n, q in saved["bufs"].items():
            bufs[n].copy_(q)
        self.opt.t.copy_(saved["t"])
        self.opt.bc.copy_(saved["bc"])
        self.opt.lr.copy_(saved["lr"])
        self.opt.ema_c.copy_(saved["ema_c"])
        self.opt.stale = saved["stale"]
        self.opt.refresh_bf16()                                      # the bf16 values of the restored weights
        torch.cuda.synchronize()

    # ------------------------------------------------------------------ use
    def update(self, inputs: torch.Tensor, targets: torch.Tensor, base_lr: float, wd_scale: float = 1.0,
               ema_c: float = 0.0) -> Tuple[dict, bool]:
        """One optimizer update over the micro-batches of the pinned (accum * rows, L) inputs / targets (``accum``
        their number: the update's tokens over the micro-batch's, any number) at the base learning rate ``base_lr``,
        the weight decay scale ``wd_scale`` (1: the groups' values) and the moving average's step ``ema_c`` (0: the
        average untouched).  Returns (metrics, applied); a non-finite gradient norm skips the update."""
        if inputs.ndim != 2 or inputs.shape[1] != self.L or inputs.shape[0] % self.rows or inputs.shape[0] == 0:
            raise ValueError(f"update of shape {tuple(inputs.shape)}, engine is (n x {self.rows}) x {self.L}")
        self._check_split()
        self.opt.check_bf16()                                        # (a replayed graph reads the bf16 values kept
        accum = int(inputs.shape[0]) // self.rows                    # by the optimizer and cannot check them itself)
        self.set_accum(accum)
        for m in range(accum):
            sl = slice(m * self.rows, (m + 1) * self.rows)
            self.micro_batch(inputs[sl], targets[sl], last=m == accum - 1)
        self.graphs["norm"].replay()
        sums = torch.cat([torch.stack([self.gnorm, self.loss_sum, self.ce_sum, self.aux_sum]), self.gnorms,
                          self.gnorm_buckets]).tolist()
        gn, ng = sums[0], len(self.group_names)
        metrics = {"loss": sums[1] / accum, "ce": sums[2] / accum, "aux": sums[3] / accum,
                   "grad_norm": gn, "grad_norms": dict(zip(self.group_names, sums[4:4 + ng])),
                   "grad_norm_buckets": dict(zip(self.opt.buckets, sums[4 + ng:]))}
        if not math.isfinite(gn) or not math.isfinite(metrics["loss"]):
            self.opt.zero_grad()
            self._zero_sums()
            return metrics, False
        self.opt.lr.fill_(base_lr)
        self.opt.wd.fill_(wd_scale)
        self.opt.ema_c.fill_(ema_c)
        self.graphs["step"].replay()
        self.opt.stale = self.sharding is not None
        return metrics, True

    def _check_split(self) -> None:
        """Refuse to replay graphs captured for another split of the optimizer among the replicas, or for bf16
        buffers the optimizer no longer uses (a split moves them into symmetric memory, and unsharding leaves them
        there): they would reduce and update the old parts while the optimizer reads the new ones, or read and write
        memory that is no longer the weights'."""
        if self.opt.sharding is not self.sharding or self.opt.bf16_buffers() != self.bf16_at:
            raise RuntimeError("StepEngine: the optimizer's split among the data-parallel replicas or its bf16 buffers "
                               "changed since this engine was captured (FlatAdamW.shard / unshard): release it and "
                               "build a new engine")

    def set_accum(self, accum: int) -> None:
        """The number of micro-batches of the next update (per replica): the loss scale ``1 / (accum * world)``."""
        if accum != self.accum:
            self.accum = accum
            self.inv_accum.fill_(1.0 / (accum * self.world))

    def micro_batch(self, x: torch.Tensor, y: torch.Tensor, last: bool) -> None:
        """Forward and backward of one micro-batch of pinned (rows, L) inputs / targets, its gradient added into the
        flat gradient buffers; ``last``: the update's last micro-batch, whose backward sums the replicas' gradients
        (nothing is reduced in the others)."""
        self._check_split()
        self.x.copy_(x, non_blocking=True)
        self.y.copy_(y, non_blocking=True)
        self.graphs["micro_last" if last and self.rep is not None else "micro"].replay()

    def release(self) -> None:
        for g in self.graphs.values():
            g.release()
        self.graphs.clear()
        if self.rep is not None:
            self.rep.detach()
