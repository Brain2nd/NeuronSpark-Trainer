"""The training step as CUDA graphs, for one shape (micro-batch rows x sequence length).

Four graphs share one memory pool:

* ``micro_first`` and ``micro``: forward + loss + backward of one micro-batch from the static input buffers, the
  first micro-batch of an update and the others; the loss is scaled by ``1 / accum`` (a device scalar: an update may
  have another number of micro-batches than the last, a batch-size ramp, nstrain.batch, without a new capture) and the
  gradients accumulate in the flat gradient buffers of :class:`FlatAdamW` (the parameters' ``.grad`` are views into
  them); the loss terms accumulate in static sums.  In ``micro_first`` the weights the optimizer lists as
  ``written_first`` (the projections and the tied embedding: the linear operator's GEMM writes each one's gradient
  before anything else in the backward) are armed (``neuronspark.ops.linear.arm_grad_overwrite_``), so their first
  GEMM writes the gradient instead of adding it into zeros, and the step does not zero their part of the buffers;
  the capture refuses a ``micro_first`` that leaves one of them armed (a weight its backward did not reach).  The
  gradients autograd itself accumulates (every parameter the linear operator does not write: the vectors, the
  lexical tables, the matrices read in fp32) are added by multi-tensor kernels instead of one add per parameter
  (:class:`FusedAccumulation`).
* ``norm``: the gradient norms (fp32) into static tensors: the groups' and the buckets' (the model's layers,
  nstrain.flat_adamw) and the global one.  The host reads them once per update: a non-finite global norm skips the
  update (the gradients are zeroed, the optimizer and the weights are untouched).
* ``step``: clip by that norm and the AdamW kernels (which also zero the gradients but those of the ``written_first``
  weights, and move the parameters' moving average by the device scalar ``ema_c``, set per update); the sums are
  zeroed.  Between updates the ``written_first`` weights' gradients are those of the last update (a skipped update
  and the end of the capture zero every gradient, :meth:`StepEngine.release` too).

Data-parallel replicas (``data_parallel``: nstrain.data_parallel, more than one process): the loss scale is
``1 / (accum * world)`` and two more graphs replace ``micro`` for the update's last micro-batch, ``micro_last`` and,
for an update of one micro-batch, ``micro_only`` (``micro_first``'s arming as well): its backward sums the flat
gradient buffers across the replicas bucket by bucket as the layers' gradients are written, then the rest of the
buffers and the loss sums; the earlier micro-batches reduce nothing.  The loss sums are then
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
from functools import partial
from typing import Dict, List, Tuple

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
_CAPTURING = None                                                # the Graph whose capture is in progress


def release_graphs() -> None:
    """Release every graph of the process that is still alive: before the process group is destroyed
    (nstrain.data_parallel), as NCCL's teardown waits for the graphs that hold its kernels (an engine that was not
    released, or the graphs of an engine whose construction failed after a capture)."""
    for g in list(_LIVE):
        g.release()


def capturing_graph():
    """The :class:`Graph` being captured in this process (between its capture_begin and capture_end), or None.  Kept
    for the process, not the thread: a captured backward runs on autograd's device thread, and the reductions it
    issues there (nstrain.data_parallel) belong to the capture begun on the engine's thread.  One Graph captures at a
    time (the engine captures its graphs one after another); a capture outside Graph made on another thread while a
    Graph captures would be counted as that Graph's."""
    return _CAPTURING


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
        global _CAPTURING
        self.device = torch.cuda.current_device()
        self.g.capture_begin(pool=pool, capture_error_mode=capture_error_mode)
        _CAPTURING = self

    def capture_end(self) -> None:
        global _CAPTURING
        _CAPTURING = None
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

    def alive(self) -> bool:
        """Whether the graph may still be launched (not released)."""
        return self.exec is not None or self.g is not None

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


class FusedAccumulation:
    """The gradients autograd hands the AccumulateGrad nodes of ``params`` in a backward, added into the parameters'
    ``.grad`` (views of the optimizer's flat buffers) by ``torch._foreach_add_`` instead of one add per parameter.

    :meth:`arm`, after a forward and before its backward, puts a pre-hook on the AccumulateGrad node of every one of
    ``params`` that requires a gradient (the node of this forward's graph: autograd makes the nodes in the forward, on
    its stream, and frees them with the graph); the hook keeps the dense gradient it is handed and replaces it with
    none, so the node adds nothing.  The node still calls the parameter's post-accumulate hook (torch 2.9: after an
    undefined gradient too), before the kept gradient is added: a hook that reads ``.grad`` must :meth:`flush` first, as
    nstrain.data_parallel does before every collective.  Left to the node: no gradient (a weight whose gradient the
    linear operator writes itself, :func:`neuronspark.ops.linear.bind_grad_`), a sparse one, and any gradient of a
    parameter without ``.grad``.  The kept gradients are added by :meth:`flush`: when they reach ``limit`` elements, when
    a collective is about to read the buffers and in :meth:`disarm` after the backward (which removes the hooks even if
    the additions fail).  The same fp32 additions as the nodes', so the buffers hold the same bits.  The forward and
    the backward run on one stream, and no AccumulateGrad node of ``params`` is kept alive from another stream (e.g. by
    ``torch.autograd.graph.get_gradient_edge``): the additions are ordered on the stream current at the flush.  Under a
    CUDA-graph capture the hooks run and the additions are captured; a replay runs only the kernels."""

    def __init__(self, params: List[torch.Tensor], limit: int = 1 << 22):
        self.params, self.limit = list(params), int(limit)
        self._kept: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._held = 0
        self._handles: list = []

    def arm(self) -> None:
        for p in self.params:
            if p.requires_grad:
                node = p.view_as(p).grad_fn.next_functions[0][0]
                self._handles.append(node.register_prehook(partial(self._keep, p)))

    def _keep(self, p: torch.Tensor, grads):
        g = grads[0]
        if g is None or g.layout != torch.strided or p.grad is None:   # (none: also kept by an earlier hook)
            return None
        self._kept.append((p, g))
        self._held += g.numel()
        if self._held >= self.limit:
            self.flush()
        return (None,)

    def flush(self) -> None:
        if self._kept:
            kept, self._kept, self._held = self._kept, [], 0
            torch._foreach_add_([p.grad for p, _ in kept], [g for _, g in kept])

    def disarm(self) -> None:
        try:
            self.flush()
        finally:
            for h in self._handles:
                h.remove()
            self._handles = []
            self._kept, self._held = [], 0


class StepEngine:
    def __init__(self, model, optimizer: FlatAdamW, rows: int, seq_len: int, accum: int,
                 grad_clip: float = 1.0, warmup_iters: int = 3, compile: bool = False, arena=None,
                 before_capture=None, data_parallel=None, after_attach=None):
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
        self.first = optimizer.written_first()                       # armed in an update's first micro-batch
        self.fused = FusedAccumulation(optimizer.params)             # (a bound weight's node gets no gradient)
        from nstrain.gpu import Arena
        self.arena = arena if arena is not None else Arena()          # the trainer's: memory it already holds
        if self.rep is not None:
            self.rep.attach(model, optimizer, self.arena.stream, overwrite=bool(self.first), fused=self.fused)
        if after_attach is not None:                                 # the replicas' memory outside the allocator (the
            after_attach()                                           # reduction's staging, the bf16 buffers) is taken:
                                                                     # the trainer holds the rest of the GPU now
        self.sharding = optimizer.sharding                           # the split the graphs are captured for
        self.bf16_at = optimizer.bf16_buffers()                      # and the bf16 buffers they read and write
        self._capture(warmup_iters, before_capture)

    # ------------------------------------------------------------------ bodies (eager and captured)
    def _backward(self, loss: torch.Tensor, first: bool) -> None:
        """``loss.backward()`` with autograd's accumulation fused (:class:`FusedAccumulation`); ``first``: the update's
        first micro-batch, the ``written_first`` weights armed (their first GEMM writes the gradient) and refused if
        one is left armed."""
        if not (first and self.first):
            try:
                self.fused.arm()
                loss.backward()
            finally:
                self.fused.disarm()
            return
        from neuronspark.ops.linear import arm_grad_overwrite_, armed_grad_overwrites
        arm_grad_overwrite_(self.first)
        try:
            self.fused.arm()
            loss.backward()
        finally:
            try:
                self.fused.disarm()
            finally:
                left = armed_grad_overwrites(self.first)
                arm_grad_overwrite_(left, armed=False)
        if left:
            names = {id(p): n for n, p in self.model.named_parameters()}
            raise RuntimeError(f"StepEngine: {len(left)} weight(s) the optimizer lists as written first got no "
                               f"gradient from the linear operator in the backward: {[names.get(id(p)) for p in left]}")

    def _micro(self, first: bool = False) -> None:
        out = self.loss_fn(self.x, self.y)
        self._backward(out.loss * self.inv_accum, first)
        self.loss_sum.add_(out.loss.detach())
        self.ce_sum.add_(out.ce.detach())
        self.aux_sum.add_(out.aux.detach())

    def _micro_last(self, first: bool = False) -> None:
        self.rep.begin()
        out = self.loss_fn(self.x, self.y)
        self.rep.forward_done()
        self._backward(out.loss * self.inv_accum, first)
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
        self.opt.step(coef, keep=True)                               # (the next update's first backward writes the
        self._zero_sums()                                            # written-first weights' gradients)

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
                    self._micro(first=True)
                    self._micro()
                else:
                    self._micro(first=True)
                    self._micro_last()
                self._norm()
                self._step()
        torch.cuda.synchronize()
        if before_capture is not None:                               # eager work whose first-call caches must be
            self.opt.gather_()                                       # (its reads of the fp32 weights: whole)
            before_capture()                                         # live before the arena's free blocks become
        bodies = [("micro_first", lambda: self._micro(first=True)), ("micro", self._micro)]
        if self.rep is not None:
            bodies += [("micro_last", self._micro_last), ("micro_only", lambda: self._micro_last(first=True))]
        bodies += [("norm", self._norm), ("step", self._step)]
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
            self.micro_batch(inputs[sl], targets[sl], first=m == 0, last=m == accum - 1)
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

    def micro_batch(self, x: torch.Tensor, y: torch.Tensor, first: bool, last: bool) -> None:
        """Forward and backward of one micro-batch of pinned (rows, L) inputs / targets, its gradient added into the
        flat gradient buffers; ``first``: the update's first micro-batch, whose backward writes the gradients of the
        ``written_first`` weights (the step left them); ``last``: the update's last micro-batch, whose backward sums
        the replicas' gradients (nothing is reduced in the others)."""
        self._check_split()
        self.x.copy_(x, non_blocking=True)
        self.y.copy_(y, non_blocking=True)
        if self.rep is not None and last:
            name = "micro_only" if first else "micro_last"
        else:
            name = "micro_first" if first else "micro"
        self.graphs[name].replay()

    def release(self) -> None:
        """Release the graphs and zero the gradients (the written-first weights' were left by the last step)."""
        for g in self.graphs.values():
            g.release()
        self.graphs.clear()
        self.opt.zero_grad()
        if self.rep is not None:
            self.rep.detach()
