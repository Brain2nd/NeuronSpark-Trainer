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

import math
from typing import Dict, Tuple

import torch

from nstrain.flat_adamw import FlatAdamW


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
        self.graphs: Dict[str, torch.cuda.CUDAGraph] = {}
        from nstrain.gpu import Arena
        self.arena = arena if arena is not None else Arena()          # the trainer's: memory it already holds
        if self.rep is not None:
            self.rep.attach(model, optimizer, self.arena.stream)
        self.sharding = optimizer.sharding                           # the split the graphs are captured for
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
        # the warm-up runs real steps at learning rate 0; a non-finite warm-up gradient would still reach the weights
        # (0 x NaN), so the parameters are restored too
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
            g = torch.cuda.CUDAGraph()
            with torch.cuda.stream(a.stream):
                g.capture_begin(pool=a.pool.id, capture_error_mode="thread_local")
                try:
                    body()
                finally:                                             # (as torch.cuda.graph's exit: a failed body
                    g.capture_end()                                  # still ends the capture)
            self.graphs[name] = g
        torch.cuda.synchronize()
        self._restore(saved)                                         # undo the warm-up
        with a.use():
            self.opt.zero_grad()
            self._zero_sums()
        torch.cuda.synchronize()

    def _snapshot(self) -> dict:
        """The state the warm-up (and the validation before the capture) may change, copied to the host: the
        optimizer's flat buffers (the trainable weights ``p`` and the moments ``m``, ``v``, one contiguous copy per
        group), the parameters outside them (none trainable), the model's buffers, the step count and the learning
        rate."""
        torch.cuda.synchronize()                                     # (the last engine's work on them: done)
        flat = [{k: g["flat"][k].to("cpu") for k in ("p", "m", "v")} for g in self.opt.groups]
        others = {n: p.detach().to("cpu") for n, p in self.model.named_parameters() if n not in self.opt.slots}
        bufs = {n: b.detach().to("cpu") for n, b in self.model.named_buffers()}
        torch.cuda.synchronize()
        return {"flat": flat, "others": others, "bufs": bufs, "t": self.opt.t.to("cpu"), "bc": self.opt.bc.to("cpu"),
                "lr": self.opt.lr.to("cpu"), "ema_c": self.opt.ema_c.to("cpu"), "stale": self.opt.stale}

    @torch.no_grad()
    def _restore(self, saved: dict) -> None:
        for g, h in zip(self.opt.groups, saved["flat"]):
            for k in ("p", "m", "v"):
                g["flat"][k].copy_(h[k])
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
        """Refuse to replay graphs captured for another split of the optimizer among the replicas: they would reduce
        and update the old parts while the optimizer reads the new ones."""
        if self.opt.sharding is not self.sharding:
            raise RuntimeError("StepEngine: the optimizer's split among the data-parallel replicas changed since this "
                               "engine was captured (FlatAdamW.shard / unshard): release it and build a new engine")

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
        self.graphs.clear()
        if self.rep is not None:
            self.rep.detach()
