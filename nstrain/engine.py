"""The training step as CUDA graphs, for one shape (micro-batch rows x sequence length).

Three graphs share one memory pool:

* ``micro``: forward + loss + backward of one micro-batch from the static input buffers; the loss is scaled by
  ``1 / accum`` and the gradients accumulate in the flat gradient buffers of :class:`FlatAdamW` (the parameters'
  ``.grad`` are views into them); the loss terms accumulate in static sums.
* ``norm``: the global gradient norm (fp32) into a static scalar.  The host reads it once per update: a non-finite
  norm skips the update (the gradients are zeroed, the optimizer and the weights are untouched).
* ``step``: clip by that norm and the AdamW kernels (which also zero the gradients); the sums are zeroed.

The optimizer state is created at construction; the warm-up before capture runs at learning rate 0 (weights unchanged)
and the optimizer state is restored afterwards; buffers the forward may write are snapshotted and restored as well.  A
new shape (a sequence-length change) gets a new engine over the same model and optimizer.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

from nstrain.flat_adamw import FlatAdamW


class StepEngine:
    def __init__(self, model, optimizer: FlatAdamW, rows: int, seq_len: int, accum: int,
                 grad_clip: float = 1.0, warmup_iters: int = 3, compile: bool = False):
        self.model, self.opt = model, optimizer
        self.loss_fn = torch.compile(model.loss, dynamic=False) if compile else model.loss
        self.rows, self.L, self.accum, self.clip = int(rows), int(seq_len), int(accum), float(grad_clip)
        self.dev = next(model.parameters()).device
        self.x = torch.zeros((self.rows, self.L), dtype=torch.int64, device=self.dev)
        self.y = torch.zeros((self.rows, self.L), dtype=torch.int64, device=self.dev)
        f32 = dict(dtype=torch.float32, device=self.dev)
        self.loss_sum, self.ce_sum, self.aux_sum = (torch.zeros((), **f32) for _ in range(3))
        self.gnorm = torch.zeros((), **f32)
        self.graphs: Dict[str, torch.cuda.CUDAGraph] = {}
        self._capture(warmup_iters)

    # ------------------------------------------------------------------ bodies (eager and captured)
    def _micro(self) -> None:
        out = self.loss_fn(self.x, self.y)
        (out.loss / self.accum).backward()
        self.loss_sum.add_(out.loss.detach())
        self.ce_sum.add_(out.ce.detach())
        self.aux_sum.add_(out.aux.detach())

    def _norm(self) -> None:
        self.gnorm.copy_(self.opt.grad_norm())

    def _step(self) -> None:
        coef = torch.clamp(self.clip / (self.gnorm + 1e-6), max=1.0)
        self.opt.step(coef)
        self._zero_sums()

    def _zero_sums(self) -> None:
        for t in (self.loss_sum, self.ce_sum, self.aux_sum):
            t.zero_()

    # ------------------------------------------------------------------ capture
    def _capture(self, warmup_iters: int) -> None:
        # the warm-up runs real steps at learning rate 0; a non-finite warm-up gradient would still reach the weights
        # (0 x NaN), so the parameters are restored too
        params = [p.detach().clone() for p in self.model.parameters()]
        bufs = {n: b.detach().clone() for n, b in self.model.named_buffers()}
        saved = {n: {k: v.detach().clone() for k, v in st.items()} for n, st in self.opt.state().items()}
        saved_lr = self.opt.lr.clone()
        self.opt.lr.zero_()                                          # warm-up steps leave the weights unchanged
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup_iters):
                for _ in range(min(2, self.accum)):
                    self._micro()
                self._norm()
                self._step()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        pool = torch.cuda.graph_pool_handle()
        for name, body in (("micro", self._micro), ("norm", self._norm), ("step", self._step)):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                body()
            self.graphs[name] = g
        torch.cuda.synchronize()
        with torch.no_grad():                                        # undo the warm-up
            for p, q in zip(self.model.parameters(), params):
                p.copy_(q)
            for n, b in self.model.named_buffers():
                b.copy_(bufs[n])
            self.opt.load_state(saved)
            self.opt.lr.copy_(saved_lr)
            self.opt.zero_grad()
            self._zero_sums()

    # ------------------------------------------------------------------ use
    def update(self, inputs: torch.Tensor, targets: torch.Tensor, base_lr: float) -> Tuple[dict, bool]:
        """One optimizer update over ``accum`` micro-batches of the pinned (accum * rows, L) inputs / targets.
        Returns (metrics, applied); a non-finite gradient norm skips the update."""
        if tuple(inputs.shape) != (self.accum * self.rows, self.L):
            raise ValueError(f"update of shape {tuple(inputs.shape)}, engine is {self.accum} x {self.rows} x {self.L}")
        for m in range(self.accum):
            sl = slice(m * self.rows, (m + 1) * self.rows)
            self.x.copy_(inputs[sl], non_blocking=True)
            self.y.copy_(targets[sl], non_blocking=True)
            self.graphs["micro"].replay()
        self.graphs["norm"].replay()
        sums = torch.stack([self.gnorm, self.loss_sum, self.ce_sum, self.aux_sum]).tolist()
        gn = sums[0]
        metrics = {"loss": sums[1] / self.accum, "ce": sums[2] / self.accum, "aux": sums[3] / self.accum,
                   "grad_norm": gn}
        if not math.isfinite(gn) or not math.isfinite(metrics["loss"]):
            self.opt.zero_grad()
            self._zero_sums()
            return metrics, False
        self.opt.lr.fill_(base_lr)
        self.graphs["step"].replay()
        return metrics, True

    def release(self) -> None:
        self.graphs.clear()
