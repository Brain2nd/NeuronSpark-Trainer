"""AdamW over flat fp32 buffers, one Triton kernel per parameter group.

Every parameter group gets four flat fp32 buffers: parameters, gradients, first and second moments (and, with
``ema``, a fifth: the exponential moving average of the parameters).  The model's parameters become views into the
parameter buffer and their ``.grad`` views into the gradient buffer (offsets padded to 64 elements), so autograd
accumulates straight into the flat gradients, the gradient norms are one pass over the buffers (per parameter
tensor, fused; the groups' and any bucketing of the tensors -- the model's layers -- are sums of squares of those,
:meth:`FlatAdamW.grad_norms`) and the update is one kernel per group that reads the gradient once: clip coefficient,
moments, decoupled weight decay, bias-corrected step, the moving average, and the gradient zeroed for the next update.

Update rule (torch.optim.AdamW, decoupled weight decay)::

    g = g * clip;  m = b1 m + (1 - b1) g;  v = b2 v + (1 - b2) g^2
    p = p (1 - lr wd);  p = p - lr / (1 - b1^t) * m / (sqrt(v) / sqrt(1 - b2^t) + eps)
    e = e + c (p - e)                        (with ema; c = 0 leaves e as it is)

``lr`` is a device tensor (base rate; each group multiplies by its ``lr_mult``), ``wd`` a device tensor scaling every
group's weight decay (1: the groups' values; a batch-size ramp scales it with the batch, nstrain.batch), ``ema_c``
the device tensor ``c`` of the moving average (the update's tokens over the average's horizon in tokens,
nstrain.trainer; the average starts as a copy of the parameters, :meth:`ema_init`), ``t`` a device step counter, so
the step can be captured in a CUDA graph.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import triton
import triton.language as tl

ALIGN = 64


@triton.jit
def _adamw_kernel(P, G, M, V, E, n, lr_ptr, wd_ptr, c_ptr, coef_ptr, bc_ptr, lr_mult, wd_group, b1, b2, eps,
                  EMA: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    lr = tl.load(lr_ptr) * lr_mult
    wd = tl.load(wd_ptr) * wd_group
    coef = tl.load(coef_ptr)
    bc1 = tl.load(bc_ptr)                  # 1 - b1^t
    bc2_sqrt = tl.load(bc_ptr + 1)         # sqrt(1 - b2^t)
    g = tl.load(G + offs, mask=mask, other=0.0) * coef
    m = tl.load(M + offs, mask=mask, other=0.0)
    v = tl.load(V + offs, mask=mask, other=0.0)
    p = tl.load(P + offs, mask=mask, other=0.0)
    m = b1 * m + (1.0 - b1) * g
    v = b2 * v + (1.0 - b2) * g * g
    p = p * (1.0 - lr * wd)
    p = p - (lr / bc1) * m / (tl.sqrt(v) / bc2_sqrt + eps)
    tl.store(M + offs, m, mask=mask)
    tl.store(V + offs, v, mask=mask)
    tl.store(P + offs, p, mask=mask)
    tl.store(G + offs, tl.zeros_like(g), mask=mask)
    if EMA:
        c = tl.load(c_ptr)
        e = tl.load(E + offs, mask=mask, other=0.0)
        tl.store(E + offs, e + c * (p - e), mask=mask)


class FlatAdamW:
    BLOCK = 2048

    def __init__(self, model, groups: List[dict], betas=(0.9, 0.95), eps: float = 1e-8, ema: bool = False,
                 bucket=None):
        """``ema``: keep the parameters' exponential moving average (``e``); ``bucket``: a function of a parameter's
        name giving the bucket its gradient norm is summed into (:meth:`grad_norms`; None: no buckets)."""
        self.b1, self.b2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.ema = bool(ema)
        names = {id(p): n for n, p in model.named_parameters()}
        dev = groups[0]["params"][0].device
        self.lr = torch.zeros((), dtype=torch.float32, device=dev)      # base learning rate
        self.wd = torch.ones((), dtype=torch.float32, device=dev)       # weight decay scale (1: the groups' values)
        self.ema_c = torch.zeros((), dtype=torch.float32, device=dev)   # the moving average's step (0: untouched)
        self.t = torch.zeros((), dtype=torch.float32, device=dev)       # updates applied
        self.bc = torch.zeros(2, dtype=torch.float32, device=dev)
        self.groups = []
        self.slots: Dict[str, tuple] = {}                                # name -> (group index, offset, numel, shape)
        keys = ("p", "g", "m", "v") + (("e",) if self.ema else ())
        for gi, g in enumerate(groups):
            params = g["params"]
            offs, o = [], 0
            for p in params:
                offs.append(o)
                o += (p.numel() + ALIGN - 1) // ALIGN * ALIGN
            n = o
            flat = {k: torch.zeros(n, dtype=torch.float32, device=dev) for k in keys}
            with torch.no_grad():
                for p, off in zip(params, offs):
                    if p.dtype != torch.float32 or not p.is_contiguous():
                        raise TypeError(f"{names[id(p)]}: fp32 contiguous parameters only")
                    k = p.numel()
                    flat["p"][off:off + k].copy_(p.reshape(-1))
                    p.data = flat["p"][off:off + k].view(p.shape)
                    p.grad = flat["g"][off:off + k].view(p.shape)
                    self.slots[names[id(p)]] = (gi, off, k, tuple(p.shape))
            self.groups.append({"name": g.get("name", str(gi)), "flat": flat, "n": n,
                                "lr_mult": float(g.get("lr_mult", 1.0)), "wd": float(g.get("weight_decay", 0.0)),
                                "params": params})
        self.params = [p for g in self.groups for p in g["params"]]
        # the gradient norms' pieces: every parameter's slice of its group's gradient buffer, its group, its bucket
        self._slices = [self.groups[gi]["flat"]["g"][off:off + k] for gi, off, k, _ in self.slots.values()]
        self._slice_group = torch.tensor([gi for gi, _, _, _ in self.slots.values()], device=dev)
        self.buckets: List[str] = []
        self.slot_bucket: Dict[str, str] = {}
        if bucket is not None:
            self.slot_bucket = {name: bucket(name) for name in self.slots}
            self.buckets = sorted(set(self.slot_bucket.values()))
            self._slice_bucket = torch.tensor([self.buckets.index(self.slot_bucket[name]) for name in self.slots],
                                              device=dev)

    # ------------------------------------------------------------------ captured pieces
    def grad_norms(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """(the gradient norm of every group in the groups' order, the norm of every bucket in ``buckets``' order),
        fp32: one fused pass over the parameters' slices of the gradient buffers (``torch._foreach_norm``), the
        groups' and the buckets' norms the square roots of the sums of squares."""
        sq = torch.stack(torch._foreach_norm(self._slices)).square()
        groups = torch.zeros(len(self.groups), dtype=torch.float32, device=sq.device).index_add_(0, self._slice_group,
                                                                                                 sq).sqrt()
        if self.buckets:
            buckets = torch.zeros(len(self.buckets), dtype=torch.float32, device=sq.device).index_add_(
                0, self._slice_bucket, sq).sqrt()
        else:
            buckets = torch.zeros(0, dtype=torch.float32, device=sq.device)
        return groups, buckets

    def grad_norm(self) -> torch.Tensor:
        return torch.linalg.vector_norm(self.grad_norms()[0])

    def step(self, coef: torch.Tensor) -> None:
        """One update with gradients scaled by the device scalar ``coef``; zeroes the gradients; moves the average
        by ``ema_c``."""
        self.t.add_(1.0)
        self.bc[0].copy_(1.0 - torch.pow(self.b1, self.t))
        self.bc[1].copy_(torch.sqrt(1.0 - torch.pow(self.b2, self.t)))
        for g in self.groups:
            f = g["flat"]
            grid = (triton.cdiv(g["n"], self.BLOCK),)
            _adamw_kernel[grid](f["p"], f["g"], f["m"], f["v"], f["e"] if self.ema else f["p"], g["n"], self.lr,
                                self.wd, self.ema_c, coef, self.bc, g["lr_mult"], g["wd"], self.b1, self.b2, self.eps,
                                EMA=self.ema, BLOCK=self.BLOCK, num_warps=8)

    def zero_grad(self) -> None:
        for g in self.groups:
            g["flat"]["g"].zero_()

    # ------------------------------------------------------------------ the moving average
    def ema_init(self) -> None:
        """The average starts as a copy of the parameters."""
        with torch.no_grad():
            for g in self.groups:
                g["flat"]["e"].copy_(g["flat"]["p"])

    def swap_ema(self) -> None:
        """Exchange the parameters and their average (the model then computes with the average; again to undo)."""
        with torch.no_grad():
            for g in self.groups:
                f = g["flat"]
                tmp = f["p"].clone()
                f["p"].copy_(f["e"])
                f["e"].copy_(tmp)
                del tmp

    # ------------------------------------------------------------------ state by parameter name
    def state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Per parameter name: ``exp_avg``, ``exp_avg_sq``, ``step`` (and ``ema`` with the moving average)."""
        out = {}
        for name, (gi, off, k, shape) in self.slots.items():
            f = self.groups[gi]["flat"]
            out[name] = {"exp_avg": f["m"][off:off + k].view(shape), "exp_avg_sq": f["v"][off:off + k].view(shape),
                         "step": self.t}
            if self.ema:
                out[name]["ema"] = f["e"][off:off + k].view(shape)
        return out

    def load_state(self, state: Dict[str, Dict[str, torch.Tensor]]) -> bool:
        """Load the moments and the step count; the moving average too when the state holds it (returns whether it
        did; a state without it leaves the average as it is)."""
        with torch.no_grad():
            steps = set()
            for name, (gi, off, k, shape) in self.slots.items():
                if name not in state:
                    raise KeyError(f"optimizer state has no entry for {name}")
                f = self.groups[gi]["flat"]
                f["m"][off:off + k].copy_(state[name]["exp_avg"].reshape(-1))
                f["v"][off:off + k].copy_(state[name]["exp_avg_sq"].reshape(-1))
                steps.add(float(state[name]["step"]))
            if len(steps) != 1:
                raise ValueError(f"optimizer state holds different step counts {sorted(steps)}")
            self.t.fill_(steps.pop())
            have = [name for name in self.slots if "ema" in state[name]]
            if have and len(have) != len(self.slots):
                raise ValueError(f"optimizer state holds the moving average of {len(have)} of {len(self.slots)} "
                                 "parameters")
            if have and self.ema:
                for name, (gi, off, k, shape) in self.slots.items():
                    self.groups[gi]["flat"]["e"][off:off + k].copy_(state[name]["ema"].reshape(-1))
            return bool(have) and self.ema

    def reset_state(self) -> None:
        with torch.no_grad():
            for g in self.groups:
                g["flat"]["m"].zero_()
                g["flat"]["v"].zero_()
            self.t.zero_()
