"""AdamW over flat fp32 buffers, one Triton kernel per parameter group.

Every parameter group gets four flat fp32 buffers: parameters, gradients, first and second moments.  The model's
parameters become views into the parameter buffer and their ``.grad`` views into the gradient buffer (offsets padded to
64 elements), so autograd accumulates straight into the flat gradients, the gradient norm is one reduction per group
and the update is one kernel per group that reads the gradient once: clip coefficient, moments, decoupled weight decay,
bias-corrected step, and the gradient zeroed for the next update.

Update rule (torch.optim.AdamW, decoupled weight decay)::

    g = g * clip;  m = b1 m + (1 - b1) g;  v = b2 v + (1 - b2) g^2
    p = p (1 - lr wd);  p = p - lr / (1 - b1^t) * m / (sqrt(v) / sqrt(1 - b2^t) + eps)

``lr`` is a device tensor (base rate; each group multiplies by its ``lr_mult``), ``t`` a device step counter, so the
step can be captured in a CUDA graph.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import triton
import triton.language as tl

ALIGN = 64


@triton.jit
def _adamw_kernel(P, G, M, V, n, lr_ptr, coef_ptr, bc_ptr, lr_mult, wd, b1, b2, eps, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    lr = tl.load(lr_ptr) * lr_mult
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


class FlatAdamW:
    BLOCK = 2048

    def __init__(self, model, groups: List[dict], betas=(0.9, 0.95), eps: float = 1e-8):
        self.b1, self.b2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        names = {id(p): n for n, p in model.named_parameters()}
        dev = groups[0]["params"][0].device
        self.lr = torch.zeros((), dtype=torch.float32, device=dev)      # base learning rate
        self.t = torch.zeros((), dtype=torch.float32, device=dev)       # updates applied
        self.bc = torch.zeros(2, dtype=torch.float32, device=dev)
        self.groups = []
        self.slots: Dict[str, tuple] = {}                                # name -> (group index, offset, numel, shape)
        for gi, g in enumerate(groups):
            params = g["params"]
            offs, o = [], 0
            for p in params:
                offs.append(o)
                o += (p.numel() + ALIGN - 1) // ALIGN * ALIGN
            n = o
            flat = {k: torch.zeros(n, dtype=torch.float32, device=dev) for k in ("p", "g", "m", "v")}
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

    # ------------------------------------------------------------------ captured pieces
    def grad_norm(self) -> torch.Tensor:
        return torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g["flat"]["g"]) for g in self.groups]))

    def step(self, coef: torch.Tensor) -> None:
        """One update with gradients scaled by the device scalar ``coef``; zeroes the gradients."""
        self.t.add_(1.0)
        self.bc[0].copy_(1.0 - torch.pow(self.b1, self.t))
        self.bc[1].copy_(torch.sqrt(1.0 - torch.pow(self.b2, self.t)))
        for g in self.groups:
            f = g["flat"]
            grid = (triton.cdiv(g["n"], self.BLOCK),)
            _adamw_kernel[grid](f["p"], f["g"], f["m"], f["v"], g["n"], self.lr, coef, self.bc, g["lr_mult"], g["wd"],
                                self.b1, self.b2, self.eps, BLOCK=self.BLOCK, num_warps=8)

    def zero_grad(self) -> None:
        for g in self.groups:
            g["flat"]["g"].zero_()

    # ------------------------------------------------------------------ state by parameter name
    def state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        out = {}
        for name, (gi, off, k, shape) in self.slots.items():
            f = self.groups[gi]["flat"]
            out[name] = {"exp_avg": f["m"][off:off + k].view(shape), "exp_avg_sq": f["v"][off:off + k].view(shape),
                         "step": self.t}
        return out

    def load_state(self, state: Dict[str, Dict[str, torch.Tensor]]) -> None:
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

    def reset_state(self) -> None:
        with torch.no_grad():
            for g in self.groups:
                g["flat"]["m"].zero_()
                g["flat"]["v"].zero_()
            self.t.zero_()
