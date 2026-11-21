"""AdamW over flat fp32 buffers, one Triton kernel per parameter group.

Every parameter group gets four flat fp32 buffers: parameters, gradients, first and second moments (and, with
``ema``, a fifth: the exponential moving average of the parameters).  The model's parameters become views into the
parameter buffer and their ``.grad`` views into the gradient buffer (offsets padded to 64 elements), so autograd
accumulates straight into the flat gradients, the gradient norms are one pass over the buffers (per parameter
tensor, fused; the groups' and any bucketing of the tensors -- the model's layers -- are sums of squares of those in
a fixed order, :meth:`FlatAdamW.grad_norms`) and the update is one kernel per group that reads the gradient once:
clip coefficient, moments, decoupled weight decay, bias-corrected step, the moving average, and the gradient zeroed for
the next update.
An ``nn.Embedding`` of the model whose weight is held here is set to ``sparse``: the gradient of its lookup is then the
rows that were read, added into the flat gradient view in place (no dense gradient of the whole table, no sort, no
add over the whole table; rows read several times are summed in no fixed order).  An embedding with a padding index,
a norm bound or gradients scaled by frequency is left as it is.

A group with ``"linear": True`` holds the weights of the model's linear operator (``neuronspark.ops.linear``): its
2-D parameters are bound with the operator's ``bind_grad_``, so the operator's backward accumulates each weight
gradient into the flat gradient buffer with its own GEMM (no temporary, no separate add).  The group also has a flat
bf16 buffer ``h`` holding the bf16 values of its parameters: the kernel stores them with the new weights in the same
pass, and the operator's training forward reads each weight's part of it instead of casting the weight
(``bind_train_bf16_``).  A 2-D parameter (N, K) of such a group occupies ``padded_extent(N) * K`` elements of every
buffer (the operator's padded extent of N; the elements past the parameter are zero in every buffer and stay zero
through the update: ``eps`` must be positive in fp32), so its bf16 part is the operator's padded weight.  ``h``
follows every write of the weights made here (the update, :meth:`swap_ema`); after any other write of a weight call
:meth:`refresh_bf16` (the step engine does before it captures and after it restores its snapshot, and refuses an
update when a weight was written in place through its parameter since, :meth:`check_bf16`).

Update rule (torch.optim.AdamW, decoupled weight decay)::

    g = g * clip;  m = b1 m + (1 - b1) g;  v = b2 v + (1 - b2) g^2
    p = p (1 - lr wd);  p = p - lr / (1 - b1^t) * m / (sqrt(v) / sqrt(1 - b2^t) + eps)
    e = e + c (p - e)                        (with ema; c = 0 leaves e as it is)

``lr`` is a device tensor (base rate; each group multiplies by its ``lr_mult``), ``wd`` a device tensor scaling every
group's weight decay (1: the groups' values; a batch-size ramp scales it with the batch, nstrain.batch), ``ema_c``
the device tensor ``c`` of the moving average (the update's tokens over the average's horizon in tokens,
nstrain.trainer; the average starts as a copy of the parameters, :meth:`ema_init`), ``t`` a device step counter, so
the step can be captured in a CUDA graph.

Sharded state (:meth:`shard`, data-parallel replicas, nstrain.data_parallel): ranges of a linear group's buffers whose
weights the training forward reads only through their bf16 values are split among the replicas in equal parts.  A
replica updates its part of every such range (its reduction leaves the summed gradient there and zeroes the rest of
the range) and every range outside them (replicated: every replica holds the summed gradient and updates it); the bf16
values of the split ranges are then gathered from their owners, so they are whole on every replica after every update.
The weights, moments and moving average of the parts other replicas own are left as they were (stale) until
:meth:`gather_` brings them from the owners.  The gradient norms are the sums of squares of each replica's parts
(the replicated ranges split among the replicas too), gathered and added in rank order.
"""
from __future__ import annotations

import bisect
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import triton
import triton.language as tl

ALIGN = 64
SHARD_ALIGN = 16                    # the boundaries of the replicas' parts: multiples of 16 elements


@triton.jit
def _adamw_kernel(P, G, M, V, E, H, n, starts, ends, lr_ptr, wd_ptr, c_ptr, coef_ptr, bc_ptr, lr_mult, wd_group, b1,
                  b2, eps, EMA: tl.constexpr, BF16: tl.constexpr, RAGGED: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    if RAGGED:                             # the program's block [start, end) of a range (multiples of SHARD_ALIGN)
        lo = tl.multiple_of(tl.load(starts + pid), 16)
        hi = tl.multiple_of(tl.load(ends + pid), 16)
        offs = lo + tl.arange(0, BLOCK)
        mask = offs < hi
    else:
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
    if BF16:
        tl.store(H + offs, p.to(tl.bfloat16), mask=mask)
    tl.store(G + offs, tl.zeros_like(g), mask=mask)
    if EMA:
        c = tl.load(c_ptr)
        e = tl.load(E + offs, mask=mask, other=0.0)
        tl.store(E + offs, e + c * (p - e), mask=mask)


def _segments(ids: List[int], n: int, dev) -> torch.Tensor:
    """(n, longest) int64: row s lists the positions i with ``ids[i] == s``, padded with ``len(ids)``; a sum over the
    rows of a vector gathered with it adds every segment's values in a fixed order (unlike ``index_add_``, whose
    atomic additions have none: data-parallel replicas compute the same norms from the same gradients)."""
    rows = [[i for i, s in enumerate(ids) if s == k] for k in range(n)]
    width = max([len(r) for r in rows] + [1])
    return torch.tensor([r + [len(ids)] * (width - len(r)) for r in rows], dtype=torch.int64, device=dev)


def _blocks(ranges: List[Tuple[int, int]], block: int, dev) -> Tuple[torch.Tensor, torch.Tensor]:
    """(starts, ends), int64: one entry per program of an update over the element ranges ``ranges`` of a flat buffer:
    the first element of the program's block and the end of the block (the range's end in its last block)."""
    starts = [torch.arange(lo, hi, block, dtype=torch.int64) for lo, hi in ranges]
    ends = [torch.clamp(s + block, max=hi) for s, (_, hi) in zip(starts, ranges)]
    return torch.cat(starts).to(dev), torch.cat(ends).to(dev)


class FlatAdamW:
    BLOCK = 2048

    def __init__(self, model, groups: List[dict], betas=(0.9, 0.95), eps: float = 1e-8, ema: bool = False,
                 bucket=None):
        """``ema``: keep the parameters' exponential moving average (``e``); ``bucket``: a function of a parameter's
        name giving the bucket its gradient norm is summed into (:meth:`grad_norms`; None: no buckets)."""
        self.b1, self.b2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        if not float(torch.tensor(self.eps, dtype=torch.float32)) > 0.0:
            raise ValueError(f"eps {eps} is not positive in fp32 (the update of an element whose gradient and moments "
                             "are zero, the padding of the flat buffers among them, would be 0 / 0)")
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
        self.sharding: Optional[dict] = None                             # the replicas' parts (:meth:`shard`)
        self.stale = False                                               # split: the parts other replicas own have
        #                                                                  been updated since the last gather_
        keys = ("p", "g", "m", "v") + (("e",) if self.ema else ())
        for gi, g in enumerate(groups):
            params = g["params"]
            linear = bool(g.get("linear"))
            if linear:
                from neuronspark.ops.linear import bind_grad_, padded_extent
            offs, o = [], 0
            for p in params:
                offs.append(o)
                extent = padded_extent(p.shape[0]) * p.shape[1] if linear and p.ndim == 2 else p.numel()
                o += (extent + ALIGN - 1) // ALIGN * ALIGN
            n = o
            flat = {k: torch.zeros(n, dtype=torch.float32, device=dev) for k in keys}
            if linear:
                flat["h"] = torch.zeros(n, dtype=torch.bfloat16, device=dev)
            with torch.no_grad():
                for p, off in zip(params, offs):
                    if p.dtype != torch.float32 or not p.is_contiguous():
                        raise TypeError(f"{names[id(p)]}: fp32 contiguous parameters only")
                    k = p.numel()
                    flat["p"][off:off + k].copy_(p.reshape(-1))
                    p.data = flat["p"][off:off + k].view(p.shape)
                    p.grad = flat["g"][off:off + k].view(p.shape)
                    self.slots[names[id(p)]] = (gi, off, k, tuple(p.shape))
            bf16 = []                                                # (parameter, its (padded N, K) part of h)
            if linear:
                for p, off in zip(params, offs):
                    if p.ndim == 2:
                        bind_grad_(p)
                        rows = padded_extent(p.shape[0])
                        bf16.append((p, flat["h"][off:off + rows * p.shape[1]].view(rows, p.shape[1])))
            self.groups.append({"name": g.get("name", str(gi)), "flat": flat, "n": n,
                                "lr_mult": float(g.get("lr_mult", 1.0)), "wd": float(g.get("weight_decay", 0.0)),
                                "params": params, "linear": linear, "bf16": bf16, "bf16_versions": []})
        self.params = [p for g in self.groups for p in g["params"]]
        self._names = names
        self.refresh_bf16()
        # a lookup table held here receives the gradient of its lookup as the rows that were read: autograd adds the
        # sparse gradient into the table's flat gradient view in place
        held = {id(p) for p in self.params}
        for m in model.modules():
            if isinstance(m, torch.nn.Embedding) and id(m.weight) in held and m.padding_idx is None \
                    and m.max_norm is None and not m.scale_grad_by_freq:
                m.sparse = True
        # the gradient norms' pieces: every parameter's slice of its group's gradient buffer, its group, its bucket
        self._slices = [self.groups[gi]["flat"]["g"][off:off + k] for gi, off, k, _ in self.slots.values()]
        self._slice_group = _segments([gi for gi, _, _, _ in self.slots.values()], len(self.groups), dev)
        self.buckets: List[str] = []
        self.slot_bucket: Dict[str, str] = {}
        if bucket is not None:
            self.slot_bucket = {name: bucket(name) for name in self.slots}
            self.buckets = sorted(set(self.slot_bucket.values()))
            self._slice_bucket = _segments([self.buckets.index(self.slot_bucket[name]) for name in self.slots],
                                           len(self.buckets), dev)

    # ------------------------------------------------------------------ captured pieces
    def grad_norms(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """(the gradient norm of every group in the groups' order, the norm of every bucket in ``buckets``' order),
        fp32: one fused pass over the parameters' slices of the gradient buffers (``torch._foreach_norm``), the
        groups' and the buckets' norms the square roots of the sums of squares.  Sharded: the pass is over this
        replica's pieces of the parameters (:meth:`shard`), and every parameter's sum of squares is the sum of the
        replicas' in rank order (an all-gather: the same values on every replica)."""
        sh = self.sharding
        if sh is None:
            sq = torch.stack(torch._foreach_norm(self._slices)).square()
        else:
            mine = torch.zeros(len(self._slices), dtype=torch.float32, device=self.t.device)
            if sh["pieces"]:
                mine.index_copy_(0, sh["piece_slot"], torch.stack(torch._foreach_norm(sh["pieces"])).square())
            every = torch.empty(sh["world"] * mine.numel(), dtype=torch.float32, device=mine.device)
            dist.all_gather_into_tensor(every, mine, group=sh["group"])
            every = every.view(sh["world"], -1)
            sq = every[0]
            for r in range(1, sh["world"]):
                sq = sq + every[r]
        sq = torch.cat([sq, sq.new_zeros(1)])                        # (the padding of the segments reads the zero)
        groups = sq[self._slice_group].sum(1).sqrt()
        if self.buckets:
            buckets = sq[self._slice_bucket].sum(1).sqrt()
        else:
            buckets = torch.zeros(0, dtype=torch.float32, device=sq.device)
        return groups, buckets

    def grad_norm(self) -> torch.Tensor:
        return torch.linalg.vector_norm(self.grad_norms()[0])

    def step(self, coef: torch.Tensor) -> None:
        """One update with gradients scaled by the device scalar ``coef``; zeroes the gradients; moves the average
        by ``ema_c``.  Sharded: this replica's parts of the split ranges first, then the replicated ranges while the
        bf16 values of the split ranges are gathered from their owners (on a side stream the step waits for)."""
        self.t.add_(1.0)
        self.bc[0].copy_(1.0 - torch.pow(self.b1, self.t))
        self.bc[1].copy_(torch.sqrt(1.0 - torch.pow(self.b2, self.t)))
        sh = self.sharding
        if sh is None:
            for g in self.groups:
                self._update(g, coef)
            return
        self.stale = True                                            # (a replay: the engine marks it)
        for gi, starts, ends in sh["owned"]:
            self._update(self.groups[gi], coef, starts, ends)
        cur, side = torch.cuda.current_stream(), sh["stream"]
        side.wait_stream(cur)
        with torch.cuda.stream(side):
            self._gather_bf16()
        for gi, starts, ends in sh["replicated"]:
            self._update(self.groups[gi], coef, starts, ends)
        cur.wait_stream(side)

    def _update(self, g: dict, coef: torch.Tensor, starts: Optional[torch.Tensor] = None,
                ends: Optional[torch.Tensor] = None) -> None:
        """The update kernel over the whole group, or over the blocks ``starts`` / ``ends`` (:func:`_blocks`)."""
        f = g["flat"]
        ragged = starts is not None
        grid = (int(starts.numel()) if ragged else triton.cdiv(g["n"], self.BLOCK),)
        _adamw_kernel[grid](f["p"], f["g"], f["m"], f["v"], f["e"] if self.ema else f["p"],
                            f["h"] if g["linear"] else f["p"], g["n"], starts if ragged else f["p"],
                            ends if ragged else f["p"], self.lr, self.wd, self.ema_c, coef, self.bc, g["lr_mult"],
                            g["wd"], self.b1, self.b2, self.eps, EMA=self.ema, BF16=g["linear"], RAGGED=ragged,
                            BLOCK=self.BLOCK, num_warps=8)

    def zero_grad(self) -> None:
        for g in self.groups:
            g["flat"]["g"].zero_()

    # ------------------------------------------------------------------ the linear operator's bf16 weights
    def refresh_bf16(self) -> None:
        """Bring the bf16 values of the linear groups up to the weights as they are now and bind every weight's part
        to it again: after any write of a weight that is not this optimizer's update.  Sharded: the parts other
        replicas own are gathered from them (a collective: every replica calls it)."""
        from_linear = [g for g in self.groups if g["linear"]]
        if not from_linear:
            return
        from neuronspark.ops.linear import bind_train_bf16_
        with torch.no_grad():
            for g in from_linear:
                g["flat"]["h"].copy_(g["flat"]["p"])
            self._gather_bf16()
            for g in from_linear:
                for p, part in g["bf16"]:
                    bind_train_bf16_(p, part)
                g["bf16_versions"] = [p._version for p, _ in g["bf16"]]

    def check_bf16(self) -> None:
        """Raise when a weight of a linear group was written in place through its parameter since its bf16 values
        were last brought up to date: a captured step would compute with the values before the write."""
        for g in self.groups:
            for (p, _), version in zip(g["bf16"], g["bf16_versions"]):
                if p._version != version:
                    raise RuntimeError(f"nstrain.flat_adamw: {self._names[id(p)]} was written in place outside the "
                                       "optimizer and its bf16 values are those before the write: call refresh_bf16()")

    def bf16_copies(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Per parameter name of the linear groups: (the parameter, its bf16 part ``(padded N, K)``)."""
        return {self._names[id(p)]: (p, part) for g in self.groups for p, part in g["bf16"]}

    # ------------------------------------------------------------------ the moving average
    def ema_init(self) -> None:
        """The average starts as a copy of the parameters."""
        with torch.no_grad():
            for g in self.groups:
                g["flat"]["e"].copy_(g["flat"]["p"])

    def swap_ema(self) -> None:
        """Exchange the parameters and their average (the model then computes with the average; again to undo).
        Sharded: both are gathered whole first (a collective)."""
        self.gather_()
        with torch.no_grad():
            for g in self.groups:
                f = g["flat"]
                tmp = f["p"].clone()
                f["p"].copy_(f["e"])
                f["e"].copy_(tmp)
                del tmp
        self.refresh_bf16()

    # ------------------------------------------------------------------ state by parameter name
    def state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Per parameter name: ``exp_avg``, ``exp_avg_sq``, ``step`` (and ``ema`` with the moving average); views of
        the buffers (sharded: whole after :meth:`gather_`, refused before it)."""
        if self.stale:
            raise RuntimeError("FlatAdamW.state: the parts other replicas own were updated by them since the last "
                               "gather_ (sharded): call gather_() on every replica first")
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

    # ------------------------------------------------------------------ the replicas' parts (data parallel)
    def shard(self, group, ranges: Dict[int, List[Tuple[int, int]]]) -> None:
        """Split the element ranges ``ranges[gi]`` of linear group gi's buffers among the replicas of the process
        group ``group`` (None: the default group) in equal parts: replica r of W owns the r-th W-th of every range
        (whose summed gradient its reduction leaves there), updates it and nothing else of the range, and the range's
        bf16 values are gathered from the owners after every update; every other range is replicated (see the module
        documentation).  The ranges must be weights the training forward reads only through their bf16 values, with
        boundaries at multiples of ``SHARD_ALIGN`` elements per replica.  A layout other than the present one gathers
        the present one's state first (:meth:`gather_`), so the state is whole when the new parts are taken; the same
        layout again changes nothing.  No ranges: not sharded."""
        rank, world = dist.get_rank(group), dist.get_world_size(group)
        layout = tuple(sorted((int(gi), tuple(sorted((int(lo), int(hi)) for lo, hi in rs)))
                              for gi, rs in ranges.items() if rs))
        spec = (group, rank, world, layout)
        if self.sharding is not None and self.sharding["spec"] == spec:
            return
        self.unshard()
        if not layout:
            return
        dev = self.t.device
        buckets = []                                                 # (group, lo, hi, own lo, own hi)
        for gi, rs in layout:
            g = self.groups[gi]
            if not g["linear"]:
                raise ValueError(f"FlatAdamW.shard: group {g['name']} is not a linear group (its weights are read in "
                                 "fp32 by the forward)")
            end = 0
            for lo, hi in rs:
                if lo < end or hi > g["n"] or hi <= lo or lo % SHARD_ALIGN or (hi - lo) % (SHARD_ALIGN * world):
                    raise ValueError(f"FlatAdamW.shard: range [{lo}, {hi}) of group {g['name']} (n {g['n']}) "
                                     f"overlaps another, leaves the buffer or is not cut into {world} parts of "
                                     f"multiples of {SHARD_ALIGN} elements")
                c = (hi - lo) // world
                buckets.append((gi, lo, hi, lo + rank * c, lo + (rank + 1) * c))
                end = hi
        owned, replicated = [], []
        for gi, g in enumerate(self.groups):
            cut = [(lo, hi, olo, ohi) for bgi, lo, hi, olo, ohi in buckets if bgi == gi]
            if not cut:
                replicated.append((gi, None, None))
                continue
            owned.append((gi,) + _blocks([(olo, ohi) for _, _, olo, ohi in cut], self.BLOCK, dev))
            rest, at = [], 0
            for lo, hi, _, _ in cut:
                if lo > at:
                    rest.append((at, lo))
                at = hi
            if at < g["n"]:
                rest.append((at, g["n"]))
            if rest:
                replicated.append((gi,) + _blocks(rest, self.BLOCK, dev))
        # the gradient norms' pieces: a parameter in a split range, its intersection with this replica's part; any
        # other parameter (its summed gradient on every replica), its rank-th of world consecutive pieces
        lows = {gi: [b[1] for b in buckets if b[0] == gi] for gi in range(len(self.groups))}
        rows = {gi: [b for b in buckets if b[0] == gi] for gi in range(len(self.groups))}
        pieces, slot = [], []
        for i, (name, (gi, off, k, _)) in enumerate(self.slots.items()):
            j = bisect.bisect_right(lows[gi], off) - 1
            inside = j >= 0 and off < rows[gi][j][2]
            if inside:
                _, lo, hi, olo, ohi = rows[gi][j]
                if off + k > hi:
                    raise ValueError(f"FlatAdamW.shard: {name} crosses the end of the range [{lo}, {hi})")
                a, z = max(off, olo), min(off + k, ohi)
            else:
                if j + 1 < len(rows[gi]) and rows[gi][j + 1][1] < off + k:
                    raise ValueError(f"FlatAdamW.shard: {name} crosses the start of a range")
                a, z = off + k * rank // world, off + k * (rank + 1) // world
            if z > a:
                pieces.append(self.groups[gi]["flat"]["g"][a:z])
                slot.append(i)
        self.sharding = {"spec": spec, "group": group, "rank": rank, "world": world, "buckets": buckets,
                         "owned": owned, "replicated": replicated, "pieces": pieces,
                         "piece_slot": torch.tensor(slot, dtype=torch.int64, device=dev),
                         "stream": torch.cuda.Stream(device=dev)}

    def gather_(self) -> None:
        """Sharded: bring the weights, moments and moving average of the parts other replicas own from their owners,
        so that every replica's buffers are whole (and the same): before reading :meth:`state` or the weights in
        fp32 (a checkpoint, an evaluation, a comparison of replicas).  A collective: every replica calls it.  No-op
        when not sharded."""
        sh = self.sharding
        if sh is None:
            return
        keys = ("p", "m", "v") + (("e",) if self.ema else ())
        with torch.no_grad():
            for gi, lo, hi, olo, ohi in sh["buckets"]:
                f = self.groups[gi]["flat"]
                for k in keys:
                    dist.all_gather_into_tensor(f[k][lo:hi], f[k][olo:ohi], group=sh["group"])
        self.stale = False

    def unshard(self) -> None:
        """Gather the state whole (:meth:`gather_`) and leave the sharded layout: every replica updates everything."""
        self.gather_()
        self.sharding = None

    def _gather_bf16(self) -> None:
        """Sharded: the bf16 values of every split range from the parts' owners (in place, on the current stream)."""
        sh = self.sharding
        if sh is None:
            return
        for gi, lo, hi, olo, ohi in sh["buckets"]:
            h = self.groups[gi]["flat"]["h"]
            dist.all_gather_into_tensor(h[lo:hi], h[olo:ohi], group=sh["group"])
