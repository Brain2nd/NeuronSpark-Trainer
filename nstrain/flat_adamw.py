"""AdamW over flat fp32 buffers, one Triton kernel per parameter group.

Every parameter group gets four flat fp32 buffers: parameters, gradients, first and second moments (and, with
``ema``, a fifth: the exponential moving average of the parameters).  The model's parameters become views into the
parameter buffer and their ``.grad`` views into the gradient buffer (offsets padded to 64 elements), so autograd
accumulates straight into the flat gradients, the gradient norms are one pass over the buffers (each parameter's sum of
squares from chunks of its slice, a program per chunk; the groups' and any bucketing of the parameters -- the model's
layers -- are sums of those in a fixed order, :meth:`FlatAdamW.grad_norms`) and the update is one kernel per group
that reads the gradient once:
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
(``bind_train_bf16_``).  The group's ``"written_first"`` leading parameters (nstrain.model.param_groups: the
projections and the tied embedding, whose gradients the operator's GEMM writes before anything else in every backward)
can have their gradients written instead: a step with ``keep`` leaves their part of the gradient buffer as it is
(:meth:`FlatAdamW.step`), and the caller arms them (:meth:`FlatAdamW.written_first`,
``neuronspark.ops.linear.arm_grad_overwrite_``) before the backward of the next update's first micro-batch, whose
GEMMs then write their gradients (beta 0) instead of adding into zeros (nstrain.engine).  A 2-D parameter (N, K) of
such a group occupies ``padded_extent(N) * K`` elements of every
buffer (the operator's padded extent of N; the elements past the parameter are zero in every buffer and stay zero
through the update: ``eps`` must be positive in fp32), so its bf16 part is the operator's padded weight.  ``h``
follows every write of the weights made here (the update, :meth:`swap_ema`); after any other write of a weight call
:meth:`refresh_bf16` (the step engine does before it captures and after it restores its snapshot, and refuses an
update when a weight was written in place through its parameter since, :meth:`check_bf16`).

A group with ``"bound": True`` (the lexical n-gram tables, nstrain.model.param_groups) has its 2-D parameters bound
with ``bind_grad_`` as well: their whole-row read adds the rows of their gradients into the flat gradient buffer itself
(``neuronspark.ops.triton.lex_train``: no dense gradient of the whole table, no add over it), the rows read several
times summed in a fixed order.  The update zeroes their gradients like any other's.

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
values of each split range reach the other replicas as soon as their owner has updated its part (range by range, while
the next range's part is updated), so they are whole on every replica after every update.
The weights, moments and moving average of the parts other replicas own are left as they were (stale) until
:meth:`gather_` brings them from the owners.  The gradient norms are the sums of squares of each replica's parts
(the replicated ranges split among the replicas too), gathered and added in rank order.

How the bf16 values travel.  Replicas that are the whole job, on GPUs of one host that map each other's memory (2 to
``neuronspark.ops.triton.tp_exchange.MAX_GPUS`` of them), keep the bf16 buffers of the split groups in symmetric
memory (``torch.distributed._symmetric_memory``: each replica's buffer mapped into every other's address space; moved
there by the first split, which engines captured before refuse, nstrain.engine): a replica copies its part of a range
into every other replica's buffer at the same offsets (``Tensor.copy_``: a copy engine's copy, which takes no SM from
the update beside it) between two barriers of all the replicas (``neuronspark.ops.triton.tp_bulk.tp_fence`` on a row
of one int64 word per replica in symmetric memory: a replica stores the next epoch into its word of every other
replica's row, then waits until every other word of its own row holds it).  The first, before a replica's first copy
of the step, passes once every replica has finished everything it queued before its step, its forward and backward
among it, so no copy reaches a buffer another replica still reads; the second, at the end of the step, once every
replica's copies have landed, after which every buffer is whole.  Each update writes only its own parts.  Any other
layout of the replicas (no peer access, the GPUs numbered differently by the processes, no symmetric memory to be had)
gathers each range from its owners with NCCL's all-gather.
"""
from __future__ import annotations

import bisect
import uuid
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import triton
import triton.language as tl

ALIGN = 64
SHARD_ALIGN = 16                    # the boundaries of the replicas' parts: multiples of 16 elements
NORM_CHUNK = 16384                  # the gradient norms' pass: elements per program (at most; one parameter's)
NORM_BLOCK = 2048                   # and per step of a program


@triton.jit
def _adamw_kernel(P, G, M, V, E, H, n, keep, starts, ends, lr_ptr, wd_ptr, c_ptr, coef_ptr, bc_ptr, lr_mult, wd_group,
                  b1, b2, eps, EMA: tl.constexpr, BF16: tl.constexpr, RAGGED: tl.constexpr, BLOCK: tl.constexpr):
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
    tl.store(G + offs, tl.zeros_like(g), mask=mask & (offs >= keep))    # (below keep: written by the next writer)
    if EMA:
        c = tl.load(c_ptr)
        e = tl.load(E + offs, mask=mask, other=0.0)
        tl.store(E + offs, e + c * (p - e), mask=mask)


@triton.jit
def _sumsq_kernel(G, starts, ends, out, CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    """out[pid]: the sum of squares of the chunk [start, end) of the flat buffer G (multiples of 16, at most CHUNK
    elements), its elements added in BLOCK-wide steps, then the BLOCK lanes summed: a fixed order."""
    pid = tl.program_id(0).to(tl.int64)
    lo = tl.multiple_of(tl.load(starts + pid), 16)
    hi = tl.multiple_of(tl.load(ends + pid), 16)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.static_range(0, CHUNK, BLOCK):
        offs = lo + i + tl.arange(0, BLOCK)
        x = tl.load(G + offs, mask=offs < hi, other=0.0)
        acc += x * x
    tl.store(out + pid, tl.sum(acc, axis=0))


@triton.jit
def _slot_sums_kernel(part, first, last, sq, BLOCK: tl.constexpr):
    """sq[s]: the sum of slot s's chunk sums part[first[s]:last[s]], added in BLOCK-wide steps, then the lanes (0 for a
    slot with no chunk)."""
    s = tl.program_id(0)
    c0 = tl.load(first + s)
    c1 = tl.load(last + s)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for c in range(c0, c1, BLOCK):
        offs = c + tl.arange(0, BLOCK)
        acc += tl.load(part + offs, mask=offs < c1, other=0.0)
    tl.store(sq + s, tl.sum(acc, axis=0))


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


def _norm_pass(pieces: List[Tuple[int, int, int, int]], n_slots: int, dev) -> dict:
    """The gradient norms' pass over ``pieces`` (group, first element, end, slot; at most one piece per slot, the first
    element a multiple of 16): every piece cut into chunks of at most NORM_CHUNK elements, the piece's end rounded up
    to a multiple of 16 (into its slot's padding, whose gradient stays zero); per group the chunks' starts and ends and
    where in ``part`` their sums go, per slot the range of its chunks' sums (none: the slot's sum is 0), and ``sq``,
    the slots' sums of squares."""
    first, last = [0] * n_slots, [0] * n_slots
    groups, c = [], 0
    for gi in sorted({p[0] for p in pieces}):
        se, base = [], c
        for _, a, z, s in sorted(p for p in pieces if p[0] == gi):
            if a % 16:
                raise ValueError(f"FlatAdamW: a piece of the gradient norms' pass starts at {a}, not a multiple of 16")
            z = -(-z // 16) * 16
            st = torch.arange(a, z, NORM_CHUNK, dtype=torch.int64)
            se.append(torch.stack([st, torch.clamp(st + NORM_CHUNK, max=z)]))
            first[s], c = c, c + st.numel()
            last[s] = c
        se = torch.cat(se, dim=1).to(dev)
        groups.append((gi, se[0].contiguous(), se[1].contiguous(), base, c - base))
    return {"groups": groups, "part": torch.empty(max(c, 1), dtype=torch.float32, device=dev),
            "first": torch.tensor(first, dtype=torch.int32, device=dev),
            "last": torch.tensor(last, dtype=torch.int32, device=dev),
            "sq": torch.empty(n_slots, dtype=torch.float32, device=dev)}


class _PeerBuffers:
    """The symmetric memory of the replicas (:meth:`FlatAdamW._push_setup`): this replica's bf16 buffers of the split
    groups and its row of the step's barrier words (one int64 per replica; ``torch.distributed._symmetric_memory``
    allocations), their rendezvous handles, the views of the other replicas' buffers and rows, the step's epoch (on
    the device) and the token the replicas drew together when they made their holders.  torch aborts the process when
    a symmetric allocation is freed before its rendezvous handle (``CUDA driver error: invalid argument`` in the
    handle's destruction), so this replica's allocations are held here until the views and the handles have gone
    (:meth:`close`, when the holder goes)."""

    def __init__(self, pg, rank: int, world: int, flag: torch.Tensor, token: str):
        """``pg``: the replicas' process group (the holder serves that group object only), this replica of rank
        ``rank`` among ``world``; ``flag``: this replica's row of barrier words, a zeroed symmetric allocation of
        ``world`` int64; ``token``: drawn by the replicas together (a collective: every replica creates its holder
        with the others)."""
        import torch.distributed._symmetric_memory as symm_mem
        self.pg, self.group, self.rank, self.world, self.token = pg, pg.group_name, rank, world, token
        self.peers = [j for j in range(world) if j != rank]
        self.local: Dict[Optional[int], torch.Tensor] = {}             # None: the barrier row; gi: group gi's buffer
        self.handles: Dict[Optional[int], object] = {}
        self.remote: Dict[int, Dict[int, torch.Tensor]] = {}           # gi: {replica j: its buffer of group gi}
        self.flag = flag
        self.local[None] = flag
        self.handles[None] = symm_mem.rendezvous(flag, self.group)
        self.flags: List[torch.Tensor] = [flag if j == rank else self.handles[None].get_buffer(j, (world,), torch.int64)
                                          for j in range(world)]       # every replica's row, this one's at ``rank``
        self.epoch = torch.zeros(1, dtype=torch.int64, device=flag.device)

    def add(self, gi: int, h: torch.Tensor) -> None:
        """Group gi's bf16 buffer ``h`` (a symmetric allocation of this replica): its handle and every other replica's
        buffer mapped here (a collective: every replica adds the same groups in the same order)."""
        import torch.distributed._symmetric_memory as symm_mem
        self.local[gi] = h
        self.handles[gi] = symm_mem.rendezvous(h, self.group)
        self.remote[gi] = {j: self.handles[gi].get_buffer(j, tuple(h.shape), h.dtype) for j in self.peers}

    def close(self) -> None:
        """The views of the other replicas' memory, then the handles, then this replica's allocations (which live on
        as long as anything else holds them)."""
        self.remote, self.flags = {}, []
        self.handles = {}
        self.local, self.flag = {}, None

    def __del__(self):
        self.close()


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
        self._symm: Optional[_PeerBuffers] = None                        # two replicas: the symmetric buffers and the
        #                                                                  step's barrier (:meth:`_push_setup`)
        self.stale = False                                               # split: the parts other replicas own have
        #                                                                  been updated since the last gather_
        keys = ("p", "g", "m", "v") + (("e",) if self.ema else ())
        for gi, g in enumerate(groups):
            params = g["params"]
            linear = bool(g.get("linear"))
            bound = bool(g.get("bound"))
            if linear or bound:
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
            first = int(g.get("written_first", 0))
            if first and (not linear or first > len(params) or any(p.ndim != 2 for p in params[:first])):
                raise ValueError(f"FlatAdamW: group {g.get('name', gi)}: written_first {first} must count 2-D leading "
                                 "parameters of a linear group")
            keep = 0 if not first else (offs[first] if first < len(params) else n)
            bf16 = []                                                # (parameter, its (padded N, K) part of h)
            if linear:
                for p, off in zip(params, offs):
                    if p.ndim == 2:
                        bind_grad_(p)
                        rows = padded_extent(p.shape[0])
                        bf16.append((p, flat["h"][off:off + rows * p.shape[1]].view(rows, p.shape[1])))
            elif bound:
                for p in params:
                    if p.ndim == 2:
                        bind_grad_(p)
            self.groups.append({"name": g.get("name", str(gi)), "flat": flat, "n": n,
                                "lr_mult": float(g.get("lr_mult", 1.0)), "wd": float(g.get("weight_decay", 0.0)),
                                "params": params, "linear": linear, "bf16": bf16, "bf16_versions": [],
                                "keep": keep, "written_first": list(params[:first])})
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
        # the gradient norms' pass over every parameter's slice of its group's gradient buffer; its group, its bucket
        self._norm = _norm_pass([(gi, off, off + k, s) for s, (gi, off, k, _) in enumerate(self.slots.values())],
                                len(self.slots), dev)
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
        fp32: one pass over the parameters' slices of the gradient buffers (:func:`_norm_pass`: per group a program
        per chunk of at most NORM_CHUNK elements of one parameter, :func:`_sumsq_kernel`; then each parameter's chunk
        sums added, :func:`_slot_sums_kernel`; every sum in a fixed order), the groups' and the buckets' norms the
        square roots of the sums of the parameters' sums of squares.  Sharded: the pass is over this replica's pieces
        of the parameters (:meth:`shard`), and every parameter's sum of squares is the sum of the replicas' in rank
        order (an all-gather: the same values on every replica)."""
        sh = self.sharding
        plan = self._norm if sh is None else sh["norm"]
        for gi, starts, ends, base, n in plan["groups"]:
            _sumsq_kernel[(n,)](self.groups[gi]["flat"]["g"], starts, ends, plan["part"][base:], CHUNK=NORM_CHUNK,
                                BLOCK=NORM_BLOCK, num_warps=8)
        sq = plan["sq"]
        if sq.numel():
            _slot_sums_kernel[(sq.numel(),)](plan["part"], plan["first"], plan["last"], sq, BLOCK=1024, num_warps=4)
        if sh is not None:
            every = torch.empty(sh["world"] * sq.numel(), dtype=torch.float32, device=sq.device)
            dist.all_gather_into_tensor(every, sq, group=sh["group"])
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

    def step(self, coef: torch.Tensor, keep: bool = False) -> None:
        """One update with gradients scaled by the device scalar ``coef``; zeroes the gradients (with ``keep``: but the
        gradients of the parameters :meth:`written_first` lists, which the caller has the next backward write); moves
        the average by ``ema_c``.  Sharded: this replica's part of every split range in turn, each range's bf16 values
        brought to the other replicas as soon as its part is updated (on a side stream, while the next part and then
        the replicated ranges are updated; the step waits for the side stream): replicas that map each other's memory
        copy their part into every other replica's buffer between two barriers of all of them (before the first copy:
        every replica has finished everything it queued before the step, its backward among it; at the end: every copy
        has landed), others gather each range from its owners (module docstring)."""
        self.t.add_(1.0)
        self.bc[0].copy_(1.0 - torch.pow(self.b1, self.t))
        self.bc[1].copy_(torch.sqrt(1.0 - torch.pow(self.b2, self.t)))
        sh = self.sharding
        if sh is None:
            for g in self.groups:
                self._update(g, coef, keep=keep)
            return
        self.stale = True                                            # (a replay: the engine marks it)
        cur, side = torch.cuda.current_stream(), sh["stream"]
        push = sh["push"]
        if push is not None:
            from neuronspark.ops.triton.tp_bulk import tp_fence
        for i, ((gi, starts, ends), rng) in enumerate(zip(sh["owned"], sh["buckets"])):
            self._update(self.groups[gi], coef, starts, ends, keep=keep)
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                if push is None:
                    self._gather_range(*rng)
                else:                                                # (this replica's part into every other's buffer)
                    if i == 0:
                        push.epoch.add_(1)
                        tp_fence(push.flags, push.rank, push.epoch)
                    _, _, _, olo, ohi = rng
                    part = self.groups[gi]["flat"]["h"][olo:ohi]
                    for j in push.peers:
                        push.remote[gi][j][olo:ohi].copy_(part)
        for gi, starts, ends in sh["replicated"]:
            self._update(self.groups[gi], coef, starts, ends, keep=keep)
        cur.wait_stream(side)
        if push is not None:
            push.epoch.add_(1)
            tp_fence(push.flags, push.rank, push.epoch)

    def _update(self, g: dict, coef: torch.Tensor, starts: Optional[torch.Tensor] = None,
                ends: Optional[torch.Tensor] = None, keep: bool = False) -> None:
        """The update kernel over the whole group, or over the blocks ``starts`` / ``ends`` (:func:`_blocks`); with
        ``keep`` the gradient below the group's ``keep`` element is left as it is."""
        f = g["flat"]
        ragged = starts is not None
        grid = (int(starts.numel()) if ragged else triton.cdiv(g["n"], self.BLOCK),)
        _adamw_kernel[grid](f["p"], f["g"], f["m"], f["v"], f["e"] if self.ema else f["p"],
                            f["h"] if g["linear"] else f["p"], g["n"], g["keep"] if keep else 0,
                            starts if ragged else f["p"],
                            ends if ragged else f["p"], self.lr, self.wd, self.ema_c, coef, self.bc, g["lr_mult"],
                            g["wd"], self.b1, self.b2, self.eps, EMA=self.ema, BF16=g["linear"], RAGGED=ragged,
                            BLOCK=self.BLOCK, num_warps=8)

    def zero_grad(self) -> None:
        for g in self.groups:
            g["flat"]["g"].zero_()

    def written_first(self) -> List[torch.Tensor]:
        """The parameters whose gradients a step with ``keep`` leaves (the leading ``written_first`` parameters of
        their groups): the caller arms them (``neuronspark.ops.linear.arm_grad_overwrite_``) before the backward of
        the next update's first micro-batch."""
        return [p for g in self.groups for p in g["written_first"]]

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

    def bf16_buffers(self) -> Tuple[int, ...]:
        """The addresses of the linear groups' bf16 buffers (the first split among two replicas moves them into
        symmetric memory, :meth:`shard`: nstrain.engine refuses graphs captured on the old ones)."""
        return tuple(g["flat"]["h"].data_ptr() for g in self.groups if g["linear"])

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
        # every process decides together (a collective): a process whose optimizer was built again would otherwise
        # return here while the others enter the collectives below, or enter them alone
        every: List[Optional[tuple]] = [None] * world
        dist.all_gather_object(every, (layout, self.sharding["spec"][2:] if self.sharding is not None else None),
                               group=group)
        if any(e != every[0] for e in every):
            raise RuntimeError("FlatAdamW.shard: the processes ask for different layouts or hold different ones (an "
                               "optimizer built again or sharded on some processes only); shard the replicas' "
                               "optimizers together, from the same state")
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
        owned = [(gi,) + _blocks([(olo, ohi)], self.BLOCK, dev) for gi, _, _, olo, ohi in buckets]   # (per range)
        replicated = []
        for gi, g in enumerate(self.groups):
            cut = [(lo, hi) for bgi, lo, hi, _, _ in buckets if bgi == gi]
            if not cut:
                replicated.append((gi, None, None))
                continue
            rest, at = [], 0
            for lo, hi in cut:
                if lo > at:
                    rest.append((at, lo))
                at = hi
            if at < g["n"]:
                rest.append((at, g["n"]))
            if rest:
                replicated.append((gi,) + _blocks(rest, self.BLOCK, dev))
        # the gradient norms' pieces: a parameter in a split range, its intersection with this replica's part; any
        # other parameter (its summed gradient on every replica), its rank-th of world consecutive pieces, cut at
        # multiples of 16 elements (the pass's alignment)
        lows = {gi: [b[1] for b in buckets if b[0] == gi] for gi in range(len(self.groups))}
        rows = {gi: [b for b in buckets if b[0] == gi] for gi in range(len(self.groups))}
        pieces = []
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
                cut = [off + k * r // world // SHARD_ALIGN * SHARD_ALIGN for r in range(world)] + [off + k]
                a, z = cut[rank], cut[rank + 1]
            if z > a:
                pieces.append((gi, a, z, i))
        push = self._push_setup(group, rank, sorted({gi for gi, _ in layout}))
        self.sharding = {"spec": spec, "group": group, "rank": rank, "world": world, "buckets": buckets,
                         "owned": owned, "replicated": replicated, "norm": _norm_pass(pieces, len(self.slots), dev),
                         "push": push,
                         "stream": torch.cuda.Stream(device=dev, priority=torch.cuda.Stream.priority_range()[1])}
        #                (the highest: the gathers dispatched ahead of the update's pending blocks, nstrain.engine.Graph)

    def _push_setup(self, group, rank: int, gis: List[int]) -> Optional[_PeerBuffers]:
        """Replicas that are the whole job (2 to ``MAX_GPUS`` of them), on GPUs of one host that map each other's
        memory and that every process numbers alike (all agree, collectives): the bf16 buffer of every linear group in
        ``gis`` moved into symmetric memory, its values taken from the fp32 weights (whole here: :meth:`shard` has
        gathered them; as :meth:`refresh_bf16`) and the weights' bf16 parts bound to the new buffer, with every other
        replica's buffer mapped here, and the words of the step's barriers (a row of one int64 per replica in
        symmetric memory, the epoch on the device), all held by a :class:`_PeerBuffers` kept for the optimizer's life
        (a later layout of the same process group object reuses it; when the processes' holders differ, all start
        anew).  Every new allocation is made before any is shared, all replicas agree that all were made, and a new
        holder's views of the other processes' rows are checked to be those processes' rows.  None otherwise: the step
        gathers with NCCL (module docstring)."""
        import warnings

        import torch.distributed._symmetric_memory as symm_mem
        from neuronspark.ops.linear import bind_train_bf16_
        from neuronspark.ops.triton.tp_exchange import MAX_GPUS
        dev = self.t.device
        pg = group if group is not None else dist.group.WORLD
        world = dist.get_world_size(pg)
        with torch.cuda.device(dev):                                 # (the collectives on this replica's device)
            # torch's rendezvous of symmetric memory counts the rendezvous of each process (not of each group) and
            # refuses two buffers on GPUs of the same host with the same index: the job is these processes, and all
            # see the same GPUs in the same order
            seen = [str(torch.cuda.get_device_properties(i).uuid) for i in range(torch.cuda.device_count())]
            s = self._symm if self._symm is not None and self._symm.pg is pg else None
            views: List[Optional[tuple]] = [None] * world
            dist.all_gather_object(views, (seen, dev.index, (s.token, sorted(s.remote)) if s is not None else None),
                                   group=pg)
            others = [views[j][1] for j in range(world) if j != rank]
            ok = (dist.get_world_size() == world and 2 <= world <= MAX_GPUS and all(v[0] == seen for v in views)
                  and len({v[1] for v in views}) == world
                  and all(torch.cuda.can_device_access_peer(dev.index, o) for o in others))
            if not self._agree(ok, pg, dev):
                return None
            if any(v[2] != views[0][2] for v in views):              # (the holders differ: made apart, or an
                s = None                                             # optimizer built again on one side: all start
            new = [gi for gi in gis if s is None or gi not in s.remote]   # anew, all rendezvous the same allocations)
            fresh: Dict[Optional[int], torch.Tensor] = {}
            failed = None
            try:
                if s is None:
                    fresh[None] = symm_mem.empty(world, dtype=torch.int64, device=dev)
                for gi in new:
                    fresh[gi] = symm_mem.empty(self.groups[gi]["n"], dtype=torch.bfloat16, device=dev)
            except (torch.OutOfMemoryError, RuntimeError) as e:
                fresh, failed = {}, e
            if not self._agree(failed is None, pg, dev):             # (nothing shared yet: dropped as it is)
                warnings.warn("nstrain.flat_adamw: no symmetric memory for the split groups' bf16 buffers "
                              f"({failed or 'on the other process'}); the replicas gather them with NCCL")
                return None
            with torch.no_grad():
                if s is None:
                    fresh[None].zero_()
                    tokens: List[Optional[str]] = [None] * world
                    dist.all_gather_object(tokens, uuid.uuid4().hex, group=pg)
                    s = _PeerBuffers(pg, rank, world, fresh[None], tokens[0])
                    if not self._peer_check(s, rank, pg, dev):
                        self._symm = None
                        warnings.warn("nstrain.flat_adamw: the symmetric memory of this process group does not map "
                                      "the other processes' memory (a process group created again under a name torch "
                                      "had given an earlier one?); the replicas gather with NCCL")
                        return None
                    self._symm = s
                for gi in new:
                    g = self.groups[gi]
                    old, h = g["flat"]["h"], fresh[gi]
                    h.copy_(g["flat"]["p"])
                    s.add(gi, h)
                    g["flat"]["h"] = h
                    at = [part.storage_offset() - old.storage_offset() for _, part in g["bf16"]]
                    g["bf16"] = [(p, h[a:a + part.numel()].view(part.shape)) for (p, part), a in zip(g["bf16"], at)]
                    for p, part in g["bf16"]:
                        bind_train_bf16_(p, part)
                    g["bf16_versions"] = [p._version for p, _ in g["bf16"]]
            torch.cuda.synchronize(dev)
            dist.barrier(group=pg, device_ids=[dev.index])
        return s

    @staticmethod
    def _agree(ok: bool, pg, dev: torch.device) -> bool:
        """Whether every process of ``pg`` has ``ok`` (a collective)."""
        t = torch.tensor([int(bool(ok))], device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MIN, group=pg)
        return bool(t.item())

    @staticmethod
    def _peer_check(s: _PeerBuffers, rank: int, pg, dev: torch.device) -> bool:
        """Whether the holder's views of the other processes' rows are those processes' rows (all agree, a
        collective): each process writes a mark of its own into its row and reads the others' through the views, then
        clears its row.  torch keeps a process group's ranks and store under the group's name, which a group created
        again in the same processes can reuse."""
        s.flag.fill_(1000 + rank)
        torch.cuda.synchronize(dev)
        dist.barrier(group=pg, device_ids=[dev.index])
        ok = all(bool((s.flags[j] == 1000 + j).all()) for j in s.peers)
        dist.barrier(group=pg, device_ids=[dev.index])
        s.flag.zero_()
        torch.cuda.synchronize(dev)
        return FlatAdamW._agree(ok, pg, dev)

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
        for rng in sh["buckets"]:
            self._gather_range(*rng)

    def _gather_range(self, gi: int, lo: int, hi: int, olo: int, ohi: int) -> None:
        """The bf16 values of the split range [lo, hi) of group gi (this replica's part [olo, ohi)) from the owners of
        its parts (in place, on the current stream)."""
        h = self.groups[gi]["flat"]["h"]
        dist.all_gather_into_tensor(h[lo:hi], h[olo:ohi], group=self.sharding["group"])
