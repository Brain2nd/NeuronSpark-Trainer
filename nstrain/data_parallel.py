"""Data-parallel replicas of the training step: one process per GPU, each with its own rows of the global batch.

Every process holds the whole model and the optimizer's buffers (:class:`nstrain.flat_adamw.FlatAdamW`).  The step
engine (:class:`nstrain.engine.StepEngine` with ``data_parallel``) scales each micro-batch's loss by
``1 / (accum * world)`` and, once per update, sums the flat gradient buffers across the processes (fp32, NCCL, inside
the captured CUDA graph of the update's last micro-batch), so the update's micro-batches end with the gradient of the
mean loss over the global batch: the same gradient the one-GPU step computes over the same rows.

The weights the training forward reads only through their bf16 values (the projection weights of the model's layers:
those the module holding the layers lists with ``inference_linear_weights``, the weights of ``neuronspark.ops.linear``;
not the tied embedding, whose lookup reads its fp32 rows, nor the weights other operators read in fp32 -- the
dendritic convolution taps, the PKM values and sub-keys, the attention's short convolution) are updated in parts
(ZeRO stage 1): their buckets are summed by a reduce-scatter, which leaves replica r the sum of
the r-th part of every bucket (the rest of the bucket, this replica's own gradient, is zeroed for the next update, or
left for the next update's first backward to write when the engine arms the weights, ``overwrite``:
nstrain.engine);
the optimizer updates its parts and gathers the bf16 values of the buckets from their owners after the update
(:meth:`nstrain.flat_adamw.FlatAdamW.shard`).  Every other gradient is summed by an all-reduce and updated on every
replica.  The collectives hand every process the same values, and everything computed from them (the norms from the
replicas' pieces added in rank order, the clip, the update of the replicated ranges) is computed the same way on every
GPU: the bf16 values and the replicated ranges stay bit-identical, and the fp32 weights and moments of the split
ranges are each held by one replica (:meth:`nstrain.flat_adamw.FlatAdamW.gather_` makes them whole on every one).

The reduction is hidden behind the backward of the update's last micro-batch.  The flat gradient buffers of the large
groups are cut into buckets aligned to the model's layers (the slots of a layer's parameters in a group's buffer, the
projection weights and the others in buckets of their own; consecutive layers are merged until a bucket holds at least
``BUCKET_MIN_NUMEL`` elements, so the projections of consecutive layers merge when they are contiguous, as
nstrain.model.param_groups places them); a bucket is summed (on a
side stream, inside the captured graph) as soon as the backward has written every gradient in it.  That moment is
known without counting the writers: autograd runs the ready node of the highest sequence number first, and a node's
gradient comes only from nodes created after it, so when the node that produced a layer's input runs, every node
created after it -- every operation of that layer and of the layers above, the readout and the losses -- has run and
issued its kernels (the weight gradients of the linear operator written by its own GEMM, the others by autograd's
accumulation, which runs as soon as it is ready).  A hook on the input of every layer (attached during the forward,
not during a recomputation in the backward) issues the buckets of that layer.  The rest of the buffers -- the groups
too small to cut, the parts of a buffer outside the layers, and the layers' vectors (a joined dendritic pre-norm gain
is written by the layer below) -- is summed by parts: a part none of whose weights the linear operator writes itself
(their gradients all autograd's accumulation: the lexical tables, the vectors) as soon as autograd has accumulated
every gradient in it (a hook after the accumulation on each of its parameters; neuronspark makes the lexical tables'
lookup just before the first hippocampal layer, so their part is summed behind the backward of the layers below it),
and the others (the tail: the readout balance's projections) with the loss sums after the backward returns.  Every
collective is issued on the one side stream, in the same order on every process; the capture stream waits for it
before the norms.

A lookup table held by the optimizer (an ``nn.Embedding`` set to ``sparse``: the tied embedding) gets its gradient
from two writers: the readout's GEMM, at the start of the backward, and the lookup, at its very end.  The lookup's
rows are taken from the backward (a hook on the weight returns an empty gradient in their place, so nothing is added
to the buffer), the table's slice is summed as soon as the node producing the readout's input (the output of the
module holding the layers) runs, and after the backward the rows of every process are gathered and added into the
summed slice in rank order (``index_put_`` with accumulation, deterministic): every process adds the same rows the
same way.

The process group is NCCL over the GPUs of one host (``torchrun``'s environment: ``RANK``, ``WORLD_SIZE``,
``LOCAL_RANK``); it is created eagerly (bound to this process's GPU), so the communicator exists before any graph is
captured.

The split buckets' reduce-scatter by the copy engines.  When the optimizer's bf16 values travel by copies between
replicas that map each other's memory (:mod:`nstrain.flat_adamw`, "How the bf16 values travel"), a split bucket is
summed without NCCL as well: every replica copies each other replica's part of the bucket into that replica's staging
memory (``Tensor.copy_`` into the mapped view: a copy engine's copy, which takes no SM from the backward beside it;
replica s writes slot (s - o - 1) mod N of replica o's N - 1 slots of the bucket), then passes a barrier of all the
replicas on the bucket's row of flag words (``neuronspark.ops.triton.tp_bulk.tp_fence``: it stores the update's epoch
into its word of every other replica's row, then waits until every other word of its own row holds it, so every other
replica's copies into this one have landed), and one kernel adds the N values of its part in rank order into its part
of the bucket.  The epoch is a device counter advanced once per update before the first split bucket (inside the
captured graph as well), equal on every replica, and the flag words are compared with >=, so they are never cleared.
A bucket's slots are written again only in the next update, which every replica starts after the tail's all-reduce of
this one, which no replica passes before every replica has issued it on the side stream, after its sums of this
update's buckets.  As with NCCL's collectives, every replica issues the same buckets in the same order, and an update
that fails on one replica after its first copy leaves the others waiting: the replicas are out of step, the update is
not retried (:meth:`DataParallel.begin` refuses) and the job ends.  Two replicas add the same two values as NCCL;
more add in rank order, the same on every run, which NCCL's ring need not.  The staging is made by every replica
together (the same parts, a token drawn in common, every allocation made before any is shared), kept while the parts
stay the same, and a staging no longer used is freed once every graph that may address it is gone, on every replica
alike.  Other layouts sum split buckets with NCCL's reduce-scatter.
"""
from __future__ import annotations

import os
import re
import uuid
import weakref
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from nstrain.engine import capturing_graph, release_graphs

BUCKET_MIN_NUMEL = 1 << 24          # 16 M fp32 elements (64 MB) per bucket at least, consecutive layers merged
GROUP_MIN_NUMEL = 1 << 24           # a group smaller than this is summed whole in the tail
HEAD = 1 << 30                      # the bucket of the lookup tables' slices (issued at the readout's input)
CROSS_LAYER = (".pre_norm_gain",)   # a layer's parameters the layer below writes (a joined dendritic pre-norm): tail
SUM_BLOCK = 4096                    # the rank-order sum of a split bucket's part: elements per program
SUM_WARPS = 8


@triton.jit
def _rank_sum_kernel(OWN, STG, c, RANK: tl.constexpr, NR: tl.constexpr, BLOCK: tl.constexpr):
    """OWN[i] = the NR replicas' values of element i of this replica's part (c fp32), added in rank order: this
    replica's own in OWN, replica s's in slot (s - RANK - 1) mod NR of the staging STG (slots of c elements)."""
    o = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = o < c
    own = tl.load(OWN + o, mask=m)
    if RANK == 0:
        acc = own
    else:
        acc = tl.load(STG + tl.full([], (NR - RANK - 1) % NR, tl.int64) * c + o, mask=m)
    for s in tl.static_range(1, NR):
        if s == RANK:
            acc = acc + own
        else:
            acc = acc + tl.load(STG + tl.full([], (s - RANK - 1 + NR) % NR, tl.int64) * c + o, mask=m)
    tl.store(OWN + o, acc, mask=m)


class _ScatterBuffers:
    """The symmetric memory of the split buckets' reduce-scatter by the copy engines (module docstring): this replica's
    staging (N - 1 slots of each split bucket's part, in issue order) and flag rows (N int64 words per split bucket),
    their rendezvous handles, the views of every other replica's staging and rows, the epoch (on the device) and the
    token the replicas drew together when they shared it.  torch aborts the process when a symmetric allocation is
    freed before its rendezvous handle, so the allocations are held here until the views and handles have gone
    (:meth:`close`)."""

    def __init__(self, dev: torch.device, rank: int, world: int, parts: List[int]):
        """This replica's allocations on ``dev`` for ``parts``, the elements of each split bucket's part (its numel /
        world) in issue order; nothing is shared yet (:meth:`share`)."""
        import torch.distributed._symmetric_memory as symm_mem
        self.rank, self.world = rank, world
        self.peers = [j for j in range(world) if j != rank]
        self.offsets, at = [], 0                                     # each bucket's first staging element
        for c in parts:
            self.offsets.append(at)
            at += (world - 1) * c
        self.parts = list(parts)
        self.stage = symm_mem.empty(max(at, 1), dtype=torch.float32, device=dev)
        self.flag = symm_mem.empty(len(parts) * world, dtype=torch.int64, device=dev)
        self.epoch = torch.zeros(1, dtype=torch.int64, device=dev)
        self.handles: list = []
        self.remote: Dict[int, torch.Tensor] = {}
        self.flags: List[torch.Tensor] = []
        self.token: Optional[str] = None
        self.graphs: weakref.WeakSet = weakref.WeakSet()           # the nstrain graphs captured with it
        self.pinned = False                                          # captured outside nstrain's graphs: kept

    def share(self, pg, token: str) -> None:
        """Map every other replica's staging and rows here (a collective: every replica shares the holder it made
        for the same parts, with the same ``token``).  This replica's rows are cleared and the device synchronized
        first, so no other replica can store into a row before it is clear."""
        import torch.distributed._symmetric_memory as symm_mem
        self.flag.zero_()
        torch.cuda.synchronize(self.flag.device)
        self.token = token
        self.handles = [symm_mem.rendezvous(self.stage, pg.group_name), symm_mem.rendezvous(self.flag, pg.group_name)]
        self.remote = {j: self.handles[0].get_buffer(j, tuple(self.stage.shape), torch.float32) for j in self.peers}
        self.flags = [self.flag if j == self.rank else
                      self.handles[1].get_buffer(j, tuple(self.flag.shape), torch.int64) for j in range(self.world)]

    def rows(self, b: int) -> List[torch.Tensor]:
        """Bucket b's row of flag words of every replica (this one's at ``rank``)."""
        return [f[b * self.world:(b + 1) * self.world] for f in self.flags]

    def slot(self, j: int, b: int) -> torch.Tensor:
        """The slot of bucket b that this replica writes in replica j's staging (mapped here)."""
        c = self.parts[b]
        at = self.offsets[b] + ((self.rank - j - 1) % self.world) * c
        return self.remote[j][at:at + c]

    def close(self) -> None:
        self.remote, self.flags = {}, []
        self.handles = []
        self.stage = self.flag = None

    def __del__(self):
        self.close()


class DataParallel:
    def __init__(self, group=None):
        self.group = group
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.side = None                                             # the reduction's stream (attach)
        self.layer_buckets: Dict[int, List[Tuple[torch.Tensor, bool, bool]]] = {}   # layer -> its input hook's
        #                  buckets (a bucket: its view, whether it is split, whether the rest is zeroed after its sum)
        self.tail: List[torch.Tensor] = []
        self.early: List[Tuple[torch.Tensor, Tuple[str, ...]]] = []  # the parts summed once autograd has accumulated
        #                                                              every gradient in them (view, parameter names)
        self.head: List[torch.Tensor] = []                           # the lookup tables' slices (the readout's input)
        self.lookups: Dict[str, torch.Tensor] = {}                   # weight name -> its (rows, cols) gradient view
        self._rows: Dict[str, torch.Tensor] = {}                     # weight name -> the lookup's sparse gradient
        self.bucket_numels: List[int] = []                           # in issue order (the tail last)
        self.split_numel = 0                                         # elements of the split buckets
        self.order: List[str] = []                                   # what the last armed backward issued, in order
        self._stream = None
        self._watch = False                                          # attach the input hooks (the forward)
        self._armed = False                                          # the hooks issue their buckets
        self._issued: set = set()
        self._left: List[int] = []                                   # per early part: gradients still to accumulate
        self._seen: set = set()                                      # the early parts' parameters accumulated
        self._handles: list = []                                     # the module hooks (removed by detach)
        self.fused = None                                            # the engine's fused accumulation (attach)
        self.scatter: Optional[_ScatterBuffers] = None               # the copy engines' reduce-scatter (attach)
        self._retired: List[_ScatterBuffers] = []                    # holders no longer used (_prune_retired)
        self._bucket: Dict[int, int] = {}                            # a split view's data_ptr -> its bucket index
        self._epoch_due = False                                      # the update's epoch not yet advanced
        self._broken = False                                         # a reduction failed after its first copy

    @classmethod
    def from_env(cls) -> "DataParallel":
        """The process group of ``torchrun``'s environment, this process on GPU ``LOCAL_RANK``."""
        local = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local)
        if not dist.is_initialized():
            dist.init_process_group("nccl", device_id=torch.device("cuda", local))
        return cls()

    def close(self) -> None:
        """Destroy the process group, every captured graph still alive released first (nstrain.engine.release_graphs:
        NCCL's teardown waits for the graphs that hold its kernels) and the reduce-scatter's symmetric memory let go
        (its handles' teardown needs the group)."""
        release_graphs()
        self._retire_scatter()
        torch.cuda.synchronize()
        for s in self._retired:
            s.close()
        self._retired = []
        if dist.is_initialized():
            dist.destroy_process_group()

    def _retire_scatter(self) -> None:
        """The reduce-scatter's holder no longer used, kept while a graph captured with it may still be launched
        (:meth:`_prune_retired`) or until :meth:`close`."""
        if self.scatter is not None:
            self._retired.append(self.scatter)
            self.scatter = None
        self._bucket = {}

    def _prune_retired(self) -> None:
        """Free the retired holders that no graph may address any more (every nstrain graph captured with it released
        or collected; none captured otherwise) on every replica (a collective: a holder is freed only where every
        replica can free it)."""
        free = [s.token for s in self._retired if not s.pinned and not any(g.alive() for g in s.graphs)]
        every: List[Optional[list]] = [None] * self.world
        dist.all_gather_object(every, free, group=self.group)
        gone = set(free).intersection(*(set(e) for e in every))
        if not gone:
            return
        torch.cuda.synchronize(self.device)                          # (launches in flight may still use them)
        for s in self._retired:
            if s.token in gone:
                s.close()
        self._retired = [s for s in self._retired if s.token not in gone]

    def _set_scatter(self, parts: Optional[List[int]]) -> None:
        """The copy engines' reduce-scatter for split buckets of ``parts`` elements per replica (None: NCCL's), decided
        by every replica together (a collective): the present holder kept when every replica holds the same one for
        these parts, else a new one that every replica makes (every allocation made, then agreed, before any is
        shared) and the present one retired."""
        cur = self.scatter
        mine = (cur.token, cur.parts) if cur is not None else None
        every: List[Optional[tuple]] = [None] * self.world
        dist.all_gather_object(every, (parts, mine), group=self.group)
        if parts is not None and mine is not None and cur.parts == parts and all(e == every[0] for e in every):
            return
        self._retire_scatter()
        if parts is None or any(e[0] != parts for e in every):
            return
        failed = None
        try:
            with torch.cuda.device(self.device):
                s = _ScatterBuffers(self.device, self.rank, self.world, parts)
        except (torch.OutOfMemoryError, RuntimeError) as e:
            s, failed = None, e
        ok = torch.tensor([int(failed is None)], device=self.device)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=self.group)
        if not bool(ok.item()):                                      # (nothing shared yet: dropped as it is)
            import warnings
            warnings.warn("nstrain.data_parallel: no symmetric memory for the split buckets' reduce-scatter "
                          f"({failed or 'on another process'}); the replicas sum them with NCCL")
            return
        tokens: List[Optional[str]] = [None] * self.world
        dist.all_gather_object(tokens, uuid.uuid4().hex, group=self.group)
        with torch.cuda.device(self.device):
            s.share(self.group if self.group is not None else dist.group.WORLD, tokens[0])
        self.scatter = s

    # ------------------------------------------------------------------ collectives
    def barrier(self) -> None:
        dist.barrier(group=self.group, device_ids=[self.device.index])

    def all_reduce_(self, t: torch.Tensor) -> None:
        """Sum ``t`` over the replicas, in place (captured when called inside a graph capture)."""
        dist.all_reduce(t, group=self.group)

    # ------------------------------------------------------------------ the reduction hidden behind the backward
    def attach(self, model, opt, stream, overwrite: bool = False, fused=None) -> None:
        """Cut the flat gradient buffers of ``opt`` into the layers' buckets and the tail, split the linear group's
        layer buckets among the replicas (``opt``'s layout, :meth:`nstrain.flat_adamw.FlatAdamW.shard`), and hook the
        input of every layer of ``model`` (its modules named ``...layers.<i>``); ``stream``: the stream the backward
        runs on; ``overwrite``: the caller arms the optimizer's ``written_first`` weights in every update's first
        backward (their gradients written, not added), so a split bucket of theirs leaves the rest of the bucket as it
        is after its reduce-scatter; ``fused``: the caller's :class:`nstrain.engine.FusedAccumulation`, which keeps
        autograd's gradients and adds them later: its kept gradients are added before every collective (a kept
        gradient's post-accumulate hook runs when it is kept, before it is added)."""
        self._stream = stream
        self.fused = fused
        self.side = torch.cuda.Stream(priority=torch.cuda.Stream.priority_range()[1])   # (the highest: its
        #                     collectives dispatched ahead of the backward's pending blocks, nstrain.engine.Graph)
        layers, prefix = {}, None                                    # the outermost ``<prefix>layers.<i>`` modules
        for name, mod in model.named_modules():
            m = re.fullmatch(r"(.*?)layers\.(\d+)", name)
            if m and (prefix is None or m.group(1) == prefix):
                prefix = m.group(1)
                layers[int(m.group(2))] = mod
        if not layers:
            raise RuntimeError("DataParallel: the model has no modules named ...layers.<i>")
        of_layer = re.compile(re.escape(prefix) + r"layers\.(\d+)\.")
        trunk = model.get_submodule(prefix[:-1]) if prefix else model   # the module holding the layers
        names = {id(q): n for n, q in model.named_parameters()}
        tables = {names[id(m.weight)]: m.weight for m in model.modules()
                  if isinstance(m, torch.nn.Embedding) and m.sparse and names.get(id(m.weight)) in opt.slots}
        # the weights the training forward reads only through their bf16 values: the projection weights the module
        # holding the layers lists (the weights of neuronspark.ops.linear); none listed, nothing is split
        listed = trunk.inference_linear_weights() if hasattr(trunk, "inference_linear_weights") else []
        projections = {names[id(q)] for _, q in listed if id(q) in names}
        # the weights whose gradients the linear operator adds into their buffers itself (bound by the optimizer):
        # autograd never accumulates them; every other weight's gradient is autograd's accumulation
        bound = {names[id(q)] for g in opt.groups for q, _ in g["bf16"] if id(q) in names}
        buckets: Dict[int, List[Tuple[torch.Tensor, bool, bool]]] = {}   # (view, split, the rest zeroed)
        tail: List[torch.Tensor] = []
        early: List[Tuple[torch.Tensor, Tuple[str, ...]]] = []
        head: List[torch.Tensor] = []
        split: Dict[int, List[Tuple[int, int]]] = {}                 # group -> its split buckets' ranges

        def outside(view, held):                                     # a part outside the layers' buckets (an empty
            if not held or any(name in bound for name in held):      # group: no hook would issue it)
                tail.append(view)
            else:
                early.append((view, tuple(held)))
        for gi, g in enumerate(opt.groups):
            buf, n = g["flat"]["g"], g["n"]
            slots = sorted((off, name) for name, (sgi, off, _, _) in opt.slots.items() if sgi == gi)
            runs: List[Tuple[Tuple[int, bool], int, int, Tuple[str, ...]]] = []   # ((layer or -1, split), lo, hi,
            for i, (off, name) in enumerate(slots):                  # names): consecutive slots
                m = of_layer.match(name)
                layer = int(m.group(1)) if m and not name.endswith(CROSS_LAYER) else -1
                if name in tables:
                    layer = HEAD
                kind = (layer, bool(g["linear"]) and 0 <= layer < HEAD and name in projections)
                hi = slots[i + 1][0] if i + 1 < len(slots) else n
                lo = 0 if i == 0 else off
                if runs and runs[-1][0] == kind:
                    runs[-1] = (kind, runs[-1][1], hi, runs[-1][3] + (name,))
                else:
                    runs.append((kind, lo, hi, (name,)))
            if n < GROUP_MIN_NUMEL and not any(r[0][0] == HEAD for r in runs):
                outside(buf, [name for r in runs for name in r[3]])
                continue
            # merge consecutive layers from the top: a bucket is issued by the hook of its lowest layer; a bucket holds
            # projection weights only (split among the replicas) or none (all-reduced): weights read in fp32 (the
            # dendritic convolution taps, the PKM values and sub-keys, the attention's short convolution) placed between
            # a layer's projections would cut them into several buckets (nstrain.model.param_groups places the
            # projections of all layers first, contiguous)
            keep = g["keep"] if overwrite else 0                     # (below: written by the first backward)

            def close(cur, buf=buf, gi=gi, keep=keep):
                buckets.setdefault(cur[0], []).append((buf[cur[1]:cur[2]], cur[3], cur[3] and cur[2] > keep))
                if cur[3]:
                    split.setdefault(gi, []).append((cur[1], cur[2]))
            cur = None                                               # [lowest layer, lo, hi, split]
            for (layer, part), lo, hi, held in reversed(runs):
                if layer == HEAD:                                    # (a slot of its own: a table and its padding)
                    if cur is not None:
                        close(cur)
                        cur = None
                    head.append(buf[lo:hi])
                    continue
                if layer < 0:
                    if cur is not None:
                        close(cur)
                        cur = None
                    outside(buf[lo:hi], held)
                    continue
                if cur is not None and cur[3] == part and cur[1] == hi and cur[2] - cur[1] < BUCKET_MIN_NUMEL:
                    cur = [layer, lo, cur[2], part]
                else:
                    if cur is not None:
                        close(cur)
                    cur = [layer, lo, hi, part]
            if cur is not None:
                close(cur)
        covered = sum(v.numel() for vs in buckets.values() for v, _, _ in vs) + \
            sum(v.numel() for v in tail + head + [v for v, _ in early])
        if covered != sum(g["n"] for g in opt.groups):
            raise RuntimeError("DataParallel: the buckets do not cover the gradient buffers exactly once")
        self.layer_buckets, self.tail, self.head, self.early = buckets, tail, head, early
        self.bucket_numels = [v.numel() for v in head] + \
            [v.numel() for j in sorted(buckets, reverse=True) for v, _, _ in buckets[j]] + \
            [sum(v.numel() for v in tail)]
        self.split_numel = sum(hi - lo for rs in split.values() for lo, hi in rs)
        self._unhook()
        opt.shard(self.group, split)                                 # (a new layout gathers the present one first)
        # the copy engines' reduce-scatter where the optimizer's bf16 values travel by copies (the same layout of the
        # replicas, agreed by every replica in FlatAdamW._push_setup): the split buckets in issue order; the staging and
        # flags of the same buckets kept (a later engine's graphs address the same memory), others made by every
        # replica together; the holders no graph addresses any more freed
        sh = opt.sharding
        views = [v for j in sorted(buckets, reverse=True) for v, s, _ in buckets[j] if s]
        parts = [v.numel() // self.world for v in views]
        self._prune_retired()
        self._set_scatter(parts if sh is not None and sh.get("push") is not None else None)
        self._bucket = {v.data_ptr(): b for b, v in enumerate(views)} if self.scatter is not None else {}
        for j, mod in layers.items():
            self._handles.append(mod.register_forward_pre_hook(partial(self._forward_pre_hook, j)))
        params = dict(model.named_parameters())
        for i, (_, held) in enumerate(early):
            for name in held:
                self._handles.append(params[name].register_post_accumulate_grad_hook(
                    partial(self._accumulated, i, name)))
        self.lookups = {}
        if head:
            self._handles.append(trunk.register_forward_hook(self._trunk_output))
            for name, w in tables.items():
                gi, off, k, shape = opt.slots[name]
                self.lookups[name] = opt.groups[gi]["flat"]["g"][off:off + k].view(shape)
                self._handles.append(w.register_hook(partial(self._lookup_grad, name)))

    def detach(self) -> None:
        """Remove the hooks :meth:`attach` installed (an engine's release; a new engine attaches again).  The
        optimizer keeps its split layout: a new engine of the same layout continues it, another layout gathers it
        first (:meth:`nstrain.flat_adamw.FlatAdamW.shard`)."""
        self._unhook()
        self.fused = None                                            # (it holds the engine's parameters)

    def _unhook(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def _trunk_output(self, module, args, out):
        if self._watch and torch.is_tensor(out) and out.requires_grad:
            out.register_hook(partial(self._input_done, HEAD))
        return None

    def _lookup_grad(self, name, grad):
        """The lookup's sparse gradient of a table during the last micro-batch: kept, an empty one accumulated.  The
        table's other writer (the readout) adds into the flat buffer itself and returns no gradient for the weight, so
        autograd hands the hook the lookup's rows alone, once per backward."""
        if not self._armed:
            return None
        if not grad.is_sparse:
            raise RuntimeError(f"DataParallel: {name} received a dense gradient through autograd (only the lookup's "
                               "sparse rows are expected there)")
        if name in self._rows:
            raise RuntimeError(f"DataParallel: {name} received two lookup gradients in one backward")
        self._rows[name] = grad
        return torch.sparse_coo_tensor(grad._indices()[:, :0], grad._values()[:0], grad.shape)

    def _forward_pre_hook(self, layer, module, args):
        if self._watch and args and torch.is_tensor(args[0]) and args[0].requires_grad:
            args[0].register_hook(partial(self._input_done, layer))
        return None

    def _input_done(self, layer, grad):
        """The node that produced ``layer``'s input runs: every gradient of the layer (and above) is issued."""
        if self._armed and layer not in self._issued:
            self._issued.add(layer)
            if layer == HEAD:
                self._issue([(v, False, False) for v in self.head], "head")
            else:
                self._issue(self.layer_buckets.get(layer, ()), f"layer {layer}")
        return None

    def _accumulated(self, i, name, param) -> None:
        """Autograd has accumulated ``name``'s gradient into its view of the buffer, or the engine's fused accumulation
        has kept it (added in :meth:`_issue`, before the sum): the early part ``i`` holding it is issued once every
        gradient in it is.  Autograd calls this on the stream it ran the accumulation on (the accumulator's, current
        here), which the side stream waits for as well as for the backward's."""
        if not self._armed:
            return
        if name in self._seen:                                       # (the part would be issued before the second)
            raise RuntimeError(f"DataParallel: {name}'s gradient was accumulated twice in one backward")
        self._seen.add(name)
        self._left[i] -= 1
        if self._left[i] == 0:
            self._issue([(self.early[i][0], False, False)], f"early {i}", after=torch.cuda.current_stream())

    def _issue(self, views, what: str, after=None) -> None:
        """Sum the (view, split, zero the rest) triples ``views`` across the replicas, in order, on the side stream,
        once the backward's stream (and ``after``, when given) has reached this point: a split view by a reduce-scatter
        (this replica's part of the sum stays in its part of the view; the rest of the view, this replica's own
        gradient, is zeroed for the next update unless the next update's first backward writes it; by the copy engines
        where the replicas map each other's memory, module docstring), any other by an all-reduce.  The gradients the
        engine's fused accumulation holds are added first, on the current stream."""
        if not views:
            return
        if self.fused is not None:                                   # (the gradients it kept: added before the sum,
            self.fused.flush()                                       # on the stream current here)
            after = after if after is not None else torch.cuda.current_stream()
        self.order.append(what)
        self.side.wait_stream(self._stream)
        if after is not None and after != self._stream:
            self.side.wait_stream(after)
        with torch.cuda.stream(self.side):
            for v, split, zero_rest in views:
                if not split:
                    dist.all_reduce(v, group=self.group)
                    continue
                c = v.numel() // self.world
                if self.scatter is None:
                    dist.reduce_scatter_tensor(v[self.rank * c:(self.rank + 1) * c], v, group=self.group)
                else:
                    self._scatter(v, c)
                if zero_rest and self.rank > 0:
                    v[:self.rank * c].zero_()
                if zero_rest and self.rank < self.world - 1:
                    v[(self.rank + 1) * c:].zero_()

    def _scatter(self, v: torch.Tensor, c: int) -> None:
        """The reduce-scatter of the split bucket ``v`` by the copy engines (module docstring), on the current stream:
        every other replica's part copied into its slot of the bucket on that replica, the barrier on the bucket's
        row, the N values of this replica's part added in rank order into its part of ``v``."""
        from neuronspark.ops.triton.tp_bulk import tp_fence
        s = self.scatter
        b = self._bucket.get(v.data_ptr())
        if b is None or s.parts[b] != c:
            raise RuntimeError("DataParallel: a split bucket the reduce-scatter's staging was not made for")
        if torch.cuda.is_current_stream_capturing():                 # (the graphs that will address it)
            g = capturing_graph()
            if g is None:
                s.pinned = True
            else:
                s.graphs.add(g)
        try:
            if self._epoch_due:                                      # (once per update, before its first bucket)
                s.epoch.add_(1)
                self._epoch_due = False
            for j in s.peers:
                s.slot(j, b).copy_(v[j * c:(j + 1) * c])
            tp_fence(s.rows(b), self.rank, s.epoch)
            stage = s.stage[s.offsets[b]:s.offsets[b] + (self.world - 1) * c]
            _rank_sum_kernel[(triton.cdiv(c, SUM_BLOCK),)](v[self.rank * c:(self.rank + 1) * c], stage, c,
                                                           RANK=self.rank, NR=self.world, BLOCK=SUM_BLOCK,
                                                           num_warps=SUM_WARPS)
        except BaseException:                                        # (the replicas are out of step: module docstring)
            self._broken = True
            raise

    def begin(self) -> None:
        """Before the forward of an update's last micro-batch: hook the layers' inputs, issue buckets when ready."""
        if self._broken:
            raise RuntimeError("DataParallel: an earlier update failed in the middle of its reduction; the replicas "
                               "are out of step (module docstring): end the job")
        self._epoch_due = True
        self._issued = set()
        self._rows = {}
        self._left = [len(held) for _, held in self.early]
        self._seen = set()
        self.order = []
        self._watch = self._armed = True

    def forward_done(self) -> None:
        """After that forward (a recomputation in the backward attaches no hooks)."""
        self._watch = False

    def finish(self, sums: torch.Tensor) -> None:
        """After that backward: the tail and ``sums`` summed; the backward's stream waits for every collective."""
        self._armed = False
        missing = [j for j in list(self.layer_buckets) + ([HEAD] if self.head else []) if j not in self._issued]
        unaccumulated = sorted(name for _, held in self.early for name in held if name not in self._seen)
        if missing or unaccumulated or set(self._rows) != set(self.lookups):
            raise RuntimeError(f"DataParallel: buckets not issued {sorted(missing)}, gradients autograd did not "
                               f"accumulate {unaccumulated}, lookup gradients {sorted(self._rows)} of "
                               f"{sorted(self.lookups)}")
        gathered = []
        for name in sorted(self.lookups):                            # (allocated on the backward's stream)
            g = self._rows[name]
            idx, val = g._indices()[0].contiguous(), g._values().contiguous()
            gathered.append((name, idx, val, idx.new_empty(self.world * idx.numel()),
                             val.new_empty((self.world * val.shape[0],) + tuple(val.shape[1:]))))
        self._issue([(v, False, False) for v in self.tail + [sums]], "tail")
        if gathered:
            self.side.wait_stream(self._stream)
            with torch.cuda.stream(self.side):
                for _, idx, val, all_idx, all_val in gathered:
                    dist.all_gather_into_tensor(all_idx, idx, group=self.group)
                    dist.all_gather_into_tensor(all_val, val, group=self.group)
        self._stream.wait_stream(self.side)
        if gathered:
            det = torch.are_deterministic_algorithms_enabled()
            warn = torch.is_deterministic_algorithms_warn_only_enabled()
            torch.use_deterministic_algorithms(True)                 # (the sorted, fixed-order accumulation)
            try:
                for name, _, _, all_idx, all_val in gathered:
                    self.lookups[name].index_put_((all_idx,), all_val, accumulate=True)
            finally:
                torch.use_deterministic_algorithms(det, warn_only=warn)
        self._rows = {}

    @torch.no_grad()
    def broadcast_(self, model, opt) -> None:
        """Rank 0's trainable weights, optimizer state (moments, moving average, step count), the parameters outside
        the optimizer and the model's buffers on every replica; the bf16 values of the linear groups refreshed from
        the weights.  A split optimizer is gathered whole first, so rank 0's split ranges are then their owners' state
        (a checkpoint is loaded on every replica, or before the first engine splits the optimizer)."""
        opt.gather_()
        for g in opt.groups:
            for k in ("p", "m", "v", "e"):
                if k in g["flat"]:
                    dist.broadcast(g["flat"][k], 0, group=self.group)
        dist.broadcast(opt.t, 0, group=self.group)
        for n, q in model.named_parameters():
            if n not in opt.slots and q.is_cuda:
                dist.broadcast(q.data, 0, group=self.group)
        for _, b in model.named_buffers():
            if b.is_cuda:
                dist.broadcast(b, 0, group=self.group)
        torch.cuda.synchronize()
        opt.refresh_bf16()

    # ------------------------------------------------------------------ checks
    @staticmethod
    def _bits(t: torch.Tensor) -> torch.Tensor:
        """Three int64 checksums of the bits of ``t`` (the sums of its 32- or 16-bit words, of their squares and of
        each word times its position plus one, wrapping): equal for equal tensors whatever the order of the additions,
        and different when words are moved."""
        w = t.detach().reshape(-1)
        w = w.view(torch.int32) if w.element_size() == 4 else w.view(torch.int16)
        out = torch.zeros(3, dtype=torch.int64, device=w.device)
        step = 1 << 24
        for i, c in enumerate(w.split(step)):
            c = c.to(torch.int64)
            out[0] += c.sum()
            out[1] += (c * c).sum()
            out[2] += (c * torch.arange(i * step + 1, i * step + 1 + c.numel(), device=c.device)).sum()
        return out

    def checksums(self, opt, gnorm: torch.Tensor) -> torch.Tensor:
        """Checksums of every flat buffer of the optimizer but the gradients (weights, moments, moving average, bf16
        values), the step count and ``gnorm``."""
        parts: List[torch.Tensor] = []
        for g in opt.groups:
            for k in ("p", "m", "v", "e", "h"):
                if k in g["flat"]:
                    parts.append(self._bits(g["flat"][k]))
        parts += [self._bits(opt.t.reshape(1)), self._bits(gnorm.reshape(1))]
        return torch.cat(parts)

    def consistent(self, opt, gnorm: torch.Tensor) -> bool:
        """Whether every replica holds the same weights, optimizer state and gradient norm (bit for bit).  A split
        optimizer is gathered whole first: the split ranges' fp32 state is then each part's owner's on every replica,
        and the comparison is of the bf16 values, the replicated ranges, the step count and the norm."""
        opt.gather_()
        mine = self.checksums(opt, gnorm)
        every = [torch.empty_like(mine) for _ in range(self.world)]
        dist.all_gather(every, mine, group=self.group)
        return all(torch.equal(every[0], e) for e in every[1:])

    def describe(self) -> dict:
        """The reduction's layout: the layers' buckets (elements, in issue order), the early parts (elements and
        parameter names, issued once autograd has accumulated every gradient in them), the tail's elements and the
        elements of the split buckets (reduce-scattered, updated in parts)."""
        sizes = self.bucket_numels or [0]
        return {"world": self.world, "buckets": len(sizes) - 1, "bucket_numel": sizes[:-1], "tail_numel": sizes[-1],
                "early_numel": [v.numel() for v, _ in self.early], "early_params": [list(h) for _, h in self.early],
                "split_numel": self.split_numel}
