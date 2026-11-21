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
the r-th part of every bucket (the rest of the bucket, this replica's own gradient, is zeroed for the next update);
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
not during a recomputation in the backward) issues the buckets of that layer.  Everything else -- the groups too small
to cut, the parts of a buffer outside the layers (the tied embedding, whose lookup gradient is the backward's last
write), and the layers' vectors (a joined dendritic pre-norm gain is written by the layer below) -- is the tail,
summed with the loss sums after the backward returns.  Every collective is issued on the one side stream, in the same
order on every process; the capture stream waits for it before the norms.

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
"""
from __future__ import annotations

import os
import re
from functools import partial
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist

from nstrain.engine import release_graphs

BUCKET_MIN_NUMEL = 1 << 24          # 16 M fp32 elements (64 MB) per bucket at least, consecutive layers merged
GROUP_MIN_NUMEL = 1 << 24           # a group smaller than this is summed whole in the tail
HEAD = 1 << 30                      # the bucket of the lookup tables' slices (issued at the readout's input)
CROSS_LAYER = (".pre_norm_gain",)   # a layer's parameters the layer below writes (a joined dendritic pre-norm): tail


class DataParallel:
    def __init__(self, group=None):
        self.group = group
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.side = None                                             # the reduction's stream (attach)
        self.layer_buckets: Dict[int, List[Tuple[torch.Tensor, bool]]] = {}   # layer -> its input hook's buckets
        #                                                              (a bucket: its view, whether it is split)
        self.tail: List[torch.Tensor] = []
        self.head: List[torch.Tensor] = []                           # the lookup tables' slices (the readout's input)
        self.lookups: Dict[str, torch.Tensor] = {}                   # weight name -> its (rows, cols) gradient view
        self._rows: Dict[str, torch.Tensor] = {}                     # weight name -> the lookup's sparse gradient
        self.bucket_numels: List[int] = []                           # in issue order (the tail last)
        self.split_numel = 0                                         # elements of the split buckets
        self._stream = None
        self._watch = False                                          # attach the input hooks (the forward)
        self._armed = False                                          # the hooks issue their buckets
        self._issued: set = set()
        self._handles: list = []                                     # the module hooks (removed by detach)

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
        NCCL's teardown waits for the graphs that hold its kernels)."""
        release_graphs()
        if dist.is_initialized():
            dist.destroy_process_group()

    # ------------------------------------------------------------------ collectives
    def barrier(self) -> None:
        dist.barrier(group=self.group, device_ids=[self.device.index])

    def all_reduce_(self, t: torch.Tensor) -> None:
        """Sum ``t`` over the replicas, in place (captured when called inside a graph capture)."""
        dist.all_reduce(t, group=self.group)

    # ------------------------------------------------------------------ the reduction hidden behind the backward
    def attach(self, model, opt, stream) -> None:
        """Cut the flat gradient buffers of ``opt`` into the layers' buckets and the tail, split the linear group's
        layer buckets among the replicas (``opt``'s layout, :meth:`nstrain.flat_adamw.FlatAdamW.shard`), and hook the
        input of every layer of ``model`` (its modules named ``...layers.<i>``); ``stream``: the stream the backward
        runs on."""
        self._stream = stream
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
        buckets: Dict[int, List[Tuple[torch.Tensor, bool]]] = {}
        tail: List[torch.Tensor] = []
        head: List[torch.Tensor] = []
        split: Dict[int, List[Tuple[int, int]]] = {}                 # group -> its split buckets' ranges
        for gi, g in enumerate(opt.groups):
            buf, n = g["flat"]["g"], g["n"]
            slots = sorted((off, name) for name, (sgi, off, _, _) in opt.slots.items() if sgi == gi)
            runs: List[Tuple[Tuple[int, bool], int, int]] = []        # ((layer or -1, split), lo, hi): consecutive
            for i, (off, name) in enumerate(slots):                  # slots
                m = of_layer.match(name)
                layer = int(m.group(1)) if m and not name.endswith(CROSS_LAYER) else -1
                if name in tables:
                    layer = HEAD
                kind = (layer, bool(g["linear"]) and 0 <= layer < HEAD and name in projections)
                hi = slots[i + 1][0] if i + 1 < len(slots) else n
                lo = 0 if i == 0 else off
                if runs and runs[-1][0] == kind:
                    runs[-1] = (kind, runs[-1][1], hi)
                else:
                    runs.append((kind, lo, hi))
            if n < GROUP_MIN_NUMEL and not any(r[0][0] == HEAD for r in runs):
                tail.append(buf)
                continue
            # merge consecutive layers from the top: a bucket is issued by the hook of its lowest layer; a bucket holds
            # projection weights only (split among the replicas) or none (all-reduced): weights read in fp32 (the
            # dendritic convolution taps, the PKM values and sub-keys, the attention's short convolution) placed between
            # a layer's projections would cut them into several buckets (nstrain.model.param_groups places the
            # projections of all layers first, contiguous)
            def close(cur, buf=buf, gi=gi):
                buckets.setdefault(cur[0], []).append((buf[cur[1]:cur[2]], cur[3]))
                if cur[3]:
                    split.setdefault(gi, []).append((cur[1], cur[2]))
            cur = None                                               # [lowest layer, lo, hi, split]
            for (layer, part), lo, hi in reversed(runs):
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
                    tail.append(buf[lo:hi])
                    continue
                if cur is not None and cur[3] == part and cur[1] == hi and cur[2] - cur[1] < BUCKET_MIN_NUMEL:
                    cur = [layer, lo, cur[2], part]
                else:
                    if cur is not None:
                        close(cur)
                    cur = [layer, lo, hi, part]
            if cur is not None:
                close(cur)
        covered = sum(v.numel() for vs in buckets.values() for v, _ in vs) + sum(v.numel() for v in tail + head)
        if covered != sum(g["n"] for g in opt.groups):
            raise RuntimeError("DataParallel: the buckets do not cover the gradient buffers exactly once")
        self.layer_buckets, self.tail, self.head = buckets, tail, head
        self.bucket_numels = [v.numel() for v in head] + \
            [v.numel() for j in sorted(buckets, reverse=True) for v, _ in buckets[j]] + [sum(v.numel() for v in tail)]
        self.split_numel = sum(hi - lo for rs in split.values() for lo, hi in rs)
        self._unhook()
        opt.shard(self.group, split)                                 # (a new layout gathers the present one first)
        for j, mod in layers.items():
            self._handles.append(mod.register_forward_pre_hook(partial(self._forward_pre_hook, j)))
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
            self._issue([(v, False) for v in self.head] if layer == HEAD else self.layer_buckets.get(layer, ()))
        return None

    def _issue(self, views) -> None:
        """Sum the (view, split) pairs ``views`` across the replicas, in order, on the side stream: a split view by a
        reduce-scatter (this replica's part of the sum stays in its part of the view; the rest of the view, this
        replica's own gradient, is zeroed for the next update), any other by an all-reduce."""
        if not views:
            return
        self.side.wait_stream(self._stream)
        with torch.cuda.stream(self.side):
            for v, split in views:
                if not split:
                    dist.all_reduce(v, group=self.group)
                    continue
                c = v.numel() // self.world
                dist.reduce_scatter_tensor(v[self.rank * c:(self.rank + 1) * c], v, group=self.group)
                if self.rank > 0:
                    v[:self.rank * c].zero_()
                if self.rank < self.world - 1:
                    v[(self.rank + 1) * c:].zero_()

    def begin(self) -> None:
        """Before the forward of an update's last micro-batch: hook the layers' inputs, issue buckets when ready."""
        self._issued = set()
        self._rows = {}
        self._watch = self._armed = True

    def forward_done(self) -> None:
        """After that forward (a recomputation in the backward attaches no hooks)."""
        self._watch = False

    def finish(self, sums: torch.Tensor) -> None:
        """After that backward: the tail and ``sums`` summed; the backward's stream waits for every collective."""
        self._armed = False
        missing = [j for j in list(self.layer_buckets) + ([HEAD] if self.head else []) if j not in self._issued]
        if missing or set(self._rows) != set(self.lookups):
            raise RuntimeError(f"DataParallel: buckets not issued {sorted(missing)}, lookup gradients "
                               f"{sorted(self._rows)} of {sorted(self.lookups)}")
        gathered = []
        for name in sorted(self.lookups):                            # (allocated on the backward's stream)
            g = self._rows[name]
            idx, val = g._indices()[0].contiguous(), g._values().contiguous()
            gathered.append((name, idx, val, idx.new_empty(self.world * idx.numel()),
                             val.new_empty((self.world * val.shape[0],) + tuple(val.shape[1:]))))
        self._issue([(v, False) for v in self.tail + [sums]])
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
        """The reduction's layout: the layers' buckets (elements, in issue order), the tail's elements and the elements
        of the split buckets (reduce-scattered, updated in parts)."""
        sizes = self.bucket_numels or [0]
        return {"world": self.world, "buckets": len(sizes) - 1, "bucket_numel": sizes[:-1], "tail_numel": sizes[-1],
                "split_numel": self.split_numel}
