"""Data-parallel replicas of the training step: one process per GPU, each with its own rows of the global batch.

Every process holds the whole model and the whole optimizer state (:class:`nstrain.flat_adamw.FlatAdamW`).  The step
engine (:class:`nstrain.engine.StepEngine` with ``data_parallel``) scales each micro-batch's loss by
``1 / (accum * world)`` and, once per update, sums every group's flat gradient buffer across the processes (fp32,
NCCL, inside the captured CUDA graph of the update's last micro-batch), so every process ends the update's
micro-batches with the gradient of the mean loss over the global batch: the same gradient the one-GPU step computes
over the same rows.  The collective hands every process the same reduced values, and everything computed from them
afterwards (the norms in a fixed order, the clip, the update) is computed the same way on every GPU, so the replicas
stay bit-identical without exchanging weights.

The reduction is hidden behind the backward of the update's last micro-batch.  The flat gradient buffers of the large
groups are cut into buckets aligned to the model's layers (the slots of a layer's parameters in a group's buffer;
consecutive layers are merged until a bucket holds at least ``BUCKET_MIN_NUMEL`` elements); a bucket is summed (on a
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
        self.layer_buckets: Dict[int, List[torch.Tensor]] = {}       # layer -> the buckets its input's hook issues
        self.tail: List[torch.Tensor] = []
        self.head: List[torch.Tensor] = []                           # the lookup tables' slices (the readout's input)
        self.lookups: Dict[str, torch.Tensor] = {}                   # weight name -> its (rows, cols) gradient view
        self._rows: Dict[str, torch.Tensor] = {}                     # weight name -> the lookup's sparse gradient
        self.bucket_numels: List[int] = []                           # in issue order (the tail last)
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
        """Cut the flat gradient buffers of ``opt`` into the layers' buckets and the tail, and hook the input of every
        layer of ``model`` (its modules named ``...layers.<i>``); ``stream``: the stream the backward runs on."""
        self._stream = stream
        self.side = torch.cuda.Stream()
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
        buckets: Dict[int, List[torch.Tensor]] = {}
        tail: List[torch.Tensor] = []
        head: List[torch.Tensor] = []
        for gi, g in enumerate(opt.groups):
            buf, n = g["flat"]["g"], g["n"]
            slots = sorted((off, name) for name, (sgi, off, _, _) in opt.slots.items() if sgi == gi)
            runs: List[Tuple[int, int, int]] = []                    # (layer or -1, lo, hi): consecutive slots
            for i, (off, name) in enumerate(slots):
                m = of_layer.match(name)
                layer = int(m.group(1)) if m and not name.endswith(CROSS_LAYER) else -1
                if name in tables:
                    layer = HEAD
                hi = slots[i + 1][0] if i + 1 < len(slots) else n
                lo = 0 if i == 0 else off
                if runs and runs[-1][0] == layer:
                    runs[-1] = (layer, runs[-1][1], hi)
                else:
                    runs.append((layer, lo, hi))
            if n < GROUP_MIN_NUMEL and not any(r[0] == HEAD for r in runs):
                tail.append(buf)
                continue
            # merge consecutive layers from the top: a bucket is issued by the hook of its lowest layer
            cur = None                                               # [lowest layer, lo, hi]
            for layer, lo, hi in reversed(runs):
                if layer == HEAD:                                    # (a slot of its own: a table and its padding)
                    if cur is not None:
                        buckets.setdefault(cur[0], []).append(buf[cur[1]:cur[2]])
                        cur = None
                    head.append(buf[lo:hi])
                    continue
                if layer < 0:
                    if cur is not None:
                        buckets.setdefault(cur[0], []).append(buf[cur[1]:cur[2]])
                        cur = None
                    tail.append(buf[lo:hi])
                    continue
                if cur is not None and cur[1] == hi and cur[2] - cur[1] < BUCKET_MIN_NUMEL:
                    cur = [layer, lo, cur[2]]
                else:
                    if cur is not None:
                        buckets.setdefault(cur[0], []).append(buf[cur[1]:cur[2]])
                    cur = [layer, lo, hi]
            if cur is not None:
                buckets.setdefault(cur[0], []).append(buf[cur[1]:cur[2]])
        covered = sum(v.numel() for vs in buckets.values() for v in vs) + sum(v.numel() for v in tail + head)
        if covered != sum(g["n"] for g in opt.groups):
            raise RuntimeError("DataParallel: the buckets do not cover the gradient buffers exactly once")
        self.layer_buckets, self.tail, self.head = buckets, tail, head
        self.bucket_numels = [v.numel() for v in head] + \
            [v.numel() for j in sorted(buckets, reverse=True) for v in buckets[j]] + [sum(v.numel() for v in tail)]
        self.detach()
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
        """Remove the hooks :meth:`attach` installed (an engine's release; a new engine attaches again)."""
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
            self._issue(self.head if layer == HEAD else self.layer_buckets.get(layer, ()))
        return None

    def _issue(self, views) -> None:
        if not views:
            return
        self.side.wait_stream(self._stream)
        with torch.cuda.stream(self.side):
            for v in views:
                dist.all_reduce(v, group=self.group)

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
        self._issue(self.tail + [sums])
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
        the weights."""
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
        """Whether every replica holds the same weights, optimizer state and gradient norm (bit for bit)."""
        mine = self.checksums(opt, gnorm)
        every = [torch.empty_like(mine) for _ in range(self.world)]
        dist.all_gather(every, mine, group=self.group)
        return all(torch.equal(every[0], e) for e in every[1:])

    def describe(self) -> dict:
        """The reduction's layout: the layers' buckets (elements, in issue order) and the tail's elements."""
        sizes = self.bucket_numels or [0]
        return {"world": self.world, "buckets": len(sizes) - 1, "bucket_numel": sizes[:-1], "tail_numel": sizes[-1]}
