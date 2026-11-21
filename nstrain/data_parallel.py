"""Data-parallel replicas of the training step: one process per GPU, each with its own rows of the global batch.

Every process holds the whole model and the whole optimizer state (:class:`nstrain.flat_adamw.FlatAdamW`).  The step
engine (:class:`nstrain.engine.StepEngine` with ``data_parallel``) scales each micro-batch's loss by ``1 / (accum * world)``
and, once per update, sums every group's flat gradient buffer across the processes (fp32, NCCL, inside a captured CUDA
graph), so every process ends the update's micro-batches with the gradient of the mean loss over the global batch:
the same gradient the one-GPU step computes over the same rows.  The collective hands every process the same reduced
values, and everything computed from them afterwards (the norms in a fixed order, the clip, the update) is computed
the same way on every GPU, so the replicas stay bit-identical without exchanging weights.

The process group is NCCL over the GPUs of one host (``torchrun``'s environment: ``RANK``, ``WORLD_SIZE``,
``LOCAL_RANK``); it is created eagerly (bound to this process's GPU), so the communicator exists before any graph is
captured.
"""
from __future__ import annotations

import os
from typing import List

import torch
import torch.distributed as dist


class DataParallel:
    def __init__(self, group=None):
        self.group = group
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        self.device = torch.device("cuda", torch.cuda.current_device())

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

    def reduce_grads(self, opt) -> None:
        """Sum every group's flat gradient buffer over the replicas, in place (fp32)."""
        for g in opt.groups:
            self.all_reduce_(g["flat"]["g"])

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
        return {"world": self.world, "reduce": "all-reduce per group after the micro-batches"}
