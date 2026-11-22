"""A line on several GPUs: the trainer (the leader, rank 0) and its followers, data-parallel replicas of the step.

``NSTRAIN_GPUS=N`` (N > 1; the GPUs are the first N of ``CUDA_VISIBLE_DEVICES``) makes the trainer the leader of N
replicas of the training step (:mod:`nstrain.data_parallel`): it starts N - 1 follower processes (``nstrain follow
<run>``, one per further GPU), each of which holds the model and the optimizer and nothing else of the line.  The
leader does everything a trainer does -- the data, the schedule, the checkpoints, the validations, W&B, the guards on
its own GPU -- and before each step the replicas make together it broadcasts a command (8 float64 values, then the
command's tensors; NCCL, in the one order every replica follows):

    STATE    the leader's weights, buffers and optimizer state on every replica (``DataParallel.broadcast_``):
             before the first engine, after the checkpoint's load or the launch calibration
    ENGINE   the step engine of a sequence length, captured on every replica at once (its warm-up and its graphs
             hold the replicas' collectives)
    UPDATE   one update: the learning rate, the weight decay scale, the moving average's step and the update's rows
             (inputs and targets, (2, R, L) int64); replica r trains rows [r R / N, (r + 1) R / N)
    GATHER   the optimizer's split state whole on every replica (``FlatAdamW.gather_``): before a checkpoint, a
             validation (the leader then holds every weight and moment in fp32)
    CHECK    the replicas compared bit for bit (``DataParallel.consistent``: the bf16 values, the replicated ranges,
             the moments, the step count, the gradient norm): before every checkpoint; a difference ends the run
    PROBE    the loss of one micro-batch computed by every replica (no gradient) and compared: once, before the
             leader's first update (a replica whose forward differs -- a buffer, a flag set when a state is loaded
             -- would train another model than the one the checkpoints hold)
    EMA_INIT the parameters' moving average started on every replica as a copy of the whole parameters
             (``FlatAdamW.gather_`` then ``ema_init``), at the update the line's ``ema.from_tokens`` is reached
    EMA_SWAP the parameters and their average exchanged on every replica (``FlatAdamW.swap_ema``: the split state and
             the bf16 values are gathered, collectives): before and after the leader's validation of the average

What a checkpoint of a line on N GPUs holds is what a one-GPU checkpoint holds, written by the leader after GATHER and
CHECK: the fp32 weights and the buffers, the moments and the step count, the data position, the counters, the W&B
run, the rows logged since the checkpoint before, and the update's shape (micro-batch rows, length, micro-batches of
the whole update), plus ``gpus`` (N, for the record).  Nothing of a replica is kept: the split of the optimizer is
made again from the whole state (the same parts for the same N), the bf16 values are the fp32 weights rounded, the
gradient buffers are empty at an update boundary, the training forward draws no random numbers, and a follower gets
its state from the leader (STATE; it then loads its own state dict, so that what a module derives when a state is
loaded -- the hippocampal layers' PKM flag -- is derived there too).  A line may continue on another N.
    EXIT     release the engine, leave the process group, exit

The line's arithmetic does not change with N: an update is the same ``tokens_per_update`` tokens in the same
micro-batches (``lengths.micro_rows``); each replica runs 1 / N of the update's micro-batches and the gradients are
summed inside the step, so the update is the one-GPU update of the same rows up to the order of the fp32 sums.  A
checkpoint written by N replicas is read by any number of them (it holds the whole state, as a one-GPU checkpoint).
The number of micro-batches of every update must divide by N (refused otherwise, with the count).

A follower dies with the leader (PR_SET_PDEATHSIG, SIGKILL: it holds nothing on disk).  The leader checks that
every follower is alive before each command and ends as a failure when one is not (the supervisor starts the line
again from its last checkpoint).  ``NSTRAIN_REPLICA_CPUS="a-b,c-d;e-f,g-h"``: the CPUs of rank 0; rank 1; ... (the
NUMA node of each GPU), applied by each process to itself.

Not with a GPU shared with other processes (``guards.share_reserve_gb``, ``guards.allow_shared_gpu``): refused at
the start.  The parameters' moving average (``ema``) is kept on every replica (its step is part of the captured
update; the split ranges' average is updated by their owners and gathered with the weights and the moments).
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from typing import Optional, Set, Tuple

import torch
import torch.distributed as dist

ENV = "NSTRAIN_GPUS"
CPUS_ENV = "NSTRAIN_REPLICA_CPUS"
LEADER = "NSTRAIN_LEADER_PID"
STATE, ENGINE, UPDATE, GATHER, EXIT, CHECK, PROBE, EMA_INIT, EMA_SWAP = 1, 2, 3, 4, 5, 6, 7, 8, 9
PROBE_RTOL = 1e-4                    # the replicas' loss on one micro-batch: equal up to the kernels' fp32 sums
LIVE: list = []                      # the leader's Replicas whose followers may be alive (kill_all)
EXIT_WAIT_S = 120.0


def world() -> int:
    """The number of replicas asked for (1: the one-GPU trainer)."""
    n = int(os.environ.get(ENV, "1") or 1)
    if n < 1:
        raise ValueError(f"{ENV}={n}")
    return n


def _cpus(rank: int) -> Optional[Set[int]]:
    parts = (os.environ.get(CPUS_ENV) or "").split(";")
    if rank >= len(parts) or not parts[rank].strip():
        return None
    cpus: Set[int] = set()
    for piece in parts[rank].split(","):
        a, _, b = piece.strip().partition("-")
        cpus.update(range(int(a), int(b or a) + 1))
    return cpus


def pin(rank: int) -> None:
    """This process on the CPUs ``NSTRAIN_REPLICA_CPUS`` names for ``rank`` (nothing named: as it is)."""
    cpus = _cpus(rank)
    if cpus:
        os.sched_setaffinity(0, cpus)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Replicas:
    """The leader's side: the followers' processes, the process group, the commands."""

    def __init__(self, run, n: int):
        from nstrain.data_parallel import DataParallel
        if torch.cuda.device_count() < n:
            raise RuntimeError(f"{ENV}={n} with {torch.cuda.device_count()} visible GPU(s)")
        port = _free_port()
        common = {"WORLD_SIZE": str(n), "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)}
        self.procs = []
        for r in range(1, n):
            env = dict(os.environ, **common, RANK=str(r), LOCAL_RANK=str(r))
            env[LEADER] = str(os.getpid())
            self.procs.append(subprocess.Popen([sys.executable, "-u", "-m", "nstrain", "follow", str(run)], env=env))
        os.environ.update(common, RANK="0", LOCAL_RANK="0")
        pin(0)
        self.world = n
        try:
            self.rep = DataParallel.from_env()                       # (waits for every follower to join)
        except BaseException:
            self.kill()
            raise
        self.dev = self.rep.device
        self.ctrl = torch.zeros(8, dtype=torch.float64, device=self.dev)
        self.rows: Optional[torch.Tensor] = None
        self.synced = False
        self.probed = False
        self.open = True
        LIVE.append(self)

    # ------------------------------------------------------------------ commands
    def check(self) -> None:
        dead = [(r + 1, p.returncode) for r, p in enumerate(self.procs) if p.poll() is not None]
        if dead:
            raise RuntimeError(f"data-parallel follower(s) exited (rank, status): {dead}")

    def tell(self, op: int, *vals: float) -> None:
        self.check()
        msg = [float(op)] + [float(v) for v in vals]
        self.ctrl.copy_(torch.tensor(msg + [0.0] * (8 - len(msg)), dtype=torch.float64))
        dist.broadcast(self.ctrl, 0, group=self.rep.group)

    def sync(self, model, opt) -> None:
        """The leader's state on every replica, once (before the first engine)."""
        if not self.synced:
            self.tell(STATE)
            self.rep.broadcast_(model, opt)
            self.synced = True

    def engine(self, L: int, accum: int) -> None:
        self.tell(ENGINE, L, accum)

    def split(self, micro_batches: int) -> int:
        """The micro-batches of one replica of an update of ``micro_batches``."""
        if micro_batches % self.world or micro_batches < self.world:
            raise ValueError(f"an update of {micro_batches} micro-batches does not divide among {self.world} GPUs "
                             f"({ENV})")
        return micro_batches // self.world

    def update(self, inputs: torch.Tensor, targets: torch.Tensor, lr: float, wd_scale: float,
               ema_c: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Send the update's rows and settings; returns the leader's rows (the first R / N)."""
        R, L = int(inputs.shape[0]), int(inputs.shape[1])
        if R % self.world:
            raise ValueError(f"an update of {R} rows does not divide among {self.world} GPUs ({ENV})")
        if self.rows is None or tuple(self.rows.shape) != (2, R, L):
            self.rows = torch.empty((2, R, L), dtype=torch.int64, device=self.dev)
        self.rows[0].copy_(inputs)
        self.rows[1].copy_(targets)
        self.tell(UPDATE, lr, wd_scale, ema_c, R, L)
        dist.broadcast(self.rows, 0, group=self.rep.group)
        n = R // self.world
        return inputs[:n], targets[:n]

    def gather(self, opt) -> None:
        """The optimizer's state whole on the leader (and on every replica) when it is split and was updated."""
        if opt.sharding is not None and opt.stale:
            self.tell(GATHER)
            opt.gather_()

    def same_forward(self, model, arena, x: torch.Tensor, y: torch.Tensor) -> list:
        """The loss of the micro-batch (x, y) on every replica; raises when one differs from the leader's by more
        than PROBE_RTOL.  Returns the losses in rank order."""
        buf = torch.empty((2,) + tuple(x.shape), dtype=torch.int64, device=self.dev)
        buf[0].copy_(x)
        buf[1].copy_(y)
        self.tell(PROBE, x.shape[0], x.shape[1])
        dist.broadcast(buf, 0, group=self.rep.group)
        every = _probe(model, arena, buf, self.rep)
        self.probed = True
        if any(abs(v - every[0]) > PROBE_RTOL * abs(every[0]) or v != v for v in every):
            raise RuntimeError(f"the data-parallel replicas compute different losses on the same micro-batch: {every}")
        return every

    def ema_init(self, opt) -> None:
        """The moving average started on every replica (the caller in the trainer's arena)."""
        self.tell(EMA_INIT)
        opt.gather_()
        opt.ema_init()

    def swap_ema(self, opt) -> None:
        """The parameters and their average exchanged on every replica (the caller in the trainer's arena)."""
        self.tell(EMA_SWAP)
        opt.swap_ema()

    def consistent(self, opt, gnorm: torch.Tensor) -> bool:
        """Whether every replica holds the same weights and optimizer state (a collective on every replica)."""
        self.tell(CHECK)
        return self.rep.consistent(opt, gnorm)

    # ------------------------------------------------------------------ the end
    def close(self, engine=None) -> None:
        """The clean end: the followers told to leave, the leader's engine released, the group destroyed."""
        if not self.open:
            return
        self.open = False
        try:
            self.tell(EXIT)
            if engine is not None:
                engine.release()
            self.rep.close()
            end = time.time() + EXIT_WAIT_S
            for p in self.procs:
                try:
                    p.wait(timeout=max(1.0, end - time.time()))
                except subprocess.TimeoutExpired:
                    pass
        finally:
            self.kill()

    def kill(self) -> None:
        if self in LIVE:
            LIVE.remove(self)
        for p in self.procs:
            if p.poll() is None:
                p.kill()
        for p in self.procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass


def kill_all() -> None:
    for r in list(LIVE):
        r.kill()


def _probe(model, arena, buf: torch.Tensor, rep) -> list:
    """The loss of the micro-batch ``buf`` (2, rows, L) on this replica, gathered from every replica (rank order)."""
    with arena.use(), torch.no_grad():
        mine = model.loss(buf[0], buf[1]).loss.detach().float().reshape(1).clone()
    torch.cuda.synchronize()
    every = [torch.empty_like(mine) for _ in range(rep.world)]
    dist.all_gather(every, mine, group=rep.group)
    return [float(v) for v in every]


def follow(run) -> int:
    """A follower: rank ``RANK`` of the leader's process group, on GPU ``LOCAL_RANK`` of the visible ones."""
    from nstrain.proc import die_with_parent
    die_with_parent(LEADER, signal.SIGKILL)                          # before the heavy imports
    rank = int(os.environ["RANK"])
    pin(rank)
    from nstrain import config as C
    from nstrain import model as M
    from nstrain.data_parallel import DataParallel
    from nstrain.engine import StepEngine
    from nstrain.flat_adamw import FlatAdamW
    from nstrain.gpu import Arena
    from nstrain.trainer import param_bucket
    c = C.load(run)
    rep = DataParallel.from_env()
    dev = rep.device
    model = M.build(c.model, dev, init="empty")
    o = c.optimizer
    opt = FlatAdamW(model, M.param_groups(model, o.weight_decay, o.lexical_lr_mult), tuple(o.betas), o.eps,
                    ema=c.ema is not None, bucket=param_bucket)
    torch.cuda.empty_cache()
    arena, engine, rows = Arena(), None, None
    ctrl = torch.zeros(8, dtype=torch.float64, device=dev)
    print(f"replica {rank} of {rep.world}: ready (pid {os.getpid()}, GPU "
          f"{os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')[rank:rank + 1] or [rank]})", flush=True)
    while True:
        dist.broadcast(ctrl, 0, group=rep.group)
        op, a, b, cc, d, e = ctrl.tolist()[:6]
        op = int(op)
        if op == EXIT:
            break
        if op == STATE:
            rep.broadcast_(model, opt)
            model.load_state_dict(model.state_dict(), strict=True)  # (what a module derives at a load: see above)
        elif op == GATHER:
            with arena.use():
                opt.gather_()
        elif op == CHECK:
            with arena.use():
                rep.consistent(opt, engine.gnorm)
        elif op == EMA_INIT:
            with arena.use():
                opt.gather_()
                opt.ema_init()
        elif op == EMA_SWAP:
            with arena.use():
                opt.swap_ema()
        elif op == PROBE:
            buf = torch.empty((2, int(a), int(b)), dtype=torch.int64, device=dev)
            dist.broadcast(buf, 0, group=rep.group)
            _probe(model, arena, buf, rep)
            del buf
        elif op == ENGINE:
            L, accum = int(a), int(b)
            if engine is not None:
                engine.release()
                engine = None
                arena.close()
                arena = Arena()
            s = c.length(L, False)
            M.recompute(model, s.activation_checkpointing)
            t0 = time.time()
            engine = StepEngine(model, opt, s.micro_rows, L, accum, o.grad_clip, arena=arena, data_parallel=rep,
                                before_capture=lambda: None)         # (the leader's validation warm-up: the engine
                                                                     # gathers the optimizer before it, a collective)
            torch.cuda.empty_cache()
            print(f"replica {rank}: engine {accum} x {s.micro_rows} x {L} captured in {time.time() - t0:.0f} s, "
                  f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
        elif op == UPDATE:
            R, L = int(d), int(e)
            if rows is None or tuple(rows.shape) != (2, R, L):
                rows = torch.empty((2, R, L), dtype=torch.int64, device=dev)
            dist.broadcast(rows, 0, group=rep.group)
            n = R // rep.world
            engine.update(rows[0, rank * n:(rank + 1) * n], rows[1, rank * n:(rank + 1) * n], a, b, cc)
        else:
            raise RuntimeError(f"replica {rank}: unknown command {op}")
    if engine is not None:
        engine.release()
    rep.close()
    print(f"replica {rank}: left", flush=True)
    return 0
