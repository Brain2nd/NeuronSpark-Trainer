"""The configuration of a training line: ``<run_dir>/run.json``.

A line (the trunk, a branch, a test line) is a directory; ``run.json`` is its configuration, everything the line
produces lives next to it (checkpoints/, run.log, heartbeat, wandb_run.json, supervisor.json; see nstrain.trainer)::

    {
     "name": "trunk",                         # W&B run name
     "model": {"vocab_size": 130560, ...},    # NeuronSparkConfig fields (written from a preset by `nstrain new`)
     "data_work": "/data/trunk",              # the line's data work directory (nstrain.data.work)
     "val_work": "/data/trunk",               # the work directory holding the held-out streams (val/)
     "fork_from": null,                       # a branch: the parent's kept cycle-end checkpoint directory
     "seed": 0, "data_seed": 0,               # model launch seed; sample order seed
     "tokens_per_update": 1048576,            # the same for every sequence length
     "lengths": {"4096": {"micro_rows": 4, "activation_checkpointing": false},
                 "32768": {"micro_rows": 1, "activation_checkpointing": 3}},   # false | k (the first k dendritic
                                                    # layers) | true (every dendritic layer) | "all" (every layer);
                                                    # optional "shared": {micro_rows, activation_checkpointing,
                                                    # reserve_gb}, the shape beside other processes measured for
                                                    # that reserve (`nstrain tune --shared`)
     "optimizer": {"peak_lr": 3e-4, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.1,
                   "lexical_lr_mult": 8.0, "grad_clip": 1.0},
     "schedule": [{"from": 0, "to": 2e9, "kind": "warmup"}, {"from": 2e9, "to": null, "kind": "constant"}],
     "checkpoints": {"every_tokens": 2e9, "keep_last": 3},
     "validation": {"every_tokens": 2e9, "rows": 16},
     "stop_at_tokens": null,                  # no update starts at or past it
     "guards": {"max_skipped_in_a_row": 10, "gpu_check_every": 50, "allow_shared_gpu": false,
                "share_reserve_gb": null},    # null: the GPU to itself; a number: train beside other processes
                                              # while they use at most this much memory (see nstrain.gpu)
     "supervisor": {"max_fails": 5, "stall_s": 1800, "restart_pause_s": 30, "gpu_wait_s": 60, "prepare": true,
                    "preparer": {"lookahead": 1, "min_free_gb": 200, "dl_workers": 8, "tok_workers": 8, "rayon": 8}},
     "allow_changes": [],                     # FROZEN keys allowed to differ from the line's checkpoints (on purpose)
     "print_every": 10,
     "wandb": {"project": "ns-pretrain", "entity": null, "retry_s": 7200}
    }

Every key is checked (unknown keys are errors, also nested ones); the sections that only steer the process
(checkpoints, guards, supervisor, ...) may be left out (defaults), everything that decides what is computed is
required, so a changed default in the code never changes a line.  :data:`FROZEN` are the settings a line keeps for
its whole life: a continued line must match its checkpoints in them (a branch its parent's, without the data and
length settings), unless the key is listed in ``allow_changes``.  ``nstrain new`` / ``nstrain branch`` write every
key.
"""
from __future__ import annotations

import dataclasses
import json
import os
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from nstrain import schedule as sched

RUN_FILE = "run.json"


@dataclass
class Shape:
    micro_rows: int
    activation_checkpointing: Union[bool, str, int] = False


@dataclass
class SharedShape(Shape):
    reserve_gb: Optional[float] = None              # the reserve it was measured for (guards.share_reserve_gb)


@dataclass
class Length(Shape):
    shared: Optional[SharedShape] = None            # the shape beside other processes (guards.share_reserve_gb)


@dataclass
class Optimizer:
    peak_lr: float
    betas: List[float]
    eps: float
    weight_decay: float
    lexical_lr_mult: float
    grad_clip: float


@dataclass
class Checkpointing:
    every_tokens: float = 2e9
    keep_last: int = 3


@dataclass
class Validation:
    every_tokens: float = 2e9
    rows: int = 16


@dataclass
class Guards:
    max_skipped_in_a_row: int = 10
    gpu_check_every: int = 50
    allow_shared_gpu: bool = False
    share_reserve_gb: Optional[float] = None


@dataclass
class Preparer:
    lookahead: int = 1
    min_free_gb: float = 200.0
    dl_workers: int = 8
    tok_workers: int = 8
    rayon: int = 8


@dataclass
class Supervisor:
    max_fails: int = 5
    stall_s: float = 1800.0
    restart_pause_s: float = 30.0
    gpu_wait_s: float = 60.0
    prepare: bool = True
    preparer: Preparer = field(default_factory=Preparer)


@dataclass
class Wandb:
    project: str
    entity: Optional[str] = None
    retry_s: float = 7200.0


@dataclass
class RunConfig:
    name: str
    model: dict
    data_work: str
    val_work: str
    tokens_per_update: int
    lengths: Dict[int, Length]
    optimizer: Optimizer
    schedule: List[dict]
    wandb: Wandb
    seed: int
    data_seed: int
    fork_from: Optional[str] = None
    checkpoints: Checkpointing = field(default_factory=Checkpointing)
    validation: Validation = field(default_factory=Validation)
    stop_at_tokens: Optional[float] = None
    guards: Guards = field(default_factory=Guards)
    supervisor: Supervisor = field(default_factory=Supervisor)
    allow_changes: List[str] = field(default_factory=list)
    print_every: int = 10
    run_dir: Path = field(default=None, metadata={"internal": True})        # where run.json lives (not stored)

    # ------------------------------------------------------------------ derived
    @property
    def ckpt_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    def length(self, seq_len: int, shared: bool = False) -> Shape:
        """The engine shape of a sequence length: its own, or (``shared``) the one beside other processes."""
        if seq_len not in self.lengths:
            raise ValueError(f"{self.name}: no settings for sequence length {seq_len} "
                             f"(lengths: {sorted(self.lengths)})")
        s = self.lengths[seq_len]
        if shared and s.shared is None:
            raise ValueError(f"{self.name}: lengths.{seq_len} has no shared shape (`nstrain tune --shared`)")
        return s.shared if shared else s

    def accum(self, seq_len: int, shared: bool = False) -> int:
        return self.tokens_per_update // (self.length(seq_len, shared).micro_rows * seq_len)

    def to_dict(self) -> dict:
        """The JSON form (as in run.json; stored in every checkpoint)."""
        return _dump(self)


# FROZEN keys (see frozen_view); a branch does not inherit FORK_FREE
FROZEN = ("tokens_per_update", "lengths.micro_rows", "seed", "data_seed", "data_work", "val_work", "validation.rows",
          "optimizer.peak_lr", "optimizer.betas", "optimizer.eps", "optimizer.weight_decay",
          "optimizer.lexical_lr_mult", "optimizer.grad_clip", "wandb.project", "wandb.entity")
FORK_FREE = ("data_work", "data_seed", "lengths.micro_rows")
CHANGEABLE = FROZEN + ("model", "schedule", "run_dir")                    # what allow_changes may name


def frozen_view(d: dict) -> dict:
    """The FROZEN values of a run config in its JSON form."""
    opt, wb = d.get("optimizer", {}), d.get("wandb", {})
    return {"tokens_per_update": d.get("tokens_per_update"),
            "lengths.micro_rows": {str(k): v.get("micro_rows") for k, v in d.get("lengths", {}).items()},
            "seed": d.get("seed"), "data_seed": d.get("data_seed"), "data_work": d.get("data_work"),
            "val_work": d.get("val_work"), "validation.rows": d.get("validation", {}).get("rows"),
            **{f"optimizer.{k}": opt.get(k) for k in ("peak_lr", "betas", "eps", "weight_decay", "lexical_lr_mult",
                                                      "grad_clip")},
            "wandb.project": wb.get("project"), "wandb.entity": wb.get("entity")}


def frozen_differences(new: dict, old: dict, allowed, keys=FROZEN) -> List[str]:
    a, b = frozen_view(new), frozen_view(old)
    return [f"{k}: {b[k]!r} -> {a[k]!r}" for k in keys if k not in allowed and a[k] != b[k]]


# ---------------------------------------------------------------------- load / save
def run_file(run) -> Path:
    """``<run_dir>/run.json`` of RUN (a run directory or its run.json)."""
    p = Path(run).expanduser().resolve()
    if p.is_dir() or p.suffix != ".json":
        p = p / RUN_FILE
    if p.name != RUN_FILE:
        raise ValueError(f"{p}: a line's configuration is <run_dir>/{RUN_FILE}")
    return p


def load(run) -> RunConfig:
    p = run_file(run)
    if not p.exists():
        raise FileNotFoundError(f"{p}: no such run (create one with `nstrain new` or `nstrain branch`)")
    cfg = from_dict(json.loads(p.read_text()), where=str(p))
    cfg.run_dir = p.parent
    return cfg


def from_dict(d: dict, where: str = "run config") -> RunConfig:
    cfg = _build(RunConfig, d, where)
    problems = validate(cfg)
    if problems:
        raise ValueError(f"{where}: " + "; ".join(problems))
    return cfg


def save(cfg: RunConfig, run_dir=None) -> Path:
    """Write ``run.json`` atomically (into ``run_dir`` or the config's own run directory)."""
    d = Path(run_dir) if run_dir is not None else cfg.run_dir
    problems = validate(cfg)
    if problems:
        raise ValueError(f"{d / RUN_FILE}: " + "; ".join(problems))
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / (RUN_FILE + ".tmp")
    tmp.write_text(json.dumps(cfg.to_dict(), indent=1) + "\n")
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, d / RUN_FILE)
    cfg.run_dir = d.resolve()
    return d / RUN_FILE


# ---------------------------------------------------------------------- validation
def validate(c: RunConfig) -> List[str]:
    p = []
    if not c.name or "/" in c.name:
        p.append(f"name {c.name!r} must be non-empty, without '/'")
    for k in ("data_work", "val_work"):
        if not Path(getattr(c, k)).is_absolute():
            p.append(f"{k} must be an absolute path")
    if c.fork_from is not None and not Path(c.fork_from).is_absolute():
        p.append("fork_from must be an absolute path")
    if c.tokens_per_update <= 0:
        p.append("tokens_per_update must be positive")
    if not c.lengths:
        p.append("lengths: at least one sequence length")
    for L, s in c.lengths.items():
        for where, x in ((f"lengths.{L}", s), (f"lengths.{L}.shared", s.shared)):
            if x is None:
                continue
            if L <= 0 or x.micro_rows <= 0:
                p.append(f"{where}: sequence length and micro_rows must be positive")
            elif c.tokens_per_update % (x.micro_rows * L):
                p.append(f"{where}: tokens_per_update {c.tokens_per_update} is not a multiple of micro_rows x "
                         f"seq_len = {x.micro_rows * L}")
            a = x.activation_checkpointing
            if not (isinstance(a, bool) or a == "all" or (isinstance(a, int) and a >= 1)):
                p.append(f"{where}.activation_checkpointing must be false, true, \"all\" or a number of dendritic "
                         "layers >= 1")
            r = getattr(x, "reserve_gb", None)
            if r is not None and not r > 0:
                p.append(f"{where}.reserve_gb must be positive (or null)")
    o = c.optimizer
    if not o.peak_lr > 0:
        p.append("optimizer.peak_lr must be positive")
    if len(o.betas) != 2 or not all(0.0 <= b < 1.0 for b in o.betas):
        p.append("optimizer.betas: two values in [0, 1)")
    if not o.eps > 0 or o.weight_decay < 0 or not o.lexical_lr_mult > 0 or not o.grad_clip > 0:
        p.append("optimizer: eps, lexical_lr_mult, grad_clip must be positive, weight_decay non-negative")
    try:
        sched.validate(c.schedule)
    except (ValueError, KeyError, TypeError) as e:
        p.append(f"schedule: {e}")
    if not c.checkpoints.every_tokens > 0 or c.checkpoints.keep_last < 1:
        p.append("checkpoints: every_tokens positive, keep_last >= 1")
    if not c.validation.every_tokens > 0 or c.validation.rows < 1:
        p.append("validation: every_tokens positive, rows >= 1")
    if c.stop_at_tokens is not None and not c.stop_at_tokens > 0:
        p.append("stop_at_tokens must be positive (or null)")
    g = c.guards
    if g.max_skipped_in_a_row < 0 or g.gpu_check_every < 1:
        p.append("guards: max_skipped_in_a_row >= 0, gpu_check_every >= 1")
    if g.share_reserve_gb is not None and not g.share_reserve_gb > 0:
        p.append("guards.share_reserve_gb must be positive (or null)")
    if g.share_reserve_gb is not None and g.allow_shared_gpu:
        p.append("guards: share_reserve_gb and allow_shared_gpu exclude each other")
    s = c.supervisor
    if s.max_fails < 1 or not s.stall_s > 0 or s.restart_pause_s < 0 or not s.gpu_wait_s > 0:
        p.append("supervisor: max_fails >= 1, stall_s and gpu_wait_s positive, restart_pause_s >= 0")
    q = s.preparer
    if q.lookahead < 0 or q.min_free_gb < 0 or min(q.dl_workers, q.tok_workers, q.rayon) < 1:
        p.append("supervisor.preparer: lookahead and min_free_gb >= 0, workers >= 1")
    bad = [k for k in c.allow_changes if k not in CHANGEABLE]
    if bad:
        p.append(f"allow_changes: unknown {bad} (one of {CHANGEABLE})")
    if c.print_every < 1:
        p.append("print_every >= 1")
    if not c.wandb.project:
        p.append("wandb.project must be set")
    return p


def check_lengths(c: RunConfig, data_state: dict) -> List[str]:
    """Every sequence length of the data line's schedule has engine settings."""
    need = sorted({int(e["seq_len"]) for e in data_state["schedule"]})
    return [f"the data line {c.data_work} uses sequence length {L}, run.json has no lengths.{L}" for L in need
            if L not in c.lengths]


# ---------------------------------------------------------------------- typed build / dump
def _build(cls, d, where: str):
    if not isinstance(d, dict):
        raise ValueError(f"{where}: expected an object, got {type(d).__name__}")
    hints = typing.get_type_hints(cls)
    known = {f.name: f for f in dataclasses.fields(cls) if not f.metadata.get("internal")}
    unknown = sorted(set(d) - set(known))
    if unknown:
        raise ValueError(f"{where}: unknown keys {unknown} (known: {sorted(known)})")
    kw = {}
    for name, f in known.items():
        if name not in d:
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                raise ValueError(f"{where}: missing key {name!r}")
            continue
        kw[name] = _value(hints[name], d[name], f"{where}.{name}")
    return cls(**kw)


def _value(tp, v, where: str):
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin is Union:
        if v is None and type(None) in args:
            return None
        errors = []
        for a in (a for a in args if a is not type(None)):
            try:
                return _value(a, v, where)
            except ValueError as e:
                errors.append(str(e))
        raise ValueError(f"{where}: {v!r} is none of {[getattr(a, '__name__', str(a)) for a in args]}")
    if dataclasses.is_dataclass(tp):
        return _build(tp, v, where)
    if origin in (list, List):
        if not isinstance(v, list):
            raise ValueError(f"{where}: expected a list")
        return [_value(args[0], x, f"{where}[{i}]") for i, x in enumerate(v)] if args else list(v)
    if origin in (dict, Dict):
        if not isinstance(v, dict):
            raise ValueError(f"{where}: expected an object")
        kt, vt = args
        out = {}
        for k, x in v.items():
            try:
                kk = kt(k)
            except (TypeError, ValueError):
                raise ValueError(f"{where}: key {k!r} is not a {kt.__name__}") from None
            out[kk] = _value(vt, x, f"{where}.{k}")
        return out
    if tp is bool:
        if not isinstance(v, bool):
            raise ValueError(f"{where}: expected true / false, got {v!r}")
        return v
    if tp is int:
        if isinstance(v, bool) or not (isinstance(v, int) or (isinstance(v, float) and v.is_integer())):
            raise ValueError(f"{where}: expected an integer, got {v!r}")
        return int(v)
    if tp is float:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{where}: expected a number, got {v!r}")
        return float(v)
    if tp is str:
        if not isinstance(v, str):
            raise ValueError(f"{where}: expected a string, got {v!r}")
        return v
    if tp is dict:
        if not isinstance(v, dict):
            raise ValueError(f"{where}: expected an object")
        return v
    raise TypeError(f"{where}: unsupported type {tp}")


def _dump(x):
    if dataclasses.is_dataclass(x):
        return {f.name: _dump(getattr(x, f.name)) for f in dataclasses.fields(x) if not f.metadata.get("internal")}
    if isinstance(x, dict):
        return {str(k): _dump(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_dump(v) for v in x]
    return x
