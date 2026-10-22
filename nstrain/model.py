"""The NeuronSpark model as the trainer uses it: model configs (presets or files), construction, launch calibration, and
the parameter groups of AdamW (the recipe of the repository's training benchmark).

Model presets: ``nstrain/presets/models/<name>.json`` (NeuronSparkConfig fields): ``ns-2b`` (2048 x 24, the production
model), ``ns-1p3b`` (the repository's 1.3B), ``tiny-32k`` (8 layers, for quick runs).  Presets and the run defaults per
preset (:data:`RUN_DEFAULTS`: engine settings per sequence length, tokens per update, warmup) are templates of
``nstrain new``: a line's run.json holds the resolved values, so a later change of a preset never changes a line.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

PRESETS = Path(__file__).resolve().parent / "presets" / "models"

# the AdamW recipe of the repository's training benchmark
OPTIMIZER = {"betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.1, "lexical_lr_mult": 8.0, "grad_clip": 1.0}

# what `nstrain new` writes for a preset (shapes measured on a GB300)
RUN_DEFAULTS: Dict[str, dict] = {
    "ns-2b": {"tokens_per_update": 1048576,
              "lengths": {"4096": {"micro_rows": 4, "activation_checkpointing": False},
                          "32768": {"micro_rows": 1, "activation_checkpointing": True}},
              "warmup_tokens": 2e9},
    "ns-1p3b": {"tokens_per_update": 1048576,
                "lengths": {"4096": {"micro_rows": 4, "activation_checkpointing": False},
                            "32768": {"micro_rows": 1, "activation_checkpointing": True}},
                "warmup_tokens": 2e9},
    "tiny-32k": {"tokens_per_update": 262144,
                 "lengths": {"4096": {"micro_rows": 4, "activation_checkpointing": False},
                             "32768": {"micro_rows": 1, "activation_checkpointing": True}},
                 "warmup_tokens": 4 * 262144},
}


def presets() -> List[str]:
    return sorted(p.stem for p in PRESETS.glob("*.json"))


def model_config(spec: str) -> dict:
    """The model configuration named by ``spec``: a preset name or a path to a JSON file."""
    p = PRESETS / f"{spec}.json"
    if "/" not in spec and p.exists():
        return json.loads(p.read_text())
    q = Path(spec).expanduser()
    if q.suffix == ".json" and q.exists():
        return json.loads(q.read_text())
    raise ValueError(f"model {spec!r}: neither a preset ({', '.join(presets())}) nor a model-config JSON file")


def build(cfg: dict, device, init: str):
    """The model on ``device``: ``init`` "launch" (the repository's launch initialisation; seed the RNG first) or
    "empty" (to be loaded from a checkpoint)."""
    from neuronspark import NeuronSparkConfig
    from neuronspark.models.neuronspark_lm import NeuronSparkLMHeadModel
    return NeuronSparkLMHeadModel(NeuronSparkConfig(**cfg), device=device, init=init)


def dendritic_layers(cfg: dict) -> int:
    """The number of dendritic neuron layers of a model configuration."""
    from neuronspark import NeuronSparkConfig
    return NeuronSparkConfig(**cfg).n_dendritic_layers


def recompute(model, setting) -> None:
    """Apply an activation-recomputation setting of run.json: false (none), true (every dendritic neuron layer), an
    integer k (the first k dendritic neuron layers: what a memory budget needs, no more), "all" (every layer)."""
    if setting is True or setting == "all":
        model.gradient_checkpointing = True
    elif setting is False:
        model.gradient_checkpointing = False
    else:
        model.gradient_checkpointing = int(setting)
    model.hippocampal_checkpointing = setting == "all"


def calibrate(model, inputs, micro_rows: int, device) -> None:
    """Launch calibration of a fresh model on the first rows of the line (deterministic)."""
    from neuronspark.utils.calibration import CALIBRATION_ROWS, calibrate_fresh_model
    n = max(1, CALIBRATION_ROWS // micro_rows)
    cal = [inputs[i * micro_rows:(i + 1) * micro_rows].to(device) for i in range(n)]
    calibrate_fresh_model(model, cal, rows=n * micro_rows)

WEIGHT_DECAY = 0.1
LEXICAL_LR_MULT = 8.0
# dendritic neuron population vectors decayed with the matrices (their own group)
DENDRITIC_DECAYED_VECTORS = ("A_log", "dt_bias", "D", "gamma", "conv_bias", "pre_norm_gain")


def param_groups(model, weight_decay: float = WEIGHT_DECAY, lexical_lr_mult: float = LEXICAL_LR_MULT) -> List[Dict]:
    """0 matrices (decayed), 1 vectors / scalars (not decayed), 2 lexical n-gram tables (not decayed, lr x
    ``lexical_lr_mult``), 3 dendritic population vectors (decayed); each group carries ``lr_mult``."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    lexical = {id(p) for p in model.backbone.lexical_memory.tables}
    dvec = {id(p) for n, p in named if ".mixer_a." in n and n.rsplit(".", 1)[-1] in DENDRITIC_DECAYED_VECTORS}
    groups = [
        {"params": [p for n, p in named if p.ndim >= 2 and id(p) not in lexical and id(p) not in dvec],
         "weight_decay": weight_decay, "lr_mult": 1.0, "name": "matrices"},
        {"params": [p for n, p in named if p.ndim < 2 and id(p) not in lexical and id(p) not in dvec],
         "weight_decay": 0.0, "lr_mult": 1.0, "name": "vectors"},
        {"params": [p for n, p in named if id(p) in lexical], "weight_decay": 0.0, "lr_mult": float(lexical_lr_mult),
         "name": "lexical_tables"},
        {"params": [p for n, p in named if id(p) in dvec], "weight_decay": weight_decay, "lr_mult": 1.0,
         "name": "dendritic_vectors"},
    ]
    if sum(len(g["params"]) for g in groups) != len(named):
        raise RuntimeError("param_groups: the groups do not cover every trainable parameter exactly once")
    return groups
