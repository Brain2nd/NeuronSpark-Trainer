"""The validation's dendritic probe metrics, from a ``neuronspark.utils.probe.DendriticProbe`` that saw the
validation rows: ``val_<L>/dendritic/<name>`` over every dendritic head of the model and
``val_<L>/dendritic/layer<NN>/<name>`` per dendritic neuron layer (NN the model's layer index):

* ``retention_p50``: the median head's retention over the row, ``exp(A sum_t dt_t)``, the fraction of a compartment's
  potential at the row's first token still there after its last;
* ``retention_frac_gt_0.9``: the fraction of the heads keeping more than 0.9 over the row (the slow-leak heads);
* ``retention<horizon>k_frac_gt_0.5``: the fraction of the heads that would keep more than 0.5 over ``horizon`` tokens
  at the row's mean step (the probe's horizon, 32768 by default);
* ``firing``: the mean head's firing rate (the fraction of the somatic outputs released through the threshold).
"""
from __future__ import annotations

from typing import Dict

import torch

RETENTION_SLOW = 0.9                 # a head keeping more than this over the row is a slow-leak head
HORIZON_SLOW = 0.5                   # a head keeping more than this over the horizon would carry the row's start there


def _readings(summary: dict):
    r = torch.cat([v["retention"] for v in summary.values()])
    rh = torch.cat([v["retention_horizon"] for v in summary.values()])
    f = torch.cat([v["firing"] for v in summary.values()])
    return r, rh, f


def _horizon_name(probe) -> str:
    return f"retention{int(probe.horizon) // 1024}k_frac_gt_{HORIZON_SLOW:g}"


def dendritic_metrics(key: str, probe) -> Dict[str, float]:
    """The metrics of ``probe`` under ``<key>/dendritic`` (see the module docstring)."""
    s = probe.summary()
    if not s:
        raise RuntimeError("dendritic probe: the validation forward reached no dendritic neuron layer")
    hz, out = _horizon_name(probe), {}

    def put(prefix: str, r: torch.Tensor, rh: torch.Tensor, f: torch.Tensor) -> None:
        out[f"{prefix}/retention_p50"] = float(torch.quantile(r, 0.5))
        out[f"{prefix}/retention_frac_gt_{RETENTION_SLOW:g}"] = float((r > RETENTION_SLOW).float().mean())
        out[f"{prefix}/{hz}"] = float((rh > HORIZON_SLOW).float().mean())
        out[f"{prefix}/firing"] = float(f.mean())

    for i, v in s.items():
        put(f"{key}/dendritic/layer{i:02d}", v["retention"], v["retention_horizon"], v["firing"])
    put(f"{key}/dendritic", *_readings(s))
    return out


def dendritic_line(probe) -> str:
    """One log line: the model-level readings of ``probe`` and the slow-leak head counts."""
    s = probe.summary()
    if not s:
        raise RuntimeError("dendritic probe: the forward reached no dendritic neuron layer")
    r, rh, f = _readings(s)
    return (f"dendritic probe: retention p50 {float(torch.quantile(r, 0.5)):.3g}, heads > {RETENTION_SLOW:g} over the "
            f"row {int((r > RETENTION_SLOW).sum())}/{r.numel()}, > {HORIZON_SLOW:g} over {int(probe.horizon) // 1024}k "
            f"{int((rh > HORIZON_SLOW).sum())}/{r.numel()}, firing {float(f.mean()):.3f}")
