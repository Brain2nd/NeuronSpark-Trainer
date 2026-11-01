"""The validation's dendritic probe metrics, from a ``neuronspark.utils.probe.DendriticProbe`` that saw the
validation rows: ``val_<L>/dendritic/<name>`` over every dendritic head of the model and
``val_<L>/dendritic/layer<NN>/<name>`` per dendritic neuron layer (NN the model's layer index):

* ``retention_p50``: the median head's retention over the row, ``exp(A sum_t dt_t)``, the fraction of a compartment's
  potential at the row's first token still there after its last;
* ``retention_frac_gt_0.9``: the fraction of the heads keeping more than 0.9 over the row (the slow-leak heads);
* ``retention_max``: the slowest head's retention over the row, and ``retention_frac_gt_0.5``: the fraction of the
  heads keeping more than 0.5 over it (the tail approaching the slow-leak heads, visible while their count is 0);
* ``retention<horizon>k_frac_gt_0.5``: the fraction of the heads that would keep more than 0.5 over ``horizon`` tokens
  at the row's mean step (the probe's horizon, 32768 by default);
* ``tau_p50``, ``tau_p90``, ``tau_max``: the heads' e-fold lengths in tokens at the row's mean step (``T / exponent``,
  the row length over the row's total decay exponent), their median, 90th percentile and the slowest head's; these
  read the heads' leak where the retention over the row underflows to 0;
* ``firing``: the mean head's firing rate (the fraction of the somatic outputs released through the threshold); with
  a probe of several segments also ``firing_seg<k>`` per segment of the row (over every head) and
  ``layer<NN>/firing_last_seg``, the layer's firing rate in the row's last segment.
"""
from __future__ import annotations

from typing import Dict

import torch

RETENTION_SLOW = 0.9                 # a head keeping more than this over the row is a slow-leak head
RETENTION_TAIL = 0.5                 # the heads keeping more than this over the row: the tail below the slow-leak ones
HORIZON_SLOW = 0.5                   # a head keeping more than this over the horizon would carry the row's start there


def _readings(summary: dict):
    r = torch.cat([v["retention"] for v in summary.values()])
    rh = torch.cat([v["retention_horizon"] for v in summary.values()])
    e = torch.cat([v["exponent"] for v in summary.values()])
    f = torch.cat([v["firing"] for v in summary.values()])
    return r, rh, e, f


def _tau(probe, e: torch.Tensor) -> torch.Tensor:
    """The heads' e-fold lengths in tokens: the rows' length over their mean decay exponent (finite: a head of no leak
    at all reads ``T * 1e12``)."""
    return float(probe.T) / e.double().clamp_min(1e-12)


def _horizon_name(probe) -> str:
    return f"retention{int(probe.horizon) // 1024}k_frac_gt_{HORIZON_SLOW:g}"


def dendritic_metrics(key: str, probe) -> Dict[str, float]:
    """The metrics of ``probe`` under ``<key>/dendritic`` (see the module docstring)."""
    s = probe.summary()
    if not s:
        raise RuntimeError("dendritic probe: the validation forward reached no dendritic neuron layer")
    hz, out = _horizon_name(probe), {}

    def put(prefix: str, r: torch.Tensor, rh: torch.Tensor, e: torch.Tensor, f: torch.Tensor) -> None:
        out[f"{prefix}/retention_p50"] = float(torch.quantile(r, 0.5))
        out[f"{prefix}/retention_frac_gt_{RETENTION_SLOW:g}"] = float((r > RETENTION_SLOW).float().mean())
        out[f"{prefix}/retention_max"] = float(r.max())
        out[f"{prefix}/retention_frac_gt_{RETENTION_TAIL:g}"] = float((r > RETENTION_TAIL).float().mean())
        out[f"{prefix}/{hz}"] = float((rh > HORIZON_SLOW).float().mean())
        tau = _tau(probe, e)
        out[f"{prefix}/tau_p50"] = float(torch.quantile(tau, 0.5))
        out[f"{prefix}/tau_p90"] = float(torch.quantile(tau, 0.9))
        out[f"{prefix}/tau_max"] = float(tau.max())
        out[f"{prefix}/firing"] = float(f.mean())

    for i, v in s.items():
        put(f"{key}/dendritic/layer{i:02d}", v["retention"], v["retention_horizon"], v["exponent"], v["firing"])
    put(f"{key}/dendritic", *_readings(s))
    if int(probe.segments) > 1:
        seg = torch.cat([v["firing_by_segment"] for v in s.values()], dim=1)       # (segments, every head)
        for k in range(seg.shape[0]):
            out[f"{key}/dendritic/firing_seg{k}"] = float(seg[k].mean())
        for i, v in s.items():
            out[f"{key}/dendritic/layer{i:02d}/firing_last_seg"] = float(v["firing_by_segment"][-1].mean())
    return out


def dendritic_line(probe) -> str:
    """One log line: the model-level readings of ``probe`` and the slow-leak head counts."""
    s = probe.summary()
    if not s:
        raise RuntimeError("dendritic probe: the forward reached no dendritic neuron layer")
    r, rh, e, f = _readings(s)
    tau = _tau(probe, e)
    return (f"dendritic probe: retention p50 {float(torch.quantile(r, 0.5)):.3g}, max {float(r.max()):.3g}, heads > "
            f"{RETENTION_SLOW:g} over the row {int((r > RETENTION_SLOW).sum())}/{r.numel()}, > {RETENTION_TAIL:g} "
            f"{int((r > RETENTION_TAIL).sum())}/{r.numel()}, > {HORIZON_SLOW:g} over {int(probe.horizon) // 1024}k "
            f"{int((rh > HORIZON_SLOW).sum())}/{r.numel()}, tau tokens p50 {float(torch.quantile(tau, 0.5)):.3g} max "
            f"{float(tau.max()):.3g}, firing {float(f.mean()):.3f}")
