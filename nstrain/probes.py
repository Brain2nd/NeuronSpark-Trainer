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
  ``layer<NN>/firing_last_seg``, the layer's firing rate in the row's last segment, and ``retention_seg<k>_p50`` /
  ``retention_seg<k>_max``, the median and the slowest head's retention from the row's first token to the end of
  segment k (over every head): the decay curve of the row's start along the long rows, past the training length;
* at the rows' document boundaries (an EOS followed by the next document's first token): ``boundaries``, their count
  (over every head only), and when it is not 0: ``boundary_retention_p50`` and ``boundary_retention_max``, the median
  and the slowest head's retention across a boundary (``exp(A (dt_e + dt_e+1))``, what is left of the old document's
  potential after the new one's first token); ``boundary_step_ratio_p50`` and ``boundary_step_ratio_max``, the heads'
  mean step over a boundary's two tokens over their mean step elsewhere (well above 1: a head that flushes its
  compartments at the boundaries).

The hippocampal probe metrics, from a ``neuronspark.utils.probe.HippocampalProbe`` that saw the same rows:
``val_<L>/hippocampal/layer<NN>/<name>`` per hippocampal layer and ``val_<L>/hippocampal/<name>`` over the model:

* ``decay_tau_p50``, ``decay_tau_p90``, ``decay_tau_max``: the decay channels' e-fold lengths in tokens at the row's
  mean step (the row length over the channel's decay exponent; the layer's 256 channels, or every layer's);
* ``clock_tau_head<h>`` per layer and ``clock_tau_p50``, ``clock_tau_max`` over the model: the tokens over which a
  head's pool clock (its eligibility bias) runs one e-fold at the row's mean step;
* ``clock_boundary_step_ratio``: the clock's mean increment over a document boundary's two tokens over its increment
  elsewhere (per layer; the mean over the layers for the model), with the probe's ``boundaries`` (over the model);
* ``write_head<h>`` per layer and ``write_p50``, ``write_max`` over the model: the mean write tag of a head (the
  fraction of a token's value entering the pool);
* ``gate_firing``, ``ffn_in_firing``, ``ffn_gate_firing``, ``ffn_up_firing``: the firing rates of the write gate,
  the FFN input neuron and the spiking FFN's gate and up neurons; ``copy_pi``, ``novelty``, ``read_gain``: the copy
  arbitration's mean factors; ``pkm_entropy``, ``pkm_slots_used``, ``pkm_gate``: the PKM read's mean top-k weight
  entropy, the fraction of its slots read over the rows and its mean read gate (per layer; means over the layers for
  the model);
* ``rope_ext_q_head<h>``, ``rope_ext_k_head<h>`` per layer and ``rope_ext_q_p50``, ``rope_ext_q_max``,
  ``rope_ext_k_p50``, ``rope_ext_k_max`` over the model: the share of a head's query / key energy in the RoPE pairs
  whose wavelength exceeds the training row length (the pairs whose relative phases beyond that length training
  never showed), against ``rope_ext_pairs_frac``, the share such pairs are of all (a direction-free vector's share);
* ``attn_entropy_head<h>`` per layer and ``attn_entropy_p50`` / ``_max`` over the model: the entropy (nats) of the
  pool read's weights per query, the mean over the queries of the rows' last training-length positions;
* ``attn_dist_bin<k>`` per layer (the mean over its heads) and over the model (over every head): the pool read's
  weight by distance for the queries of the rows' last training-length positions -- bin 0 the query's own event,
  bin k >= 1 the events 2^(k-1) to 2^k - 1 back -- for the bins a row of the probe's length can reach;
  ``attn_beyond_train_head<h>`` per layer and ``attn_beyond_train_p50`` / ``_max`` over the model: the weight on
  events the training row length or more back (0 on rows of the training length; on the long rows, what is read
  from beyond that length); ``attn_mean_log2_head<h>`` per layer and ``attn_mean_log2_p50`` / ``_max`` over the
  model: the weight-mean of log2(1 + distance).  The probe takes these on a subset of the rows (its ``attn_stride``).

The optimizer's state per parameter group (:func:`optimizer_metrics`, read off the flat buffers of nstrain.flat_adamw
at a validation; ``opt/<group>/<name>``) and per bucket of parameters (the model's layers and its other parts,
nstrain.trainer.param_bucket; ``opt/<bucket>/<name>``, the norms only):

* ``param_norm``: the norm of the group's parameters; ``m_norm``: the norm of its first moments;
* ``grad_rms``: ``sqrt(mean v)``, the root-mean-square gradient the second moments estimate;
* ``update_norm``: the norm of the step AdamW would take next at the current base rate (the group's rate multiplier
  and the bias corrections of the current step count, without the weight decay), and ``update_ratio``, that over
  ``param_norm``: how much of the weights one update moves.

The loss by the target's position in its document (:class:`DocPosition`, from the per-token cross entropy of the
validation rows; ``val_<L>/docpos/bin<k>``, the recipe-weighted mean over the sources): bin 0 the first token of a
document (predicted from the EOS before it), bin k >= 1 the tokens 2^(k-1) to 2^k - 1 after it, for the bins a row
of the length can reach; the tokens before a row's first EOS (a document the row cuts into) are left out
(``val_<L>/docpos/tokens_frac``: the share of the tokens counted).  How much the model gains from each doubling of
the context it has of a document.
"""
from __future__ import annotations

import math
from typing import Dict

import torch

from neuronspark.utils.probe import ATTN_BINS

DOCPOS_BINS = 17                     # position-in-document bins: 0 the document's first token, k >= 1: 2^(k-1) .. 2^k - 1

# the version of the metric set this nstrain records at a validation (the keys of _validate, _probe_long and
# optimizer_metrics and what they measure); run.json ``metrics`` names the set a line records, and a trainer of another
# version does not start that line (nstrain.checks.preflight): the set of a line is decided before it starts, not
# patched along the way
METRICS_VERSION = "2026-11-03"

RETENTION_SLOW = 0.9                 # a head keeping more than this over the row is a slow-leak head
RETENTION_TAIL = 0.5                 # the heads keeping more than this over the row: the tail below the slow-leak ones
HORIZON_SLOW = 0.5                   # a head keeping more than this over the horizon would carry the row's start there


def _readings(summary: dict):
    r = torch.cat([v["retention"] for v in summary.values()])
    rh = torch.cat([v["retention_horizon"] for v in summary.values()])
    e = torch.cat([v["exponent"] for v in summary.values()])
    f = torch.cat([v["firing"] for v in summary.values()])
    br = torch.cat([v["boundary_retention"] for v in summary.values()])
    bs = torch.cat([v["boundary_step"] for v in summary.values()])
    bb = torch.cat([v["body_step"] for v in summary.values()])
    return r, rh, e, f, br, bs, bb


def _boundaries(summary: dict) -> int:
    """The boundaries the probe saw (the same count in every layer: one row set)."""
    counts = {int(v["boundaries"]) for v in summary.values()}
    if len(counts) != 1:
        raise RuntimeError(f"dendritic probe: the layers saw different boundary counts {sorted(counts)}")
    return counts.pop()


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
    hz, out, nb = _horizon_name(probe), {}, _boundaries(s)

    def put(prefix: str, r: torch.Tensor, rh: torch.Tensor, e: torch.Tensor, f: torch.Tensor, br: torch.Tensor,
            bs: torch.Tensor, bb: torch.Tensor) -> None:
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
        if nb:
            ratio = bs / bb.clamp_min(1e-12)
            out[f"{prefix}/boundary_retention_p50"] = float(torch.quantile(br, 0.5))
            out[f"{prefix}/boundary_retention_max"] = float(br.max())
            out[f"{prefix}/boundary_step_ratio_p50"] = float(torch.quantile(ratio, 0.5))
            out[f"{prefix}/boundary_step_ratio_max"] = float(ratio.max())

    for i, v in s.items():
        put(f"{key}/dendritic/layer{i:02d}", v["retention"], v["retention_horizon"], v["exponent"], v["firing"],
            v["boundary_retention"], v["boundary_step"], v["body_step"])
    put(f"{key}/dendritic", *_readings(s))
    out[f"{key}/dendritic/boundaries"] = nb
    if int(probe.segments) > 1:
        seg = torch.cat([v["firing_by_segment"] for v in s.values()], dim=1)       # (segments, every head)
        ret = torch.cat([v["retention_by_segment"] for v in s.values()], dim=1)    # (segments, every head)
        for k in range(seg.shape[0]):
            out[f"{key}/dendritic/firing_seg{k}"] = float(seg[k].mean())
            out[f"{key}/dendritic/retention_seg{k}_p50"] = float(torch.quantile(ret[k], 0.5))
            out[f"{key}/dendritic/retention_seg{k}_max"] = float(ret[k].max())
        for i, v in s.items():
            out[f"{key}/dendritic/layer{i:02d}/firing_last_seg"] = float(v["firing_by_segment"][-1].mean())
    return out


def dendritic_line(probe) -> str:
    """One log line: the model-level readings of ``probe`` and the slow-leak head counts."""
    s = probe.summary()
    if not s:
        raise RuntimeError("dendritic probe: the forward reached no dendritic neuron layer")
    r, rh, e, f, br, bs, bb = _readings(s)
    tau, nb = _tau(probe, e), _boundaries(s)
    at = ""
    if nb:
        ratio = bs / bb.clamp_min(1e-12)
        at = (f": step ratio p50 {float(torch.quantile(ratio, 0.5)):.3g} max {float(ratio.max()):.3g}, retention "
              f"across p50 {float(torch.quantile(br, 0.5)):.3g} max {float(br.max()):.3g}")
    return (f"dendritic probe: retention p50 {float(torch.quantile(r, 0.5)):.3g}, max {float(r.max()):.3g}, heads > "
            f"{RETENTION_SLOW:g} over the row {int((r > RETENTION_SLOW).sum())}/{r.numel()}, > {RETENTION_TAIL:g} "
            f"{int((r > RETENTION_TAIL).sum())}/{r.numel()}, > {HORIZON_SLOW:g} over {int(probe.horizon) // 1024}k "
            f"{int((rh > HORIZON_SLOW).sum())}/{r.numel()}, tau tokens p50 {float(torch.quantile(tau, 0.5)):.3g} max "
            f"{float(tau.max()):.3g}, firing {float(f.mean()):.3f}, boundaries {nb}{at}")


HIPPOCAMPAL_MEANS = ("gate_firing", "ffn_in_firing", "ffn_gate_firing", "ffn_up_firing", "copy_pi", "novelty",
                     "read_gain", "pkm_entropy", "pkm_slots_used", "pkm_gate")


def _hippocampal_readings(summary: dict):
    """Over every layer: the channels' decay exponents, the heads' clock exponents and write tags, the per-layer
    scalars stacked (layers,) per name, and the clock step ratios of the layers (NaN where a layer saw no boundary)."""
    e = torch.cat([v["decay_exponent"] for v in summary.values()])
    c = torch.cat([v["clock_exponent"] for v in summary.values()])
    w = torch.cat([v["write"] for v in summary.values()])
    rq = torch.cat([v["rope_ext_q"] for v in summary.values()])
    rk = torch.cat([v["rope_ext_k"] for v in summary.values()])
    means = {name: torch.stack([v[name].reshape(()) for v in summary.values()]) for name in HIPPOCAMPAL_MEANS}
    ratio = torch.stack([(v["clock_boundary_step"] / v["clock_body_step"].clamp_min(1e-12)).reshape(())
                         for v in summary.values()])
    return e, c, w, means, ratio, rq, rk


class DocPosition:
    """Sums of the per-token cross entropy and token counts by position-in-document bin, per source (see the module
    docstring); ``add`` takes the rows' inputs (B, T) and their per-token cross entropy (B, T) on the device."""

    def __init__(self, eos_id: int, T: int) -> None:
        self.eos, self.T = int(eos_id), int(T)
        self.sums: Dict[str, torch.Tensor] = {}
        self.counts: Dict[str, torch.Tensor] = {}
        self.seen: Dict[str, int] = {}

    @torch.no_grad()
    def add(self, source: str, inputs: torch.Tensor, ce: torch.Tensor) -> None:
        B, T = (int(x) for x in inputs.shape)
        if T != self.T or tuple(ce.shape) != (B, T):
            raise ValueError(f"DocPosition: rows of {self.T} tokens, got inputs {tuple(inputs.shape)} and "
                             f"cross entropy {tuple(ce.shape)}")
        pos = torch.arange(T, device=inputs.device)
        last = torch.cummax(torch.where(inputs == self.eos, pos.expand(B, T), torch.full_like(inputs, -1)),
                            dim=1).values                                # the last EOS at or before each token
        valid = last >= 0
        d = pos.expand(B, T) - last                                      # 0: the EOS itself predicts the first token
        bins = torch.where(d <= 0, torch.zeros_like(d), torch.floor(torch.log2(d.clamp_min(1).double())).long() + 1)
        bins = bins.clamp_max(DOCPOS_BINS - 1)[valid]
        s = torch.zeros(DOCPOS_BINS, dtype=torch.float64, device=inputs.device).index_add_(0, bins, ce[valid].double())
        n = torch.bincount(bins, minlength=DOCPOS_BINS).double()
        self.sums[source] = self.sums.get(source, torch.zeros_like(s)) + s
        self.counts[source] = self.counts.get(source, torch.zeros_like(n)) + n
        self.seen[source] = self.seen.get(source, 0) + B * T

    def metrics(self, key: str, recipe: dict) -> Dict[str, float]:
        """``<key>/docpos/bin<k>`` for the bins a row can reach, the recipe-weighted mean over the recipe's sources
        of their mean cross entropy in the bin (a source without tokens in a bin is left out of that bin), and
        ``<key>/docpos/tokens_frac``."""
        mine = {n: float(w) for n, w in recipe.items() if w > 0 and n in self.sums}
        out = {}
        reach = attn_bins(self.T)
        for k in range(reach):
            num = den = 0.0
            for n, w in mine.items():
                c = float(self.counts[n][k])
                if c > 0:
                    num += w * float(self.sums[n][k]) / c
                    den += w
            if den > 0:
                out[f"{key}/docpos/bin{k}"] = num / den
        counted = sum(float(self.counts[n].sum()) for n in mine)
        seen = sum(self.seen[n] for n in mine)
        out[f"{key}/docpos/tokens_frac"] = counted / seen if seen else 0.0
        return out


def attn_bins(T: int) -> int:
    """The distance bins a row of ``T`` tokens can reach (bin 0: the query's own event; bin k >= 1: 2^(k-1) to
    2^k - 1 back), at most ATTN_BINS."""
    return min(ATTN_BINS, int(math.floor(math.log2(T - 1))) + 2) if T >= 2 else 1


def _attention_readings(summary: dict):
    """Over every layer: the heads' weight by distance (heads, ATTN_BINS), beyond the training length (heads,), mean
    log2 distance (heads,) and entropy (heads,)."""
    d = torch.cat([v["attn_dist"] for v in summary.values()])
    b = torch.cat([v["attn_beyond"] for v in summary.values()])
    m = torch.cat([v["attn_mean_log2"] for v in summary.values()])
    e = torch.cat([v["attn_entropy"] for v in summary.values()])
    return d, b, m, e


def hippocampal_metrics(key: str, probe) -> Dict[str, float]:
    """The metrics of ``probe`` (a HippocampalProbe) under ``<key>/hippocampal`` (see the module docstring)."""
    s = probe.summary()
    if not s:
        raise RuntimeError("hippocampal probe: the validation forward reached no hippocampal layer")
    nb, out, reach = _boundaries(s), {}, attn_bins(int(probe.T))
    for i, v in s.items():
        p = f"{key}/hippocampal/layer{i:02d}"
        tau = _tau(probe, v["decay_exponent"])
        out[f"{p}/decay_tau_p50"] = float(torch.quantile(tau, 0.5))
        out[f"{p}/decay_tau_p90"] = float(torch.quantile(tau, 0.9))
        out[f"{p}/decay_tau_max"] = float(tau.max())
        ct = _tau(probe, v["clock_exponent"])
        for h in range(ct.numel()):
            out[f"{p}/clock_tau_head{h}"] = float(ct[h])
        if nb:
            ratio = v["clock_boundary_step"] / v["clock_body_step"].clamp_min(1e-12)
            out[f"{p}/clock_boundary_step_ratio"] = float(ratio)
        for h in range(v["write"].numel()):
            out[f"{p}/write_head{h}"] = float(v["write"][h])
        for h in range(v["rope_ext_q"].numel()):
            out[f"{p}/rope_ext_q_head{h}"] = float(v["rope_ext_q"][h])
            out[f"{p}/rope_ext_k_head{h}"] = float(v["rope_ext_k"][h])
        for k in range(reach):
            out[f"{p}/attn_dist_bin{k}"] = float(v["attn_dist"][:, k].mean())
        for h in range(v["attn_beyond"].numel()):
            out[f"{p}/attn_beyond_train_head{h}"] = float(v["attn_beyond"][h])
            out[f"{p}/attn_mean_log2_head{h}"] = float(v["attn_mean_log2"][h])
            out[f"{p}/attn_entropy_head{h}"] = float(v["attn_entropy"][h])
        for name in HIPPOCAMPAL_MEANS:
            out[f"{p}/{name}"] = float(v[name].reshape(()))
    e, c, w, means, ratio, rq, rk = _hippocampal_readings(s)
    ad, ab, am, ae = _attention_readings(s)
    p = f"{key}/hippocampal"
    tau = _tau(probe, e)
    out[f"{p}/decay_tau_p50"] = float(torch.quantile(tau, 0.5))
    out[f"{p}/decay_tau_p90"] = float(torch.quantile(tau, 0.9))
    out[f"{p}/decay_tau_max"] = float(tau.max())
    ct = _tau(probe, c)
    out[f"{p}/clock_tau_p50"] = float(torch.quantile(ct, 0.5))
    out[f"{p}/clock_tau_max"] = float(ct.max())
    if nb:
        out[f"{p}/clock_boundary_step_ratio"] = float(ratio.mean())
    out[f"{p}/write_p50"] = float(torch.quantile(w, 0.5))
    out[f"{p}/write_max"] = float(w.max())
    out[f"{p}/rope_ext_q_p50"] = float(torch.quantile(rq, 0.5))
    out[f"{p}/rope_ext_q_max"] = float(rq.max())
    out[f"{p}/rope_ext_k_p50"] = float(torch.quantile(rk, 0.5))
    out[f"{p}/rope_ext_k_max"] = float(rk.max())
    out[f"{p}/rope_ext_pairs_frac"] = float(next(iter(s.values()))["rope_ext_pairs"])
    for k in range(reach):
        out[f"{p}/attn_dist_bin{k}"] = float(ad[:, k].mean())
    out[f"{p}/attn_beyond_train_p50"] = float(torch.quantile(ab, 0.5))
    out[f"{p}/attn_beyond_train_max"] = float(ab.max())
    out[f"{p}/attn_mean_log2_p50"] = float(torch.quantile(am, 0.5))
    out[f"{p}/attn_mean_log2_max"] = float(am.max())
    out[f"{p}/attn_entropy_p50"] = float(torch.quantile(ae, 0.5))
    out[f"{p}/attn_entropy_max"] = float(ae.max())
    for name in HIPPOCAMPAL_MEANS:
        out[f"{p}/{name}"] = float(means[name].mean())
    out[f"{p}/boundaries"] = nb
    return out


OPT_CHUNK = 1 << 28                  # elements per pass over a flat buffer (1 GiB of fp32 scratch at most)


@torch.no_grad()
def optimizer_metrics(opt, base_lr: float) -> Dict[str, float]:
    """The state of a nstrain.flat_adamw.FlatAdamW per parameter group and per bucket (see the module docstring),
    summed in float64 over chunks of the flat buffers (the padding between the parameters is zero in every buffer)."""
    out = {}
    t = float(opt.t)
    bc1 = 1.0 - opt.b1 ** t
    bc2_sqrt = math.sqrt(max(1.0 - opt.b2 ** t, 0.0))
    if opt.buckets and t >= 1.0:                                      # per bucket: the parameters' slices
        p2 = {b: 0.0 for b in opt.buckets}
        u2 = {b: 0.0 for b in opt.buckets}
        for name, (gi, off, k, _) in opt.slots.items():
            f, b = opt.groups[gi]["flat"], opt.slot_bucket[name]
            p, m, v = f["p"][off:off + k], f["m"][off:off + k], f["v"][off:off + k]
            p2[b] += float(torch.sum(p.double() * p.double()))
            step = m / (torch.sqrt(v) / bc2_sqrt + opt.eps)
            u2[b] += (float(base_lr) * opt.groups[gi]["lr_mult"] / bc1) ** 2 * float(torch.sum(step.double() * step.double()))
            del step
        for b in opt.buckets:
            pn, un = math.sqrt(p2[b]), math.sqrt(u2[b])
            out[f"opt/{b}/param_norm"] = pn
            out[f"opt/{b}/update_norm"] = un
            out[f"opt/{b}/update_ratio"] = un / pn if pn > 0 else 0.0
    for g in opt.groups:
        f, n = g["flat"], int(g["n"])
        count = sum(int(p.numel()) for p in g["params"])
        p2 = m2 = u2 = v1 = 0.0
        for a in range(0, n, OPT_CHUNK):
            p, m, v = f["p"][a:a + OPT_CHUNK], f["m"][a:a + OPT_CHUNK], f["v"][a:a + OPT_CHUNK]
            p2 += float(torch.sum(p.double() * p.double()))
            m2 += float(torch.sum(m.double() * m.double()))
            v1 += float(torch.sum(v.double()))
            if t >= 1.0:
                step = m / (torch.sqrt(v) / bc2_sqrt + opt.eps)        # the kernel's step before the rate
                u2 += float(torch.sum(step.double() * step.double()))
                del step
        name = g["name"]
        pn = math.sqrt(p2)
        out[f"opt/{name}/param_norm"] = pn
        out[f"opt/{name}/m_norm"] = math.sqrt(m2)
        out[f"opt/{name}/grad_rms"] = math.sqrt(v1 / max(count, 1))
        un = (float(base_lr) * g["lr_mult"] / bc1) * math.sqrt(u2) if t >= 1.0 else 0.0
        out[f"opt/{name}/update_norm"] = un
        out[f"opt/{name}/update_ratio"] = un / pn if pn > 0 else 0.0
    return out


def hippocampal_line(probe) -> str:
    """One log line: the model-level readings of a HippocampalProbe."""
    s = probe.summary()
    if not s:
        raise RuntimeError("hippocampal probe: the forward reached no hippocampal layer")
    e, c, w, means, ratio, rq, rk = _hippocampal_readings(s)
    ad, ab, am, ae = _attention_readings(s)
    nb = _boundaries(s)
    tau, ct = _tau(probe, e), _tau(probe, c)
    m = {k: float(v.mean()) for k, v in means.items()}
    at = f": clock step ratio {float(ratio.mean()):.3g}" if nb else ""
    return (f"hippocampal probe: decay tau tokens p50 {float(torch.quantile(tau, 0.5)):.3g} max "
            f"{float(tau.max()):.3g}, "
            f"clock tau p50 {float(torch.quantile(ct, 0.5)):.3g} max {float(ct.max()):.3g}, write p50 "
            f"{float(torch.quantile(w, 0.5)):.3g} max {float(w.max()):.3g}, firing gate {m['gate_firing']:.3f} ffn in "
            f"{m['ffn_in_firing']:.3f} gate {m['ffn_gate_firing']:.3f} up {m['ffn_up_firing']:.3f}, copy pi "
            f"{m['copy_pi']:.3g} novelty {m['novelty']:.3f} read gain {m['read_gain']:.3g}, pkm entropy "
            f"{m['pkm_entropy']:.3f} slots used {m['pkm_slots_used']:.3f} gate {m['pkm_gate']:.3f}, rope ext share q "
            f"p50 {float(torch.quantile(rq, 0.5)):.3f} k p50 {float(torch.quantile(rk, 0.5)):.3f} of pairs "
            f"{float(next(iter(s.values()))['rope_ext_pairs']):.3f}, attn beyond train p50 "
            f"{float(torch.quantile(ab, 0.5)):.3g} max {float(ab.max()):.3g} mean log2 distance p50 "
            f"{float(torch.quantile(am, 0.5)):.3g} self {float(ad[:, 0].mean()):.3f} entropy p50 "
            f"{float(torch.quantile(ae, 0.5)):.3g}, boundaries {nb}{at}")
