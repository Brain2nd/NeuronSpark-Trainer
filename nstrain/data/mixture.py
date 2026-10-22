"""Recipes (per-source weights + sequence length) and append-only schedules over a line's tokens.

A *line* is the trunk (the stable phase: constant learning rate, no length limit, extendable until the corpora are
exhausted) or a branch forked from a trunk cycle (decay, long context, ... with a fixed length).  A line's schedule is
a list of entries {from, recipe, seq_len, weights} in token positions of that line; entries are only ever appended
after the last prepared token, and each carries its weights verbatim, so a prepared cycle never changes.

Recipe basis (the MiniCPM team's published practice):

* domain shares 50 % web-en / 25 % web-zh / 8 % math / 17 % code -- the distribution of the tiered-training
  experiment in "Data Science and Technology Towards AGI Part I: Tiered Data Management" (arXiv 2602.09003,
  Sec. 3.4);
* tiers rise over training (L1 -> L2 -> L3 beat a 1:1:1 mix of the same tokens by +1.49 pp there, without the late
  saturation of the mix): the trunk draws L2 (plus some math L1), the decay branch L3;
* the decay keeps ~30 % of the stable distribution as replay (their decay verification: 30 % new data + 70 % default;
  MiniCPM5-1B's decay is L3-centred);
* MiniCPM4: 7T stable + 1T annealing at 4K, then 20B at 32K; MiniCPM5-1B: short decay 200B at 4K, long decay 75B at
  32K + 25B at 128K; MiniCPM5-2B's corpora: Ultra-FineWeb, Ultra-FineWeb-L3, UltraX, UltraData-Code, UltraData-Math.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from nstrain.data.sources import SOURCES

# domain shares: web_en 50, web_zh 25, math 8, code 17 in both 4K recipes
STABLE = {
    "web_en_l2": 40.0, "web_en_l2_recent": 10.0,                       # web_en 50 (recent crawl = a fifth)
    "web_zh_l2": 25.0,                                                 # web_zh 25
    "math_l2": 6.0, "math_l1": 2.0,                                    # math 8
    "code_l2": 12.0, "code_l2_stack": 5.0,                             # code 17
}
DECAY = {
    "web_en_l3_qa": 14.0, "web_en_l3_style": 11.0, "web_en_ultrax": 10.0, "web_en_l2": 15.0,   # web_en 50 (replay 15)
    "web_zh_l3_qa": 9.0, "web_zh_l3_style": 8.0, "web_zh_l2": 8.0,                               # web_zh 25 (replay 8)
    "math_l3": 6.0, "math_l2": 2.0,                                                              # math 8 (replay 2)
    "code_l3": 12.0, "code_l2": 5.0,                                                             # code 17 (replay 5)
}
LONG = dict({"long_mix": 70.0}, **{k: v * 30.0 / 100.0 for k, v in DECAY.items()})              # 70 % long documents

RECIPES: Dict[str, dict] = {
    "stable": {"seq_len": 4096, "weights": STABLE},
    "decay": {"seq_len": 4096, "weights": DECAY},
    "long": {"seq_len": 32768, "weights": LONG},
}


def check() -> None:
    for name, r in RECIPES.items():
        unknown = set(r["weights"]) - set(SOURCES)
        assert not unknown, f"{name}: unknown sources {unknown}"
    for name in ("stable", "decay"):
        dom: Dict[str, float] = {}
        for s, w in RECIPES[name]["weights"].items():
            dom[SOURCES[s].domain] = dom.get(SOURCES[s].domain, 0.0) + w
        tot = sum(dom.values())
        shares = {d: round(100 * v / tot, 3) for d, v in dom.items()}
        assert shares == {"web_en": 50.0, "web_zh": 25.0, "math": 8.0, "code": 17.0}, (name, shares)


def entry(recipe: str, from_token: float) -> dict:
    r = RECIPES[recipe]
    return {"from": float(from_token), "recipe": recipe, "seq_len": r["seq_len"], "weights": dict(r["weights"])}


def parse_plan(spec: str) -> Tuple[List[dict], float]:
    """A branch plan 'decay:25e9,long:4e9' -> (schedule entries from token 0, total tokens)."""
    out, at = [], 0.0
    for part in spec.split(","):
        name, _, n = part.partition(":")
        if name not in RECIPES or not n:
            raise ValueError(f"plan {spec!r}: each part is <recipe>:<tokens> with a recipe in {sorted(RECIPES)}")
        out.append(entry(name, at))
        at += float(n)
    return out, at


def quotas(schedule: List[dict], start: float, length: float,
           total: Optional[float] = None) -> List[Tuple[dict, Dict[str, float]]]:
    """Per-segment token quotas per source for tokens [start, start + length) of a line with this schedule."""
    if total is not None:
        length = min(length, total - start)
    out = []
    ents = sorted(schedule, key=lambda e: e["from"])
    for i, e in enumerate(ents):
        lo = e["from"]
        hi = ents[i + 1]["from"] if i + 1 < len(ents) else float("inf")
        a, b = max(lo, start), min(hi, start + length)
        if b > a:
            w = sum(e["weights"].values())
            out.append((e, {s: (b - a) * v / w for s, v in e["weights"].items()}))
    return out


check()
