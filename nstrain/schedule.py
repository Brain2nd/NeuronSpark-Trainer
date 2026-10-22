"""Learning rate as a function of the tokens trained (warmup-stable-decay), the same for every sequence length.

A line's schedule is a list of pieces in token order; each piece covers [from, to) of the global token count (the
trunk's tokens plus, on a branch, the branch's own; ``to`` None = open-ended)::

    {"from": 0,       "to": 2e9,  "kind": "warmup"}                      # linear 0 -> peak
    {"from": 2e9,     "to": None, "kind": "constant"}                    # the trunk's stable phase
    {"from": 100e9,   "to": 129e9, "kind": "decay", "shape": "1-sqrt", "final": 0.05}   # a branch's decay

A branch inherits the trunk's pieces up to its fork point and appends its own; pieces are append-only (the same
token always gets the same rate on a line).  The rate of an update is taken at the token count before the update.
"""
from __future__ import annotations

import math
from typing import List, Optional

SHAPES = ("linear", "1-sqrt", "cosine")


def validate(pieces: List[dict]) -> None:
    if not pieces or pieces[0]["from"] != 0:
        raise ValueError("the schedule must start at token 0")
    for a, b in zip(pieces, pieces[1:]):
        if a["to"] is None or a["to"] != b["from"]:
            raise ValueError(f"schedule pieces must be contiguous: {a} then {b}")
    for p in pieces:
        if p["to"] is not None and not p["to"] > p["from"]:
            raise ValueError(f"a schedule piece must end after it starts: {p}")
        if p["kind"] not in ("warmup", "constant", "decay"):
            raise ValueError(f"unknown piece kind {p['kind']!r}")
        if p["kind"] in ("warmup", "decay") and p["to"] is None:
            raise ValueError(f"a {p['kind']} piece needs an end: {p}")
        if p["kind"] == "decay" and (p.get("shape") not in SHAPES or not 0.0 <= float(p.get("final", 0.0)) <= 1.0):
            raise ValueError(f"decay piece needs shape in {SHAPES} and 0 <= final <= 1: {p}")


def lr_at(pieces: List[dict], peak: float, tokens: float) -> float:
    """The learning rate at ``tokens`` trained.  A decay piece starts from the rate at its start (``peak`` after a
    constant piece) and ends at ``final * peak``."""
    start_rate = 0.0
    for p in pieces:
        a, b = float(p["from"]), (math.inf if p["to"] is None else float(p["to"]))
        if p["kind"] == "warmup":
            end_rate = peak
        elif p["kind"] == "constant":
            end_rate = start_rate if start_rate > 0 else peak
        else:
            end_rate = float(p["final"]) * peak
        if tokens < b:
            if p["kind"] == "warmup":
                return peak * min(1.0, (tokens - a + 1.0) / (b - a))
            if p["kind"] == "constant":
                return end_rate
            x = (tokens - a) / (b - a)
            f = {"linear": 1.0 - x, "1-sqrt": 1.0 - math.sqrt(x),
                 "cosine": 0.5 * (1.0 + math.cos(math.pi * x))}[p["shape"]]
            return end_rate + (start_rate - end_rate) * f
        start_rate = end_rate
    raise ValueError(f"token {tokens} is past the end of the schedule ({pieces[-1]['to']})")


def end_of(pieces: List[dict]) -> Optional[float]:
    return None if pieces[-1]["to"] is None else float(pieces[-1]["to"])


def branch(trunk: List[dict], at: float, decay_to: float, shape: str = "1-sqrt", final: float = 0.05) -> List[dict]:
    """The trunk's pieces up to token ``at`` (in a constant piece or at a piece boundary) plus a decay piece
    [at, decay_to)."""
    out = cut(trunk, at)
    if out[-1]["to"] != at:
        raise ValueError(f"a branch forks in a constant piece or at the end of a piece; token {at} is inside {out[-1]}")
    out.append({"from": at, "to": decay_to, "kind": "decay", "shape": shape, "final": final})
    validate(out)
    return out


def cut(pieces: List[dict], at: float) -> List[dict]:
    """The pieces that set the rates of the tokens before ``at``: every piece starting before it; a constant piece
    (whose rate does not depend on its end) cut at ``at``, a warmup or decay piece whole (its end sets its slope)."""
    out = []
    for p in pieces:
        if p["from"] >= at:
            break
        q = dict(p)
        if q["kind"] == "constant" and (q["to"] is None or q["to"] > at):
            q["to"] = at
        out.append(q)
    return out


def same_until(a: List[dict], b: List[dict], at: float) -> bool:
    """True when two schedules give every token before ``at`` the same rate and continue a warmup or decay that
    started before ``at`` the same way."""
    return cut(a, at) == cut(b, at)
