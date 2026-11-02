"""Tokens per optimizer update as a function of the tokens trained (a batch-size ramp), the same for every sequence
length.

A line's ``tokens_per_update`` is one number -- one batch for its whole life -- or a list of pieces in token order,
each covering [from, to) of the global token count (``to`` None = open-ended)::

    {"from": 0,       "to": 11.7e9, "tokens": 393216}      # 12 micro-batches of 8 x 4096
    {"from": 11.7e9,  "to": 29e9,   "tokens": 524288}
    {"from": 29e9,    "to": None,   "tokens": 786432}

The pieces are append-only like the learning-rate schedule (nstrain.schedule): a continued line may extend them, a
token always gets the same batch on a line (:func:`same_until`).  The batch of an update is taken at the token count
before the update as the data line counts it (the tokens of the cycles' updates so far, nstrain.data.batches.plan), so
that a cycle's updates are fixed by its manifest and these pieces alone; the trainer's own count differs from it by
the samples skipped at the ends of runs of one length (fewer than one update per run).

With ``optimizer.weight_decay_batch`` set, the weight decay of an update is ``optimizer.weight_decay`` times the
update's tokens over that batch (nstrain.trainer): the AdamW time scale ``B / (lr wd D)`` in tokens then does not
move with the batch (Bergsma et al., 2505.13738: with the batch, adjust the decay, not the rate).
"""
from __future__ import annotations

import math
from functools import reduce
from typing import List, Optional, Union

Spec = Union[int, List[dict]]


def pieces(spec: Spec) -> List[dict]:
    """The pieces of a ``tokens_per_update`` value: a number is one open-ended piece from token 0."""
    if isinstance(spec, bool):
        raise TypeError("tokens_per_update: a number of tokens or a list of pieces")
    if isinstance(spec, (int, float)):
        return [{"from": 0, "to": None, "tokens": int(spec)}]
    return [dict(p) for p in spec]


def validate(ps: List[dict]) -> None:
    if not ps or ps[0]["from"] != 0:
        raise ValueError("the batch pieces must start at token 0")
    for a, b in zip(ps, ps[1:]):
        if a["to"] is None or a["to"] != b["from"]:
            raise ValueError(f"batch pieces must be contiguous: {a} then {b}")
    for p in ps:
        if set(p) != {"from", "to", "tokens"}:
            raise ValueError(f"a batch piece has from, to and tokens: {p}")
        if p["to"] is not None and not p["to"] > p["from"]:
            raise ValueError(f"a batch piece must end after it starts: {p}")
        t = p["tokens"]
        if isinstance(t, bool) or not isinstance(t, int) or t <= 0:
            raise ValueError(f"a batch piece's tokens must be a positive integer: {p}")
    if ps[-1]["to"] is not None:
        raise ValueError("the last batch piece must be open-ended (to: null)")


def at(ps: List[dict], tokens: float) -> int:
    """The tokens of the update that starts at ``tokens`` trained."""
    for p in ps:
        if p["to"] is None or tokens < float(p["to"]):
            return int(p["tokens"])
    raise ValueError(f"token {tokens} is past the end of the batch pieces ({ps[-1]['to']})")


def values(ps: List[dict]) -> List[int]:
    return [int(p["tokens"]) for p in ps]


def common_divisor(ps: List[dict]) -> int:
    """The largest number dividing every piece's tokens (a micro-batch must divide every batch of the line)."""
    return reduce(math.gcd, values(ps))


def cut(ps: List[dict], upto: float) -> List[dict]:
    """The pieces that set the batch of the tokens before ``upto``: every piece starting before it, the last one cut
    at ``upto``."""
    out = []
    for p in ps:
        if p["from"] >= upto:
            break
        q = dict(p)
        if q["to"] is None or q["to"] > upto:
            q["to"] = upto
        out.append(q)
    return out


def same_until(a: List[dict], b: List[dict], upto: float) -> bool:
    """True when two batch specifications give every token before ``upto`` the same batch."""
    return cut(a, upto) == cut(b, upto)


def end_of(ps: List[dict]) -> Optional[float]:
    return None if ps[-1]["to"] is None else float(ps[-1]["to"])
