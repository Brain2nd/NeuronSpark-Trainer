"""Checks before a line trains (the trainer at every start, ``nstrain check`` without a GPU).

:func:`start_problems`: the checkpoint the line continues from (or a branch's fork point) and run.json belong together.
:func:`preflight`: the data line and the held-out streams can serve this configuration, so a problem shows at the
start instead of hours later (at the first validation, the first cycle of a new recipe, ...).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from nstrain import config as C
from nstrain import schedule as sched


def start_problems(cfg: C.RunConfig, meta: dict, data_state: dict, forking: bool) -> List[str]:
    """Why the checkpoint ``meta`` (the line's newest, or ``fork_from`` at a branch's first start) and this
    configuration do not belong together ([] if they do)."""
    old, allowed, problems = meta["run"], set(cfg.allow_changes), []
    if meta["code"]["tokenizer_sha256"] != data_state["tokenizer_sha256"]:
        problems.append("the checkpoint was trained with another tokenizer")
    if meta["model_config"] != cfg.model and "model" not in allowed:
        problems.append(f"model configuration differs from the checkpoint's: {meta['model_config']} -> {cfg.model}")
    if not sched.same_until(cfg.schedule, old["schedule"], float(meta["tokens"])) and "schedule" not in allowed:
        problems.append(f"the schedule differs from the checkpoint's before token {meta['tokens']}")
    new = cfg.to_dict()
    if forking:
        d = meta["data"]
        if not meta.get("keep") or int(d["sample"]) != 0:
            problems.append(f"fork_from must be a kept cycle-end checkpoint (keep {meta.get('keep')}, "
                            f"sample {d['sample']})")
        par = data_state.get("parent") or {}
        if Path(par.get("root", "")) != Path(d["work"]) or int(par.get("cycle", -2)) + 1 != int(d["cycle"]):
            problems.append(f"the branch data {cfg.data_work} was forked from {par.get('root')} cycle "
                            f"{par.get('cycle')}, the checkpoint ends cycle {int(d['cycle']) - 1} of {d['work']}")
        if Path(meta["run_dir"]) == cfg.run_dir:
            problems.append("a branch needs its own run directory")
        end, total = sched.end_of(cfg.schedule), data_state.get("total_tokens")
        if end is not None and total is not None and end < float(meta["tokens"]) + float(total):
            problems.append(f"the schedule ends at {end:.0f} tokens, before the branch's data "
                            f"({float(meta['tokens']) + float(total):.0f})")
        problems += C.frozen_differences(new, old, allowed, [k for k in C.FROZEN if k not in C.FORK_FREE])
    else:
        problems += C.frozen_differences(new, old, allowed)
        if Path(meta["run_dir"]) != cfg.run_dir and "run_dir" not in allowed:
            problems.append(f"the checkpoint belongs to {meta['run_dir']} (allow_changes \"run_dir\" adopts it)")
    return problems


def preflight(cfg: C.RunConfig) -> List[str]:
    """The data line loads (tokenizer file and library as at init), every recipe source is calibrated and pinned,
    every sequence length of the line has engine settings and held-out rows."""
    from nstrain.data.batches import val_rows
    from nstrain.data.sources import SOURCES
    from nstrain.data.work import Work
    problems = []
    try:
        st = Work(cfg.data_work).load()
    except (OSError, RuntimeError, KeyError, ValueError) as e:
        return [f"data line {cfg.data_work}: {e}"]
    problems += C.check_lengths(cfg, st)
    for e in st["schedule"]:
        for s in e["weights"]:
            if s not in st["bytes_per_token"]:
                problems.append(f"source {s} of recipe {e['recipe']} is not calibrated in {cfg.data_work}")
            if s not in SOURCES or SOURCES[s].repo not in st["revisions"]:
                problems.append(f"source {s} of recipe {e['recipe']} has no pinned revision in {cfg.data_work}")
    val = Path(cfg.val_work) / "val"
    missing = sorted({s for e in st["schedule"] for s in e["weights"] if not any((val / s).glob("*.bin"))})
    if missing:
        problems.append(f"no held-out streams of {missing} in {val} (`nstrain data val --work {cfg.val_work}`)")
    for L in sorted({int(e["seq_len"]) for e in st["schedule"]}):
        try:
            if not val_rows(cfg.val_work, L, 1):
                problems.append(f"no held-out streams of {L + 1} tokens in {val}")
        except FileNotFoundError:
            break
    return problems
