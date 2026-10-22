"""Settings of the training host, applied before anything else is imported.

``$NSTRAIN_HOST`` or ``~/.nstrain/host.json``::

    {"neuronspark": "/path/to/NeuronSpark",            # the model repository (sys.path, PYTHONPATH of children)
     "env": {"TRITON_PTXAS_PATH": "/usr/local/cuda-13.2/bin/ptxas"},   # set when not already set
     "work_root": "/path/to/data_work",                 # default root of data lines
     "tokenizer": "/path/to/tokenizer.json",            # default tokenizer of `nstrain data init`
     "calibration_from": "/path/to/state.json",         # bytes per token adopted by new data lines
     "notify": "/path/to/command"}                      # run with an event (JSON) on stdin: a line finished, failed,
                                                        # stuck, killed by the watchdog, stopped for disk space

A missing file means no settings (everything given on the command line / in the environment).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

KEYS = ("neuronspark", "env", "work_root", "tokenizer", "calibration_from", "notify")


def path() -> Path:
    return Path(os.environ.get("NSTRAIN_HOST") or Path.home() / ".nstrain" / "host.json")


def load() -> dict:
    p = path()
    if not p.exists():
        return {}
    s = json.loads(p.read_text())
    unknown = set(s) - set(KEYS)
    if unknown:
        raise ValueError(f"{p}: unknown keys {sorted(unknown)} (known: {KEYS})")
    return s


def apply() -> dict:
    """Environment defaults and the NeuronSpark checkout on the import path (also for child processes)."""
    s = load()
    for k, v in s.get("env", {}).items():
        os.environ.setdefault(k, str(v))
    ns = s.get("neuronspark")
    if ns:
        ns = str(Path(ns).resolve())
        if ns not in sys.path:
            sys.path.insert(0, ns)
        parts = [x for x in os.environ.get("PYTHONPATH", "").split(os.pathsep) if x]
        if ns not in parts:
            os.environ["PYTHONPATH"] = os.pathsep.join([ns] + parts)
    return s


def get(key: str, default=None):
    return load().get(key, default)
