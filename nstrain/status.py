"""``nstrain status RUN``: where a line stands -- trainer and supervisor, checkpoints, W&B run, data line, disk."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from nstrain import config as C
from nstrain.ckpt import Checkpoints
from nstrain.ops import running


def _json(p: Path):
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def status(run, log_lines: int = 8) -> dict:
    cfg = C.load(run)
    r = cfg.run_dir
    m = cfg.model
    out = {"name": cfg.name, "run_dir": str(r), "model": f"{m.get('D', '?')} x {m.get('num_layers', '?')}",
           "data_work": cfg.data_work, "fork_from": cfg.fork_from, "stop_at_tokens": cfg.stop_at_tokens}
    live = running(cfg)
    hb = _json(r / "heartbeat")
    if hb is not None:
        hb["age_s"] = round(time.time() - hb["time"], 1)
        hb["alive"] = live["trainer"]
    out["heartbeat"] = hb
    sup = _json(r / "supervisor.json")
    if sup is not None:
        sup["alive"] = live["supervisor"]
    out["supervisor"] = sup
    ck = Checkpoints(cfg.ckpt_dir)
    latest = ck.latest()
    cks = []
    for d in ck.complete():
        m = _json(d / "trainer.json") or {}
        cks.append({"update": m.get("update"), "tokens": m.get("tokens"), "keep": bool(m.get("keep")),
                    "cycle": (m.get("data") or {}).get("cycle"), "sample": (m.get("data") or {}).get("sample"),
                    "seq_len": (m.get("shape") or {}).get("seq_len"), "latest": latest is not None and d == latest})
    out["checkpoints"] = cks
    out["latest"] = latest.name if latest is not None else None
    out["wandb"] = dict(_json(r / "wandb_run.json") or {}, project=cfg.wandb.project, entity=cfg.wandb.entity)
    out["stop_requested"] = ((r / "STOP").read_text() or "at the next update") if (r / "STOP").exists() else None
    out["rollback_pending"] = _json(r / "rollback.json")
    w = Path(cfg.data_work)
    st = _json(w / "state.json")
    if st is not None:
        by = {}
        for k, v in st["cycles"].items():
            by.setdefault(v["status"], []).append(int(k))
        prog = _json(w / "progress.json") or {}
        out["data"] = {"line": st["line"], "cycle_tokens": st["cycle_tokens"], "total_tokens": st["total_tokens"],
                       "cycles": {k: sorted(v) for k, v in by.items()},
                       "in_training": int(prog.get("cycle", 0)) + (1 if prog.get("done") else 0),
                       "inflight": (st.get("inflight") or {}).get("cycle"),
                       "schedule": [(e["recipe"], e["from"], e["seq_len"]) for e in st["schedule"]]}
    out["disk_free_gb"] = {"run_dir": round(shutil.disk_usage(r).free / 1e9), "data_work":
                           round(shutil.disk_usage(w).free / 1e9) if w.exists() else None}
    log = r / "run.log"
    out["log_tail"] = log.read_text(errors="replace").splitlines()[-log_lines:] if log.exists() else []
    return out


def format_status(s: dict) -> str:
    L = [f"line {s['name']}  ({s['run_dir']})", f"  model {s['model']}  data {s['data_work']}"
         + (f"  fork_from {s['fork_from']}" if s["fork_from"] else "")]
    sup, hb = s["supervisor"], s["heartbeat"]
    if sup is not None:
        L.append(f"  supervisor pid {sup['pid']} {'running' if sup['alive'] else 'not running'} ({sup.get('state')}); "
                 f"launches {sup.get('launches')}, failures without progress {sup.get('fails')}, "
                 f"trainer pid {sup.get('trainer_pid')}"
                 + (" (beside other processes)" if sup.get("gpu") == "shared" else ""))
        if sup.get("last_event"):
            e = sup["last_event"]
            L.append(f"  last event {e['time']}: {e['event']}: {e['message']}")
    else:
        L.append("  supervisor: never started")
    if hb is not None:
        L.append(f"  heartbeat: update {hb['update']}, {hb['stage']!r}, {hb['age_s']:.0f} s ago, pid {hb.get('pid')} "
                 f"{'alive' if hb['alive'] else 'gone'}")
    L.append(f"  latest {s['latest']}; checkpoints: " + ", ".join(
        f"u{c['update']}{'*' if c['keep'] else ''}" for c in s["checkpoints"]) + "  (* kept)")
    w = s["wandb"]
    if w.get("run_id"):
        L.append(f"  W&B {w.get('entity') or '<default entity>'}/{w['project']}/{w['run_id']} ({w.get('kind')}, "
                 f"ready {w.get('ready')})")
    if s["stop_requested"]:
        L.append(f"  STOP requested: {s['stop_requested']}")
    if s["rollback_pending"]:
        L.append(f"  rollback pending: to update {s['rollback_pending']['to']}")
    d = s.get("data")
    if d is not None:
        L.append(f"  data {d['line']}: cycle in training {d['in_training']}, cycles {d['cycles']}"
                 + (f", preparing {d['inflight']}" if d["inflight"] is not None else "")
                 + f"; cycle {d['cycle_tokens'] / 1e9:g}B tokens, total "
                 + (f"{d['total_tokens'] / 1e9:g}B" if d["total_tokens"] else "open"))
    L.append(f"  disk free: run {s['disk_free_gb']['run_dir']} GB, data {s['disk_free_gb']['data_work']} GB")
    if s["log_tail"]:
        L.append("  run.log:")
        L += [f"    {x[:200]}" for x in s["log_tail"]]
    return "\n".join(L)
