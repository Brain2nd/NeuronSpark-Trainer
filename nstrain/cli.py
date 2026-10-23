"""``nstrain``: the NeuronSpark trainer's command line (one entry point for data, lines, supervision, checks).

    nstrain data init --work W --eos-id 1 --cycle-tokens 2e10 [--tokenizer T] [--total-tokens N | --plan P]
    nstrain data {calibrate,val} --work W [--sources ...]        nstrain data adopt-calibration --work W --from STATE
    nstrain data {prepare,release,rebuild} --work W --cycle K    nstrain data append --work W --recipe R --from-token T
    nstrain data fork --work B --from TRUNK --at-cycle K --plan decay:25e9,long:4e9 --line b1 [--cycle-tokens C]
    nstrain data {plan,status} --work W                         nstrain data loop --work W   (one preparer process)
    nstrain data run-preparer --work W [--foreground]   the preparer under supervision (no trainer), in the background

    nstrain new RUN --model ns-2b --data W [--val W] [--peak-lr ..] [--warmup-tokens ..] [--wandb-project ..]
    nstrain branch RUN --from PARENT_RUN --at-cycle K --plan decay:25e9,long:4e9 --data W_B [--cycle-tokens ..]
    nstrain check RUN          start checks without a GPU
    nstrain run RUN [--foreground]   the line under supervision (trainer + data preparer), in the background
    nstrain train RUN          one trainer process (what the supervisor runs)
    nstrain status RUN [--json]
    nstrain stop RUN [--at-cycle-end K | --now]   nstrain rollback RUN --to UPDATE [--force]
    nstrain bench --model ns-2b --seq-len 4096 --rows 4 --accum 4 [--ac none|dendritic|all] [--profile]
    nstrain tune RUN [--margin-gib 10] [--dry-run] [--shared]   the fastest shape per length that fits the GPU (beside
                                                    other processes: --shared), into run.json
    nstrain presets

RUN is a line's run directory (or its run.json).  Host settings: nstrain.host.
"""
from __future__ import annotations

import argparse
import json
import sys


def _workers(p) -> None:
    p.add_argument("--dl-workers", type=int, default=8)
    p.add_argument("--tok-workers", type=int, default=8)
    p.add_argument("--rayon", type=int, default=8)


def _kw(a) -> dict:
    return dict(dl_workers=a.dl_workers, tok_workers=a.tok_workers, rayon=a.rayon)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="nstrain", description="The NeuronSpark trainer.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ---- data
    d = sub.add_parser("data", help="data lines: init, calibrate, prepare, loop, fork, ...")
    ds = d.add_subparsers(dest="dcmd", required=True)
    p = ds.add_parser("init")
    p.add_argument("--work", required=True)
    p.add_argument("--tokenizer")
    p.add_argument("--eos-id", type=int, required=True)
    p.add_argument("--cycle-tokens", type=float, required=True)
    p.add_argument("--total-tokens", type=float)
    p.add_argument("--plan", help="recipe:tokens,... (default: the open-ended stable recipe of a trunk)")
    p.add_argument("--line", default="trunk")
    for name in ("calibrate", "val"):
        p = ds.add_parser(name)
        p.add_argument("--work", required=True)
        p.add_argument("--sources", nargs="*")
        _workers(p)
    p = ds.add_parser("adopt-calibration")
    p.add_argument("--work", required=True)
    p.add_argument("--from", dest="src", required=True)
    for name in ("prepare", "release", "rebuild"):
        p = ds.add_parser(name)
        p.add_argument("--work", required=True)
        p.add_argument("--cycle", type=int, required=True)
        _workers(p)
    p = ds.add_parser("append")
    p.add_argument("--work", required=True)
    p.add_argument("--recipe", required=True)
    p.add_argument("--from-token", type=float, required=True)
    p = ds.add_parser("fork")
    p.add_argument("--work", required=True)
    p.add_argument("--from", dest="src", required=True)
    p.add_argument("--at-cycle", type=int, required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--line", required=True)
    p.add_argument("--cycle-tokens", type=float, required=True)
    _workers(p)
    p = ds.add_parser("plan")
    p.add_argument("--work", required=True)
    p.add_argument("--cycles", type=int, default=30)
    p = ds.add_parser("status")
    p.add_argument("--work", required=True)
    for name in ("loop", "run-preparer"):
        p = ds.add_parser(name)
        p.add_argument("--work", required=True)
        p.add_argument("--lookahead", type=int, default=1)
        p.add_argument("--min-free-gb", type=float, default=200.0)
        _workers(p)
        if name == "run-preparer":
            p.add_argument("--foreground", action="store_true",
                           help="in this terminal (its hangup stops it); by default in the background, independent of "
                                "the login (output in <work>/preparer-supervisor.out)")

    # ---- lines
    p = sub.add_parser("new", help="a new trunk-type line (run directory with run.json)")
    p.add_argument("run")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--val")
    p.add_argument("--name")
    p.add_argument("--peak-lr", type=float, default=3e-4)
    p.add_argument("--warmup-tokens", type=float)
    p.add_argument("--tokens-per-update", type=int)
    p.add_argument("--shape-from", help="the preset whose run defaults a model-config file uses")
    p.add_argument("--wandb-project", default="ns-pretrain")
    p = sub.add_parser("branch", help="a branch line from a parent's cycle end (data fork + run.json)")
    p.add_argument("run")
    p.add_argument("--from", dest="parent", required=True)
    p.add_argument("--at-cycle", type=int, required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--cycle-tokens", type=float)
    p.add_argument("--name")
    p.add_argument("--decay-shape", default="1-sqrt")
    p.add_argument("--final-lr-frac", type=float, default=0.05)
    _workers(p)
    for name, h in (("check", "start checks without a GPU"), ("run", "the line under supervision"),
                    ("train", "one trainer process")):
        p = sub.add_parser(name, help=h)
        p.add_argument("run")
        if name == "run":
            p.add_argument("--foreground", action="store_true",
                           help="the supervisor in this terminal (its hangup stops the line); by default it runs in "
                                "the background, independent of the login (output in <run>/supervisor.out)")
    p = sub.add_parser("status")
    p.add_argument("run")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("stop", help="stop with a checkpoint at the next update boundary (or a cycle end)")
    p.add_argument("run")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--at-cycle-end", type=int, help="after the last update of this data cycle")
    g.add_argument("--now", action="store_true", help="SIGTERM to the supervisor (stops at the next boundary too)")
    p = sub.add_parser("rollback", help="continue the line from an earlier checkpoint as a new W&B run")
    p.add_argument("run")
    p.add_argument("--to", type=int, required=True)
    p.add_argument("--force", action="store_true", help="also move checkpoints that are fork points of branches")
    p = sub.add_parser("bench", help="throughput of one training shape")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--model", default="ns-2b", help="a model preset or a model-config file")
    g.add_argument("--run", help="the model of this line's run.json")
    p.add_argument("--result", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--skip", type=int, default=5, help="updates left out of the median (warm-up)")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--updates", type=int, default=12)
    p.add_argument("--ac", default="none", help="activation recomputation: none, dendritic, all, or the number of "
                                                "dendritic layers recomputed")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--shared", action="store_true", help="beside other processes on the GPU (no check for them)")
    p.add_argument("--max-gib", type=float, help="PyTorch memory cap of the process (out of memory past it)")
    p.add_argument("--top", type=int, default=60)
    p = sub.add_parser("tune", help="the fastest shape of every sequence length that fits the GPU, into run.json")
    p.add_argument("run")
    p.add_argument("--margin-gib", type=float, default=10.0)
    p.add_argument("--updates", type=int, default=4, help="complete updates per measured shape (the first is warm-up)")
    p.add_argument("--dry-run", action="store_true", help="measure and report only")
    p.add_argument("--shared", action="store_true",
                   help="the shapes beside other processes, leaving guards.share_reserve_gb to them "
                        "(lengths.<L>.shared)")
    sub.add_parser("presets")
    return ap


def _data(a) -> int:
    if a.dcmd == "loop":
        import signal
        from nstrain.proc import SUPERVISOR, die_with_parent
        die_with_parent(SUPERVISOR, signal.SIGTERM)
        from nstrain.data.preparer import loop
        loop(a.work, a.lookahead, a.min_free_gb, **_kw(a))
        return 0
    from nstrain import host
    from nstrain.data import mixture
    from nstrain.data.sources import SOURCES
    from nstrain.data.work import Work
    w = Work(a.work)
    if a.dcmd == "run-preparer":
        from nstrain.config import Preparer
        from nstrain.supervise import run_preparer, start_preparer_background
        if a.foreground:
            return run_preparer(a.work, Preparer(a.lookahead, a.min_free_gb, a.dl_workers, a.tok_workers, a.rayon))
        args = ["--lookahead", str(a.lookahead), "--min-free-gb", str(a.min_free_gb), "--dl-workers",
                str(a.dl_workers), "--tok-workers", str(a.tok_workers), "--rayon", str(a.rayon)]
        try:
            pid = start_preparer_background(a.work, args)
        except RuntimeError as e:
            print(f"nstrain data run-preparer: {e}", file=sys.stderr)
            return 2
        print(f"preparer started in the background (pid {pid}), independent of this login; log "
              f"{w.root.resolve() / 'preparer.log'}")
        return 0
    if a.dcmd == "plan":
        st = w.load()
        C, T = st["cycle_tokens"], st["total_tokens"]
        n = a.cycles if T is None else min(a.cycles, int(-(-T // C)))
        for k in range(n):
            segs = mixture.quotas(st["schedule"], k * C, C, T)
            print(k, " | ".join(f"{e['recipe']}@{e['seq_len']}: {sum(q.values())/1e9:.2f}B" for e, q in segs))
        return 0
    if a.dcmd == "status":
        st = w.load()
        print(json.dumps({k: st[k] for k in ("line", "parent", "total_tokens", "cycle_tokens", "cursor", "cycles")},
                         indent=1))
        print([{k: e[k] for k in ("from", "recipe", "seq_len")} for e in st["schedule"]])
        print({k: round(v, 3) for k, v in st["bytes_per_token"].items()})
        return 0
    with w.lock():                                                   # the writers of state.json: one at a time
        if a.dcmd == "init":
            tok = a.tokenizer or host.get("tokenizer")
            if not tok:
                raise SystemExit("--tokenizer (or \"tokenizer\" in the host settings)")
            sched, total = mixture.parse_plan(a.plan) if a.plan else (None, a.total_tokens)
            w.init(tok, a.eos_id, a.cycle_tokens, total, sched, a.line)
            src = host.get("calibration_from")
            if src:
                st = w.load()
                st["bytes_per_token"].update(json.loads(open(src).read())["bytes_per_token"])
                w.save(st)
                print(f"bytes per token adopted from {src}")
        elif a.dcmd == "calibrate":
            for n, m in sorted(w.calibrate(sorted(a.sources or SOURCES), **_kw(a)).items()):
                print(f"{n:18s} {m['file_bytes']/1e6:9.1f} MB  {m['tokens']/1e6:9.2f} M tokens  "
                      f"{m['file_bytes']/max(1, m['tokens']):6.3f} B/tok  {m['text_bytes']/max(1, m['tokens']):6.3f} "
                      f"text B/tok  {m['docs']:8d} docs  {m['tokens']/max(1e-9, m['seconds'])/1e6:7.2f} M tok/s")
        elif a.dcmd == "adopt-calibration":
            st = w.load()
            st["bytes_per_token"].update(json.loads(open(a.src).read())["bytes_per_token"])
            w.save(st)
            print({k: round(v, 3) for k, v in st["bytes_per_token"].items()})
        elif a.dcmd == "val":
            for n, m in sorted(w.build_val(sorted(a.sources or SOURCES), **_kw(a)).items()):
                print(f"{n:18s} {m['tokens']/1e6:9.2f} M tokens  {m['bin']}")
        elif a.dcmd == "prepare":
            m = w.prepare(a.cycle, **_kw(a))
            for s, ps in sorted(m["sources"].items()):
                print(f"{s:18s} {sum(p['count'] for p in ps)/1e9:7.3f} B tokens in {len(ps)} pieces")
        elif a.dcmd == "release":
            w.release(a.cycle)
        elif a.dcmd == "rebuild":
            m = w.rebuild(a.cycle, **_kw(a))
            print(f"rebuilt cycle {a.cycle}: {sum(len(v) for v in m['sources'].values())} pieces, blake2b checked")
        elif a.dcmd == "append":
            st = w.append(a.recipe, a.from_token)
            print(json.dumps([{k: e[k] for k in ("from", "recipe", "seq_len")} for e in st["schedule"]]))
        elif a.dcmd == "fork":
            sched, total = mixture.parse_plan(a.plan)
            st = w.fork(a.src, a.at_cycle, sched, total, a.cycle_tokens, a.line, **_kw(a))
            print(f"forked {a.line} from {a.src} cycle {a.at_cycle} ({st['parent']['tokens']/1e9:.1f}B trunk "
                  f"tokens); total {total/1e9:.1f}B; schedule {[(e['recipe'], e['from']) for e in sched]}")
    return 0


OPERATOR_ERRORS = (ValueError, FileExistsError, FileNotFoundError, RuntimeError)


def main(argv=None) -> int:
    from nstrain import host
    host.apply()
    a = _parser().parse_args(argv)
    if a.cmd in ("new", "branch", "check", "status", "stop", "rollback") or (a.cmd == "run" and not a.foreground):
        try:
            return _dispatch(a)
        except OPERATOR_ERRORS as e:                                 # a refusal, not a bug: one line, exit 2
            print(f"nstrain {a.cmd}: {e}", file=sys.stderr)
            return 2
    return _dispatch(a)


def _dispatch(a) -> int:
    if a.cmd == "data":
        return _data(a)
    if a.cmd == "train":
        import signal
        from nstrain.proc import SUPERVISOR, die_with_parent
        die_with_parent(SUPERVISOR, signal.SIGTERM)                 # before the heavy imports
        from nstrain.trainer import main as train
        return train(a.run)
    if a.cmd == "run":
        if a.foreground:
            from nstrain.supervise import run_line
            return run_line(a.run)
        from nstrain import config
        from nstrain.supervise import start_background
        pid = start_background(a.run)
        d = config.load(a.run).run_dir
        print(f"supervisor started in the background (pid {pid}), independent of this login; `nstrain status {d}`, "
              f"log {d / 'run.log'}")
        return 0
    if a.cmd == "new":
        from nstrain import ops
        cfg = ops.new(a.run, a.model, a.data, a.val, a.name, a.peak_lr, a.warmup_tokens, a.wandb_project,
                      a.tokens_per_update, a.shape_from)
        print(f"{cfg.run_dir / 'run.json'}: line {cfg.name}, model {a.model}, data {cfg.data_work}")
        return 0
    if a.cmd == "branch":
        from nstrain import ops
        cfg = ops.branch(a.run, a.parent, a.at_cycle, a.plan, a.data, a.cycle_tokens, a.name, a.decay_shape,
                         a.final_lr_frac, **_kw(a))
        print(f"{cfg.run_dir / 'run.json'}: branch {cfg.name} from {cfg.fork_from}; data {cfg.data_work}; "
              f"schedule {cfg.schedule[-1]}")
        return 0
    if a.cmd == "check":
        from nstrain import ops
        problems = ops.check(a.run)
        for x in problems:
            print(f"problem: {x}")
        print("ok: the line can start" if not problems else f"{len(problems)} problems")
        return 0 if not problems else 1
    if a.cmd == "status":
        from nstrain.status import format_status, status
        s = status(a.run)
        print(json.dumps(s, indent=1) if a.json else format_status(s))
        return 0
    if a.cmd == "stop":
        from nstrain import ops
        print(ops.stop(a.run, a.now, a.at_cycle_end))
        return 0
    if a.cmd == "rollback":
        from nstrain import ops
        print(ops.rollback(a.run, a.to, a.force))
        return 0
    if a.cmd == "bench":
        import signal
        from nstrain.proc import TUNE, die_with_parent
        die_with_parent(TUNE, signal.SIGTERM)                       # a tune's measurement ends with the tune
        from nstrain import bench
        from nstrain import config
        model = config.load(a.run).model if a.run else a.model
        if a.profile:
            bench.profile(model, a.rows, a.seq_len, a.ac, a.top)
        else:
            bench.bench(model, a.rows, a.seq_len, a.accum, a.updates, a.ac, a.result, a.skip, a.shared, a.max_gib)
        return 0
    if a.cmd == "tune":
        import signal
        from nstrain.proc import SUPERVISOR, die_with_parent
        die_with_parent(SUPERVISOR, signal.SIGTERM)                 # a supervisor's tune ends with the supervisor
        from nstrain import bench
        bench.tune(a.run, a.margin_gib, a.updates, write=not a.dry_run, shared=a.shared)
        return 0
    if a.cmd == "presets":
        from nstrain import model
        for p in model.presets():
            print(p, "(run defaults)" if p in model.RUN_DEFAULTS else "")
        return 0
    raise AssertionError(a.cmd)


if __name__ == "__main__":
    sys.exit(main())
