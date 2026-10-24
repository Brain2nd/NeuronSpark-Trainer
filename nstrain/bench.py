"""Throughput of the captured training step on random tokens.

``nstrain bench``: one shape (micro-batch rows x sequence length, accumulation, activation recomputation): median
update time, tokens per second, peak memory; ``--profile``: the kernel-time breakdown of the step bodies (run eagerly:
the profiler does not see graph-replayed kernels).

``nstrain tune RUN``: the fastest shape of every sequence length of a line that fits the GPU, written into its
run.json before the line's first update (micro_rows is fixed for the life of a line).  The memory spent is turned
into speed: recomputing activations costs time, so each length gets as little recomputation as its micro-batch needs.
Memory is linear in the micro-batch rows and in the number k of dendritic layers recomputed:
``peak(rows, k) = fixed(k) + rows * per_row(k)``, both linear in k; two short probes (1 and 2 rows) at k = 0 (none)
and at k = all dendritic layers give the two lines (at a length where no row fits without recomputation, k = half of
them replaces k = 0).  For each micro-batch size the least k that fits follows; the largest size that needs none and
every larger size with its least k are measured directly (a candidate out of memory or over the limit gets one more
layer recomputed; any other failure stops the tune, a defect of the code at that shape).  Recomputing every layer
("all") is the fallback when no k fits.  The fastest candidate is chosen.
Every probe and measurement runs in its own process (an out-of-memory leaves nothing behind); a shape fits when its
peak reserved memory plus the CUDA context stays ``margin_gib`` below the GPU's memory.  What is maximised is the
training throughput: tokens per second of complete optimizer updates of the line's ``tokens_per_update`` (every
micro-batch's forward and backward, the clip, AdamW), the work of training.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional

import torch

from nstrain import config as C
from nstrain import gpu
from nstrain import model as M
from nstrain.engine import StepEngine
from nstrain.flat_adamw import FlatAdamW
from nstrain.proc import SUPERVISOR, TUNE


_LEASE = []                                                          # held for the life of the process


def setting(ac):
    """A recomputation setting of run.json from a command-line value: none, dendritic, all, or a number of dendritic
    layers."""
    if isinstance(ac, (bool, int)):
        return ac
    if str(ac).isdigit():
        return int(ac)
    named = {"none": False, "dendritic": True, "all": "all"}
    if ac not in named:
        raise ValueError(f"recomputation {ac!r}: none, dendritic, all or a number of dendritic layers")
    return named[ac]


def _setup(model_spec, rows: int, seq_len: int, ac: str, shared: bool = False, max_gib: float = None):
    """The model on the GPU, which it must have to itself (the lease, then no other compute process), or beside other
    processes (``shared``: the lease only); ``max_gib``: PyTorch's memory in this process is capped there."""
    if not _LEASE:
        _LEASE.append(gpu.lease("bench"))
        others = gpu.other_processes(exclude_pid=os.getpid())
        if others and not shared:
            raise gpu.GpuBusy(f"other processes on the GPU: {others}")
    if max_gib is not None:
        gpu.cap_memory(max_gib)
    from neuronspark.utils.calibration import CALIBRATION_ROWS
    dev = torch.device("cuda")
    cfg = model_spec if isinstance(model_spec, dict) else M.model_config(model_spec)
    torch.manual_seed(0)
    model = M.build(cfg, dev, init="launch")
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e9:.3f} B", flush=True)
    g = torch.Generator(device="cpu").manual_seed(1)
    cal_rows, cal_len = min(rows, 8), min(seq_len, 4096)            # a run's launch calibration reads 4K rows
    cal = torch.randint(0, cfg["vocab_size"], (cal_rows * max(1, CALIBRATION_ROWS // cal_rows), cal_len), generator=g)
    M.calibrate(model, cal, cal_rows, dev)
    M.recompute(model, setting(ac))
    opt = FlatAdamW(model, M.param_groups(model))
    return cfg, model, opt, g


def bench(model_spec, rows: int, seq_len: int, accum: int, updates: int = 12, ac: str = "none",
          result: bool = False, skip: int = 5, shared: bool = False, max_gib: float = None) -> float:
    """``model_spec``: a preset name, a model-config file or a model-config dict; the median over the updates after
    the first ``skip``; ``result``: also print the measurement as one ``BENCH_RESULT {json}`` line (for ``tune``);
    ``shared``: beside other processes on the GPU; ``max_gib``: out of memory past this much PyTorch memory."""
    cfg, model, opt, g = _setup(model_spec, rows, seq_len, ac, shared, max_gib)
    t0 = time.time()
    eng = StepEngine(model, opt, rows, seq_len, accum)
    print(f"captured in {time.time() - t0:.1f} s; allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB, "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    n = rows * accum
    batches = []
    for _ in range(4):
        w = torch.randint(0, cfg["vocab_size"], (n, seq_len + 1), generator=g)
        batches.append((w[:, :-1].contiguous().pin_memory(), w[:, 1:].contiguous().pin_memory()))
    times = []
    for u in range(updates):
        x, y = batches[u % 4]
        torch.cuda.synchronize()
        t = time.perf_counter()
        m, ok = eng.update(x, y, 3e-4)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
        if u < 3 or u % 10 == 9:
            print(u, m, ok, f"{times[-1] * 1e3:.1f} ms", flush=True)
    st = sorted(times[min(skip, len(times) - 1):])
    med = st[len(st) // 2]
    tps = n * seq_len / med
    print(f"update {accum} x {rows} x {seq_len} (recompute {ac}): median {med * 1e3:.1f} ms -> {tps:,.0f} tok/s; "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    if result:
        print("BENCH_RESULT " + json.dumps({"tok_s": tps,
                                            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                                            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}),
              flush=True)
    return tps


def profile(model_spec: str, rows: int, seq_len: int, ac: str = "none", top: int = 60, n: int = 3) -> None:
    cfg, model, opt, g = _setup(model_spec, rows, seq_len, ac)
    eng = StepEngine(model, opt, rows, seq_len, 1)
    w = torch.randint(0, cfg["vocab_size"], (rows, seq_len + 1), generator=g)
    x, y = w[:, :-1].contiguous().pin_memory(), w[:, 1:].contiguous().pin_memory()
    for _ in range(5):
        eng.update(x, y, 3e-4)
    torch.cuda.synchronize()
    eng.x.copy_(x)
    eng.y.copy_(y)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]):   # the first session can be empty
        eng._micro()
        torch.cuda.synchronize()
    eng.opt.zero_grad()
    eng._zero_sums()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            eng._micro()
            eng._norm()
            eng._step()
        torch.cuda.synchronize()
    tot, cnt, span = defaultdict(float), defaultdict(int), [None, None]
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            tot[e.name] += e.time_range.elapsed_us()
            cnt[e.name] += 1
            s, t = e.time_range.start, e.time_range.end
            span[0] = s if span[0] is None else min(span[0], s)
            span[1] = t if span[1] is None else max(span[1], t)
    print(f"kernels per update {sum(cnt.values()) / n:.0f}; kernel time {sum(tot.values()) / n / 1e3:.1f} ms; "
          f"wall span {(span[1] - span[0]) / n / 1e3:.1f} ms")
    cat = defaultdict(float)
    for name, t in tot.items():
        k = ("torch elementwise/reduce" if name.startswith(("void at::native", "void at_cuda_detail")) else
             "gemm" if name.startswith(("nvjet", "void cublasLt", "std::enable_if<true, void>::type internal::gemvx",
                                        "sm90", "sm100", "cutlass")) else
             "memcpy" if name.startswith("Memcpy") else "repo kernels (triton / gluon)")
        cat[k] += t
    for k, t in sorted(cat.items(), key=lambda kv: -kv[1]):
        print(f"  {k:32s} {t / n / 1e3:8.2f} ms")
    for name, t in sorted(tot.items(), key=lambda kv: -kv[1])[:top]:
        print(f"{t / n / 1e3:8.2f} ms {cnt[name] / n:6.0f}x  {name[:150]}")


# ---------------------------------------------------------------------- tune
def _setting(k, n: int):
    """The run.json value of k dendritic layers (or "all") recomputed."""
    return "all" if k == "all" else (False if k == 0 else (True if k >= n else int(k)))


def _arg(k, n: int) -> str:
    """The ``nstrain bench --ac`` value of a setting."""
    s = _setting(k, n)
    if s is False:
        return "none"
    if s is True:
        return "dendritic"
    return str(s)


def _name(k, n: int) -> str:
    return "every layer" if k == "all" else ("none" if k == 0 else f"{k} of {n} dendritic layers")


CONTEXT_GIB = gpu.CONTEXT_GIB


def _gpu_total_gib() -> float:
    out = subprocess.run(["nvidia-smi", "-i", gpu.device(), "--query-gpu=memory.total",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True, timeout=60).stdout
    return float(out.strip().splitlines()[0]) / 1024


OUT_OF_MEMORY = ("out of memory", "CUBLAS_STATUS_ALLOC_FAILED")


def _measure(run_dir, seq_len: int, rows: int, accum: int, ac: str, updates: int, skip: int = 1,
             max_gib: float = None) -> Optional[dict]:
    """One candidate in its own process: its BENCH_RESULT, or None when it does not fit the GPU's memory.  Any other
    failure is a defect of the code at that shape and raises: a shape the memory allows must run.  ``max_gib``: beside
    other processes, with PyTorch's memory capped there (a shape over it runs out of memory in its own process)."""
    p = subprocess.run([sys.executable, "-m", "nstrain", "bench", "--run", str(run_dir), "--seq-len", str(seq_len),
                        "--rows", str(rows), "--accum", str(accum), "--ac", ac, "--updates", str(updates), "--skip",
                        str(skip), "--result"] + (["--shared", "--max-gib", f"{max_gib:.3f}"] if max_gib else []),
                       capture_output=True, text=True, timeout=7200,
                       env={k: v for k, v in dict(os.environ, **{TUNE: str(os.getpid())}).items() if k != SUPERVISOR})
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("BENCH_RESULT "):
            return json.loads(line[len("BENCH_RESULT "):])
    out = p.stdout + p.stderr
    if "GpuBusy" in out or "leased by" in out:
        raise RuntimeError("another process uses the GPU: tune measures on a GPU nothing else uses")
    if any(s.lower() in out.lower() for s in OUT_OF_MEMORY):
        print(f"    out of memory (exit {p.returncode})", flush=True)
        return None
    why = [x for x in out.splitlines() if "Error" in x or "error:" in x] or out.strip().splitlines()[-1:]
    raise RuntimeError(f"{rows} rows x {seq_len}, recompute {ac}: the measurement failed (exit {p.returncode}): "
                       f"{why[-1][:300] if why else 'no output'}")


def tune(run, margin_gib: float = gpu.TUNE_MARGIN_GIB, updates: int = 4, write: bool = True,
         shared: bool = False) -> Dict:
    """The fastest fitting (micro_rows, recomputation) of every sequence length of the line (see the module
    documentation); written into run.json when ``write``.  ``shared``: the shapes beside other processes, which leave
    ``guards.share_reserve_gb`` of the GPU's memory to them (measured beside whatever runs there), written as the
    lengths' ``shared`` shapes (also on a line with checkpoints)."""
    from nstrain.ckpt import Checkpoints
    cfg = C.load(run)
    if gpu.lease_holder() is not None:
        raise RuntimeError("another nstrain process holds the GPU")
    if shared:
        reserve = cfg.guards.share_reserve_gb
        if reserve is None:
            raise RuntimeError(f"{cfg.name}: guards.share_reserve_gb is not set")
        limit = gpu.beside_limit_gib(_gpu_total_gib(), reserve, margin_gib)
        print(f"GPU {gpu.device()}: shapes beside other processes fit below {limit:.1f} GiB reserved ({reserve:g} GB "
              f"left to them, margin {margin_gib:g} GiB)", flush=True)
    else:
        if write and Checkpoints(cfg.ckpt_dir).complete():
            raise RuntimeError(f"{cfg.name} has checkpoints: micro_rows is fixed for the life of the line")
        if gpu.other_processes():
            raise RuntimeError("another process uses the GPU: tune measures on a GPU nothing else uses")
        limit = _gpu_total_gib() - margin_gib - CONTEXT_GIB
        print(f"GPU {gpu.device()}: shapes fit below {limit:.1f} GiB reserved (margin {margin_gib:g} GiB)", flush=True)
    tpu, chosen = cfg.tokens_per_update, {}
    cap = limit if shared else None                                  # beside others: out of memory at the limit
    for L in sorted(cfg.lengths):
        rows = [r for r in (2 ** i for i in range(16)) if r * L <= tpu and tpu % (r * L) == 0]
        if not rows:
            raise ValueError(f"length {L}: no micro-batch divides the {tpu} tokens of an update")
        n = M.dendritic_layers(cfg.model)
        probes = {}

        def peak(k, r):
            """Peak reserved GiB of r rows with setting k (an int, or "all"); None when it does not run."""
            if (k, r) not in probes:
                print(f"  {L}: probe {r} rows, recompute {_name(k, n)}", flush=True)
                m = _measure(cfg.run_dir, L, r, 1, _arg(k, n), 2, max_gib=cap) if r in rows else None
                probes[(k, r)] = None if m is None else m["peak_reserved_gib"]
                if m is not None:
                    print(f"    peak reserved {m['peak_reserved_gib']:.1f} GiB", flush=True)
            return probes[(k, r)]

        def line(k):
            a, b = peak(k, 1), peak(k, 2) if peak(k, 1) is not None else None
            return None if a is None or b is None else (2 * a - b, b - a)

        cands = []
        hi = line(n)
        lo_k = 0 if line(0) is not None else n // 2
        lo = line(lo_k) if lo_k != n else None
        if hi is not None and lo is not None:
            def need(r):
                for k in range(0, n + 1):
                    t = (k - lo_k) / (n - lo_k)
                    if lo[0] + (hi[0] - lo[0]) * t + r * (lo[1] + (hi[1] - lo[1]) * t) <= limit:
                        return k
                return None
            ks = {r: need(r) for r in rows}
            print(f"  {L}: least dendritic layers recomputed per micro-batch rows: "
                  + ", ".join(f"{r}: {k}" for r, k in ks.items() if k is not None), flush=True)
            free = [r for r in rows if ks[r] == 0]
            if free:
                cands.append((max(free), 0))
            cands += [(r, ks[r]) for r in rows if ks[r] is not None and ks[r] > 0 and (not free or r > max(free))]
        elif hi is not None:
            fit = [r for r in rows if hi[0] + r * hi[1] <= limit]
            if fit:
                cands.append((max(fit), n))
        if not cands:
            a = line("all")
            if a is not None:
                cands.append((max([r for r in rows if a[0] + r * a[1] <= limit] or [1]), "all"))
            elif peak("all", 1) is not None and peak("all", 1) <= limit:
                cands.append((1, "all"))
        if not cands:
            raise RuntimeError(f"length {L}: no candidate fits the GPU")
        found = []
        for r, k in cands:
            while True:
                print(f"  {L}: measure {r} rows, recompute {_name(k, n)}", flush=True)
                m = _measure(cfg.run_dir, L, r, tpu // (r * L), _arg(k, n), updates,    # a complete update
                             max_gib=cap)
                if m is not None and m["peak_reserved_gib"] <= limit:
                    print(f"    {m['tok_s']:,.0f} tok/s, peak reserved {m['peak_reserved_gib']:.1f} GiB", flush=True)
                    found.append((r, k, m))
                    break
                if m is not None:
                    print(f"    peak reserved {m['peak_reserved_gib']:.1f} GiB is over the limit", flush=True)
                if k == "all" or k >= n:
                    break
                k += 1                                               # one more layer recomputed
        if not found:
            raise RuntimeError(f"length {L}: no candidate fits the GPU")
        best = max(found, key=lambda f: f[2]["tok_s"])
        chosen[L] = (best[0], _setting(best[1], n), best[2])
        print(f"length {L}: {best[0]} rows, recompute {_name(best[1], n)}: {best[2]['tok_s']:,.0f} tok/s, peak "
              f"reserved {best[2]['peak_reserved_gib']:.1f} GiB", flush=True)
    if write:
        now = C.load(run)                                            # run.json as it is now (edited meanwhile?)
        if (now.model, now.tokens_per_update, sorted(now.lengths)) != (cfg.model, cfg.tokens_per_update,
                                                                       sorted(cfg.lengths)):
            raise RuntimeError(f"{now.run_dir / C.RUN_FILE} changed its model, tokens per update or lengths while "
                               "the shapes were measured; nothing written")
        for L, (r, ac, _) in chosen.items():
            if shared:
                now.lengths[L].shared = C.SharedShape(micro_rows=r, activation_checkpointing=ac,
                                                      reserve_gb=cfg.guards.share_reserve_gb)
            else:
                now.lengths[L] = C.Length(micro_rows=r, activation_checkpointing=ac, shared=now.lengths[L].shared)
        C.save(now)
        print(f"{now.run_dir / C.RUN_FILE}: lengths {now.to_dict()['lengths']}", flush=True)
    return {L: {"micro_rows": r, "recompute": ac, **m} for L, (r, ac, m) in chosen.items()}
