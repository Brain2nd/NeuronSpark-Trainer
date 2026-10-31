"""Optimizer updates from the prepared cycles of a line (pre-tokenized uint32 streams, see nstrain.data.work).

An update is ``tokens_per_update / L`` samples of one sequence length L (each sample L + 1 tokens: inputs and the
targets shifted by one), cut into micro-batches of ``micro_rows`` rows.  Updates never span a cycle end or a change of
sequence length: the samples left over there (fewer than one update) are skipped, so every cycle end is an update
boundary (a checkpoint there is a clean fork point) and every update has one shape.

Progress is ``(cycle, sample)``: the next update starts at sample ``sample`` of cycle ``cycle``; with the manifest's
sha256 it is the complete data state (the order inside a cycle is fixed by the manifest and the seed).
"""
from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from nstrain.data.stream import CycleStream

UNSERVED_S = 120.0       # a released cycle with no preparer serving the line this long: an error (shorter: a handover)


@dataclass
class Update:
    cycle: int
    first: int                     # first sample of the update in the cycle
    next: int                      # where the following update starts (== progress after this update)
    cycle_end: bool                # the last update of the cycle
    seq_len: int
    inputs: torch.Tensor           # (rows, L) int64, pinned
    targets: torch.Tensor          # (rows, L) int64, pinned
    sources: Dict[str, int]        # samples per source
    manifest_sha256: str
    recipe: Dict[str, float]       # the weights of the segment the update ends in (its sources validated before it)
    order_digest: str              # of the cycle's sample order (nstrain.data.stream.CycleStream.digest)
    mix: Dict[Tuple[str, str], Tuple[int, float]]   # (source, stratum) -> (its samples here, the planned number)


def plan(seq_len: np.ndarray, tokens_per_update: int) -> List[Tuple[int, int, int]]:
    """The updates of a cycle as (first, end, L): consecutive runs of ``tokens_per_update / L`` samples of one length;
    the remainder of each run of equal lengths (fewer than one update) is skipped."""
    out, i, n = [], 0, int(seq_len.size)
    while i < n:
        L = int(seq_len[i])
        if tokens_per_update % L:
            raise ValueError(f"tokens_per_update {tokens_per_update} is not a multiple of seq_len {L}")
        rows = tokens_per_update // L
        j = i
        while j < n and int(seq_len[j]) == L:
            j += 1
        while i + rows <= j:
            out.append((i, i + rows, L))
            i += rows
        i = j
    return out


def update_mix(cs: CycleStream, a: int, b: int) -> Dict[Tuple[str, str], Tuple[int, float]]:
    """(source, stratum) -> (its samples among samples a..b of the cycle, the number its planned share gives: in each
    segment the samples of its source there times the stratum's share of the source in that segment)."""
    lab, seg = np.asarray(cs.label[a:b]), np.asarray(cs.segment[a:b])
    got = np.bincount(lab, minlength=len(cs.labels))
    want: Dict[int, float] = {}
    for j in np.unique(seg).tolist():
        here = np.bincount(lab[seg == j], minlength=len(cs.labels))
        per_source: Dict[str, int] = {}
        for c, (s, _) in enumerate(cs.labels):
            per_source[s] = per_source.get(s, 0) + int(here[c])
        for c, (s, _) in enumerate(cs.labels):
            if (j, c) in cs.share:
                want[c] = want.get(c, 0.0) + per_source[s] * cs.share[(j, c)]
    return {cs.labels[c]: (int(got[c]), w) for c, w in want.items()}


class Line:
    """The cycles of one data line (an nstrain.data.work directory), consumed from a given progress."""

    def __init__(self, work: str, tokens_per_update: int, seed: int = 0, wait_s: float = 30.0):
        self.root = Path(work)
        self.tpu = int(tokens_per_update)
        self.seed = seed
        self.wait_s = wait_s

    def manifest(self, k: int, wait: bool = True) -> Optional[Path]:
        """Cycle k's manifest once the preparer made it visible; None at the end of the line (or at once with
        ``wait`` False).  A released cycle (a rollback into consumed data) is waited for while a preparer serves the
        line (it rebuilds the released cycles of its window, nstrain.data.preparer); without one it is an error."""
        from nstrain.data.preparer import served
        p = self.root / "cycles" / f"cycle_{k:05d}" / "manifest.json"
        said, unserved = False, None
        while not p.exists():
            if not wait:
                return None
            st = json.loads((self.root / "state.json").read_text())
            if st["total_tokens"] is not None and k * st["cycle_tokens"] >= st["total_tokens"]:
                return None                                          # the line's data is exhausted
            if st["cycles"].get(str(k), {}).get("status") in ("released", "releasing"):
                if served(self.root):
                    unserved = None
                else:                                                # (a preparer handing over or restarting:
                    unserved = unserved or time.time()               # a short gap)
                    if time.time() - unserved > UNSERVED_S:
                        raise RuntimeError(f"cycle {k} of {self.root} was released and no preparer has served the "
                                           f"line for {UNSERVED_S:.0f} s to rebuild it: run the line with "
                                           "`nstrain run` (its preparer rebuilds it)")
                if not said:
                    print(f"cycle {k} of {self.root} was released: waiting for the preparer to rebuild it",
                          flush=True)
                    said = True
            time.sleep(self.wait_s)                                  # the preparer is behind
        return p

    def _progress(self, k: int, done: bool) -> None:
        tmp = self.root / f"progress.json.tmp{os.getpid()}"
        tmp.write_text(json.dumps({"cycle": k, "done": done}))
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        tmp.replace(self.root / "progress.json")

    def consume(self, run_dir) -> None:
        """This trainer reads the data line alone while it runs (``consumer.lock``, held until the process ends): a
        second trainer at the same time -- another line on the same data -- would move progress.json and the
        preparer's window under the first.  Lines may use a data line one after another."""
        self._consumer = open(self.root / "consumer.lock", "a")
        try:
            fcntl.flock(self._consumer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another trainer reads the data line {self.root} now: {run_dir} starts after it has "
                               "ended, or needs its own data line (`nstrain data init` / `nstrain branch`)") from None

    def mark(self, k: int) -> None:
        """Tell the preparer (nstrain.data.preparer) that cycle k is in training (the earlier ones are consumed)."""
        self._progress(k, False)

    def release(self, k: int) -> None:
        """Tell the preparer that cycle k is consumed."""
        self._progress(k, True)

    def updates(self, cycle: int, sample: int, expect_sha256: str = "", expect_order: str = "") -> Iterator[Update]:
        k, i, want, order = cycle, sample, expect_sha256, expect_order
        while True:
            man = self.manifest(k)
            if man is None:
                return
            cs = CycleStream(str(man), seed=self.seed, start=i, expect_sha256=want)
            if order and order != cs.digest:
                raise RuntimeError(f"cycle {k}: the sample order differs from the checkpoint's ({cs.digest} != "
                                   f"{order}; another numpy or reader?): resuming would repeat and skip data")
            want = order = ""                                        # only the resumed cycle is checked
            ups = plan(cs.seq_len, self.tpu)
            if i == 0:
                taken = np.zeros(len(cs), dtype=bool)
                for a, b, _ in ups:
                    taken[a:b] = True
                print(f"cycle {k}: {len(cs)} samples in {len(ups)} updates; {int((~taken).sum())} samples "
                      f"({int(cs.seq_len[~taken].sum())} tokens) at the ends of runs of one length are skipped (fewer "
                      f"than one update), and {cs.tail} tokens at the ends of the sources' segment streams (less than "
                      "a window each)", flush=True)
            bounds = {a for a, _, _ in ups} | {b for _, b, _ in ups}    # an update's start, or the end of the one
            if i != 0 and i not in bounds:                              # before a skipped remainder
                raise RuntimeError(f"cycle {k}: progress {i} is not an update boundary")
            it = iter(cs)
            pos = i
            for n_up, (a, b, L) in enumerate(ups):
                if a < i:
                    continue
                while pos < a:                                       # skipped remainder before this update
                    next(it)
                    pos += 1
                rows = b - a
                x = torch.empty((rows, L), dtype=torch.int64).pin_memory()
                y = torch.empty((rows, L), dtype=torch.int64).pin_memory()
                src: Dict[str, int] = {}
                for r in range(rows):
                    s, name, idx = next(it)
                    assert idx == a + r
                    t = torch.from_numpy(s)
                    x[r].copy_(t[:-1])
                    y[r].copy_(t[1:])
                    src[name] = src.get(name, 0) + 1
                pos = b
                j = int(cs.segment[b - 1])
                yield Update(k, a, b, n_up == len(ups) - 1, L, x, y, src, cs.sha256,
                             cs.m["segments"][j]["weights"], cs.digest, update_mix(cs, a, b))
            k, i = k + 1, 0


class Prefetch:
    """``gen`` run in a background thread, ``depth`` updates ahead.  :meth:`get` waits at most ``timeout`` seconds:
    the next update, None at the end of the line, or :attr:`WAITING` (the preparer is behind); an error of the
    generator is re-raised."""

    WAITING = object()
    _END = object()

    def __init__(self, gen: Iterator[Update], depth: int = 2):
        self.q: Queue = Queue(maxsize=depth)
        self.ended = False

        def work():
            try:
                for u in gen:
                    self.q.put(u)
                self.q.put(Prefetch._END)
            except BaseException as e:                               # re-raised in the consumer
                self.q.put(e)

        threading.Thread(target=work, daemon=True).start()

    def get(self, timeout: float):
        if self.ended:
            return None
        try:
            u = self.q.get(timeout=timeout)
        except Empty:
            return Prefetch.WAITING
        if u is Prefetch._END:
            self.ended = True
            return None
        if isinstance(u, BaseException):
            raise u
        return u


def _rows_per_file(n: int, files: List[dict], seq_len: int, source: str, stratum: str) -> Dict[str, int]:
    """``n`` rows split over the files in proportion to their held-out tokens (largest remainder), none given more
    than it holds: a file's excess goes to the others, again in proportion."""
    from nstrain.data.hf import largest_remainder
    cap = {str(i): max(0, (int(f["tokens"]) - 1) // seq_len) for i, f in enumerate(files)}
    alloc, todo, left = {}, set(cap), n
    while True:
        a = largest_remainder(left, {k: float(files[int(k)]["tokens"]) for k in sorted(todo)})
        over = [k for k in sorted(todo) if a[k] > cap[k]]
        if not over:
            alloc.update(a)
            return alloc
        for k in over:
            alloc[k] = cap[k]
            left -= cap[k]
            todo.discard(k)
        if not todo:
            raise RuntimeError(f"{source}/{stratum or '(flat)'}: the held-out tokens hold {sum(cap.values())} rows "
                               f"of {seq_len}, fewer than {n}")


def val_rows(work: str, seq_len: int, rows: int, unit: int = 1) -> Dict[str, List[Tuple[str, float, np.ndarray]]]:
    """Each source's validation rows (``val/<source>/index.json``, nstrain.data.work.build_val): per stratum,
    (stratum, its share of the source, (n, seq_len + 1) rows), n = ``rows`` x the share rounded to a multiple of
    ``unit``, at least ``unit``; a stratum's rows split over its files by their held-out tokens (``_rows_per_file``),
    each file's rows the windows from the head of its stream (held-out documents, shuffled).  A source's loss is the
    share-weighted mean of its strata's, so every stratum counts at its share whatever its row count."""
    out: Dict[str, List[Tuple[str, float, np.ndarray]]] = {}
    root = Path(work) / "val"
    for d in sorted(root.iterdir()) if root.exists() else []:
        idx = d / "index.json"
        if not idx.exists():
            continue
        strata = []
        for t in json.loads(idx.read_text())["strata"]:
            n = max(unit, int(round(rows * t["share"] / unit)) * unit)
            files = t["files"]
            if not files:
                raise RuntimeError(f"{d.name}/{t['stratum'] or '(flat)'}: no held-out documents")
            alloc = _rows_per_file(n, files, seq_len, d.name, t["stratum"])
            got = []
            for i, f in enumerate(files):
                k = alloc[str(i)]
                if not k:
                    continue
                a = np.memmap(d / f["bin"], dtype=np.uint32, mode="r")
                if k * seq_len + 1 > a.size:
                    raise RuntimeError(f"{d / f['bin']}: {a.size} held-out tokens hold fewer than {k} rows of "
                                       f"{seq_len}")
                got += [np.asarray(a[r * seq_len:r * seq_len + seq_len + 1], dtype=np.int64) for r in range(k)]
            strata.append((t["stratum"], float(t["share"]), np.stack(got)))
        out[d.name] = strata
    return out
