"""Optimizer updates from the prepared cycles of a line (pre-tokenized uint32 streams, see nstrain.data.work).

An update is ``tokens_per_update / L`` samples of one sequence length L (each sample L + 1 tokens: inputs and the
targets shifted by one), cut into micro-batches of ``micro_rows`` rows.  Updates never span a cycle end or a change of
sequence length: the samples left over there (fewer than one update) are skipped, so every cycle end is an update
boundary (a checkpoint there is a clean fork point) and every update has one shape.

Progress is ``(cycle, sample)``: the next update starts at sample ``sample`` of cycle ``cycle``; with the manifest's
sha256 it is the complete data state (the order inside a cycle is fixed by the manifest and the seed).
"""
from __future__ import annotations

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
        said = False
        while not p.exists():
            if not wait:
                return None
            st = json.loads((self.root / "state.json").read_text())
            if st["total_tokens"] is not None and k * st["cycle_tokens"] >= st["total_tokens"]:
                return None                                          # the line's data is exhausted
            if st["cycles"].get(str(k), {}).get("status") in ("released", "releasing"):
                if not served(self.root):
                    raise RuntimeError(f"cycle {k} of {self.root} was released and no preparer serves the line to "
                                       "rebuild it: run the line with `nstrain run` (its preparer rebuilds it)")
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

    def mark(self, k: int) -> None:
        """Tell the preparer (nstrain.data.preparer) that cycle k is in training (the earlier ones are consumed)."""
        self._progress(k, False)

    def release(self, k: int) -> None:
        """Tell the preparer that cycle k is consumed."""
        self._progress(k, True)

    def updates(self, cycle: int, sample: int, expect_sha256: str = "") -> Iterator[Update]:
        k, i, want = cycle, sample, expect_sha256
        while True:
            man = self.manifest(k)
            if man is None:
                return
            cs = CycleStream(str(man), seed=self.seed, start=i, expect_sha256=want)
            want = ""                                                # only the resumed cycle is checked
            ups = plan(cs.seq_len, self.tpu)
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
                yield Update(k, a, b, n_up == len(ups) - 1, L, x, y, src, cs.sha256)
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


def val_rows(work: str, seq_len: int, rows: int) -> Dict[str, np.ndarray]:
    """The first ``rows`` samples of each source's held-out stream (``val/<source>/*.bin``), (rows, seq_len + 1)."""
    out = {}
    for d in sorted((Path(work) / "val").iterdir()):
        bins = sorted(d.glob("*.bin"))
        if not bins:
            continue
        a = np.memmap(bins[0], dtype=np.uint32, mode="r")
        n = min(rows, (a.size - 1) // seq_len)
        if n == 0:
            continue
        out[d.name] = np.stack([np.asarray(a[r * seq_len:r * seq_len + seq_len + 1], dtype=np.int64)
                                for r in range(n)])
    return out
