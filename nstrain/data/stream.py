"""Training-side reader of one prepared cycle.

The cycle's segments are read in order (stable / decay / long ...).  Within a segment every source contributes
``(quota - 1) // seq_len`` samples, interleaved by stratification: source s's n-th sample sits at (n + u_s) / c_s with
a seeded offset u_s, so any window of the cycle (e.g. one optimizer update) holds each source within two samples of its
share, and the order is fixed by (seed, cycle, segment).  A sample is ``seq_len + 1`` consecutive tokens of the
source's stream (documents joined by EOS); the next sample of that source starts ``seq_len`` tokens later (the usual
packing with one token of overlap for the target).

Resuming: ``CycleStream(manifest, start=i)`` continues at sample ``i``; each source's position is the number of its
samples before ``i`` times ``seq_len`` (no other state).  ``rank / world`` take every ``world``-th sample.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np


class _SourceReader:
    def __init__(self, pieces: List[dict]):
        self.pieces = pieces                                          # {bin, start, count, provenance...}
        self.cum = np.cumsum([0] + [p["count"] for p in pieces])
        self._mm: Dict[int, np.memmap] = {}

    def _map(self, i: int) -> np.memmap:
        if i not in self._mm:
            self._mm[i] = np.memmap(self.pieces[i]["bin"], dtype=np.uint32, mode="r")
        return self._mm[i]

    def read(self, pos: int, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.int64)
        got = 0
        i = int(np.searchsorted(self.cum, pos, side="right") - 1)
        while got < n:
            if i >= len(self.pieces):
                raise IndexError("read past the end of the source's quota")
            start, cnt = self.pieces[i]["start"], self.pieces[i]["count"]
            off = pos + got - int(self.cum[i])
            take = min(cnt - off, n - got)
            out[got:got + take] = self._map(i)[start + off:start + off + take]
            got += take
            i += 1
        return out


class CycleStream:
    def __init__(self, manifest: str, seed: int = 0, start: int = 0, rank: int = 0, world: int = 1,
                 expect_sha256: str = ""):
        text = Path(manifest).read_text()
        self.sha256 = hashlib.sha256(text.encode()).hexdigest()
        if expect_sha256 and expect_sha256 != self.sha256:
            raise RuntimeError(f"manifest {manifest} sha256 {self.sha256} != the checkpoint's {expect_sha256}")
        self.m = json.loads(text)
        self.names = sorted(self.m["sources"])
        self.readers = {s: _SourceReader(self.m["sources"][s]) for s in self.names}
        orders, lens = [], []
        for j, seg in enumerate(self.m["segments"]):
            L = seg["seq_len"]
            rng = np.random.default_rng([seed, self.m["cycle"], j])
            keys, ids = [], []
            for k, s in enumerate(self.names):
                c = max(0, seg["quota"].get(s, 0) - 1) // L            # the last sample's extra token stays inside
                if c:
                    # stratified interleaving: source s's samples at (n + u_s) / c, u_s a seeded offset in [0, 1),
                    # so every window of the cycle carries each source within two samples of its share
                    keys.append((np.arange(c) + rng.random()) / c)
                    ids.append(np.full(c, k, dtype=np.int16))
            key, ids = np.concatenate(keys), np.concatenate(ids)
            ids = ids[np.lexsort((ids, key))]
            orders.append(ids)
            lens.append(np.full(ids.size, L, dtype=np.int32))
        self.order = np.concatenate(orders)
        self.seq_len = np.concatenate(lens)
        self.start, self.rank, self.world = start, rank, world

    def __len__(self) -> int:
        return int(self.order.size)

    def positions(self, i: int) -> np.ndarray:
        """Each source's stream position before sample ``i``."""
        pos = np.zeros(len(self.names), dtype=np.int64)
        np.add.at(pos, self.order[:i], self.seq_len[:i])
        return pos

    def __iter__(self) -> Iterator[Tuple[np.ndarray, str, int]]:
        pos = self.positions(self.start)
        for i in range(self.start, self.order.size):
            k, L = int(self.order[i]), int(self.seq_len[i])
            if i % self.world == self.rank:
                yield self.readers[self.names[k]].read(int(pos[k]), L + 1), self.names[k], i
            pos[k] += L
