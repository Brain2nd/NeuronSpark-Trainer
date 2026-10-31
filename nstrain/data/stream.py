"""Training-side reader of one prepared cycle.

The cycle's segments are read in order (stable / decay / long ...).  Within a segment every source contributes
``(quota - 1) // seq_len`` samples, and the mix is uniform at every level:

* between sources: stratified interleaving -- source s's n-th sample sits at (n + u_s) / c_s with a seeded offset u_s,
  so any window of the cycle (e.g. one optimizer update) holds each source within two samples of its share;
* inside a source: the segment's stream of the source (its pieces of that segment joined -- drawn from every stratum
  and from several files of each, nstrain.data.work; documents shuffled inside each file, nstrain.data.tokenize) is
  cut into windows of ``seq_len + 1`` tokens at multiples of ``seq_len`` (the usual packing with one token of overlap
  for the target), and the source's samples take these windows in a seeded random order, so every update draws each
  source's samples uniformly from the whole segment instead of from one file after another.

The order is fixed by (seed, cycle, segment).  Resuming: ``CycleStream(manifest, start=i)`` continues at sample ``i``;
the window of a sample is a function of its index (no other state).  ``rank / world`` take every ``world``-th sample.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np

from nstrain.data.work import FORMAT
from nstrain.data.work import ORDER as ORDER_FORMAT     # (pieces per segment, windows in a random order)


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
        if self.m.get("order") != ORDER_FORMAT:
            raise RuntimeError(f"manifest {manifest}: sample order {self.m.get('order', 1)}, this reader reads order "
                               f"{ORDER_FORMAT} (pieces per segment, windows in a random order)")
        if self.m.get("format") != FORMAT:
            raise RuntimeError(f"manifest {manifest}: data line format {self.m.get('format', 1)}, this reader reads "
                               f"format {FORMAT} (its streams are of another encoding)")
        self.names = sorted(self.m["sources"])
        nseg = len(self.m["segments"])
        self.readers = {(j, s): _SourceReader([p for p in self.m["sources"][s] if p["segment"] == j])
                        for j in range(nseg) for s in self.names}
        orders, lens, wins, segs = [], [], [], []
        self.tail = 0                                                # quota tokens past a source's last window
        for j, seg in enumerate(self.m["segments"]):
            L = seg["seq_len"]
            rng = np.random.default_rng([seed, self.m["cycle"], j])
            keys, ids, win = [], [], []
            for k, s in enumerate(self.names):
                c = max(0, seg["quota"].get(s, 0) - 1) // L            # the last sample's extra token stays inside
                self.tail += max(0, seg["quota"].get(s, 0) - c * L - 1)
                if c:
                    # stratified interleaving: source s's samples at (n + u_s) / c, u_s a seeded offset in [0, 1),
                    # so every window of the cycle carries each source within two samples of its share
                    keys.append((np.arange(c) + rng.random()) / c)
                    ids.append(np.full(c, k, dtype=np.int16))
                    # the n-th sample of source s reads window perm[n] of the segment: a uniform draw without
                    # replacement over all of the segment's windows of that source
                    perm = np.random.default_rng([seed, self.m["cycle"], j, k]).permutation(c)
                    win.append(perm.astype(np.int64) * L)
            if not keys:
                continue
            key, ids, win = np.concatenate(keys), np.concatenate(ids), np.concatenate(win)
            o = np.lexsort((ids, key))                                 # a source's samples keep their order n
            orders.append(ids[o])
            wins.append(win[o])
            lens.append(np.full(o.size, L, dtype=np.int32))
            segs.append(np.full(o.size, j, dtype=np.int16))
        self.order = np.concatenate(orders) if orders else np.zeros(0, dtype=np.int16)
        self.window = np.concatenate(wins) if wins else np.zeros(0, dtype=np.int64)
        self.seq_len = np.concatenate(lens) if lens else np.zeros(0, dtype=np.int32)
        self.segment = np.concatenate(segs) if segs else np.zeros(0, dtype=np.int16)
        self.start, self.rank, self.world = start, rank, world
        self._label()

    def _label(self) -> None:
        """Every sample's stratum (``label``: an index into ``labels``, (source, stratum) pairs) and each label's
        planned share of its source's samples in its segment (``share``, by tokens) -- what the trainer compares the
        samples it reads with (nstrain.trainer: a drift of the mix shows long before it costs a line)."""
        names: Dict[Tuple[str, str], int] = {}
        self.label = np.zeros(self.order.size, dtype=np.int32)
        share: Dict[Tuple[int, int], float] = {}                       # (segment, label) -> share
        for (j, s), r in self.readers.items():
            if not r.pieces:
                continue
            sel = np.nonzero((self.segment == j) & (self.order == self.names.index(s)))[0]
            codes = np.array([names.setdefault((s, p["stratum"]), len(names)) for p in r.pieces], dtype=np.int32)
            self.label[sel] = codes[np.searchsorted(r.cum, self.window[sel], side="right") - 1]
            total = float(r.cum[-1])
            for p, c in zip(r.pieces, codes):
                share[(j, int(c))] = share.get((j, int(c)), 0.0) + p["count"] / total
        self.labels = [n for n, _ in sorted(names.items(), key=lambda kv: kv[1])]
        self.share = share

    def __len__(self) -> int:
        return int(self.order.size)

    @property
    def digest(self) -> str:
        """blake2b of the sample order (sources, windows, segments): a checkpoint records it, a resume compares it."""
        if not hasattr(self, "_digest"):
            h = hashlib.blake2b(digest_size=16)
            for a in (self.order, self.window, self.segment, self.seq_len):
                h.update(np.ascontiguousarray(a).tobytes())
            self._digest = h.hexdigest()
        return self._digest

    def __iter__(self) -> Iterator[Tuple[np.ndarray, str, int]]:
        for i in range(self.start, self.order.size):
            if i % self.world == self.rank:
                k, L, j = int(self.order[i]), int(self.seq_len[i]), int(self.segment[i])
                yield self.readers[(j, self.names[k])].read(int(self.window[i]), L + 1), self.names[k], i
