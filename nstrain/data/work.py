"""Cycles: the budget is consumed in fixed-size cycles; each cycle is downloaded, tokenized, trained on, and deleted.

Work directory layout::

    state.json                 tokenizer (+ sha256, library version), pinned dataset revisions, budget, cycle size,
                               per-source and per-stratum cursors, calibration, the open files (pool), cycle status
    listings/*.json            file listings of the sources at their pinned revisions
    staging/<source>/<stream>/ downloads in flight (deleted after tokenization)
    cycles/cycle_NNNNN/        <source>/<file>.bin|.json token streams + manifest.json (the training input)
    manifests/cycle_NNNNN.json permanent copies of the manifests (kept after the cycle is released)
    manifests/cycle_NNNNN.state.json  the state right after cycle N was prepared (cursors, pool): forks start here
    pool/<source>/             streams of the open files (their unread rest is in the next cycles; hard links)
    val/<source>/              held-out streams (several files per source, never trained on)
    progress.json              written by the trainer: {"cycle": k, "done": true} when cycle k is consumed
    lock                       held by every writer of state.json (one at a time; see ``Work.lock``)

What a cycle holds of a source -- representative of the whole source, not of whichever files come next:

* strata: a source's files are grouped by subdirectory (languages, crawl dumps, L3 sub-types ...; or each file is
  one, ``hf.stratum_of``); each segment's quota of the source is split over its strata in proportion to their tokens
  (their file bytes over their bytes per token, measured, ``bpt``), so every cycle and segment holds every stratum at
  its share;
* open files: each stratum keeps ``open_files`` x its share (at least one) files open -- fewer while they would hold
  more than ``pool_cycles`` cycles' worth of the source (disk), never fewer than one.  A cycle reads its tokens of a
  stratum from the stratum's open files and as many next files as it needs: all of each file but the n_open with the
  most unread tokens, and the rest from those in proportion to their streams' sizes (``_plan_files``), so after a
  cycle at most n_open files of a stratum are open (their unread rest in ``pool/`` for the next cycles; a file left
  open keeps its whole stream on disk).  More can be open: after a cycle whose bytes-per-token estimate of the
  stratum was low (it opened files it did not need), or on a branch whose n_open is lower than its parent's; a cycle
  then reads the files with the least unread tokens first, whole, and leaves the rest untouched for the next cycles
  (not spread over all of them: that would keep every one open).  Each segment takes its part from those files in
  proportion to what the cycle reads of them, each part cut at a document end (but the last);
* inside a file the documents are in a seeded random order (nstrain.data.tokenize), so the part of a file a cycle takes
  is a uniform sample of the file's documents; the trainer reads each source's windows of a segment in a random order
  (nstrain.data.stream);
* held-out documents (a hash of their text, ``Source.holdout`` of them) are in no training stream; ``build_val`` keeps
  those of each stratum's first files for validation;
* every token once: a stratum whose files are used up is an error, not a second pass (``max_epochs`` 1).

Lines: a work directory is one line -- the trunk (schedule ``stable`` from token 0, no total: extendable until the
corpora are exhausted) or a branch forked from a trunk cycle with its own schedule and total (``fork``).  Schedules are
append-only (``append``): an entry may only start at or after the end of the last prepared cycle.

Crash safety of ``prepare``: the files a cycle fetches are recorded in ``state.json`` (``inflight``) together with the
advanced cursors, so a restarted ``prepare`` of the same cycle takes the same files (a finished stream is reused); the
commit order is the permanent manifest, the state snapshot, the state (the commit point), then ``manifest.json`` in
the cycle directory -- the trainer's signal -- written atomically; ``ensure_visible`` restores that last file from
the permanent copy after a crash between the commit and it.

Determinism: every input is pinned -- dataset revisions (commit sha), the strata and the file order of each stratum
(fixed shuffle per (source, stratum, epoch)), the document order inside each file
(``doc_seed`` from the line's seed, source, path and epoch), the tokenizer file (sha256) and library version.  A
cycle's manifest lists, per source, pieces {bin, start, count, stratum} with their provenance (repo, path, revision,
epoch, doc_seed, tokens, blake2b of the whole stream); ``rebuild`` re-creates a released cycle from its permanent
manifest and checks every blake2b.
"""
from __future__ import annotations

import concurrent.futures as cf
import contextlib
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from nstrain.data import hf, mixture
from nstrain.data.sources import SOURCES
from nstrain.data.tokenize import TEXT, text_mode

PROV_KEYS = ("repo", "path", "revision", "epoch", "doc_seed", "tokens", "blake2b")
ORDER = 2                        # the sample order a manifest is read in (nstrain.data.stream.ORDER_FORMAT)
FORMAT = 4                       # state.json and manifest layout (strata, pool, document order, held-out documents,
                                 # documents as plain text but a source's control markers: nstrain.data.tokenize)
HOLDOUT_SALT = "nstrain-holdout-v1"   # with each source's share: the held-out documents of every line (tokenize.held)
VAL_FILES = 16                   # files a stratum's held-out documents come from (at most)
VAL_KEEP_TOKENS = 4 << 20        # held-out tokens kept per validation file (documents shuffled)
TOKENIZE_TIMEOUT_S = 6 * 3600    # one file (the largest take minutes): longer is a hang


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _write_atomic(path: Path, text: str) -> None:
    """``text`` to ``path`` through a temporary file, fsynced, renamed, the directory fsynced."""
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")              # writers in several processes never share it
    tmp.write_text(text)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _temporaries(d: Path) -> List[Path]:
    """What a crashed or aborted writer left under ``d``: a stream or its meta not finished (nstrain.data.tokenize)."""
    return [f for pat in ("*.bin.part*", "*.bin.raw*", "*.json.tmp*") for f in d.rglob(pat)]


def _blake2b_file(path: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _tokenize_subprocess(job: dict, rayon: int, stop=None) -> dict:
    """Tokenize one file in its own interpreter: independent of the caller's ``__main__`` (unlike multiprocessing
    spawn / forkserver) and of the caller's threads (unlike fork); a failure is an exception here, never a hang; the
    tokenizer dies with this process (nstrain.proc), and is killed when ``stop`` (a threading.Event) is set."""
    import tempfile
    root = str(Path(__file__).resolve().parents[2])                   # the directory holding the nstrain package
    env = dict(os.environ, RAYON_NUM_THREADS=str(rayon), TOKENIZERS_PARALLELISM="true",
               NSTRAIN_PREPARER_PID=str(os.getpid()),
               PYTHONPATH=os.pathsep.join(x for x in (root, os.environ.get("PYTHONPATH", "")) if x))
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        p = subprocess.Popen([sys.executable, "-m", "nstrain.data.tokenize", json.dumps(job)], env=env, stdout=out,
                             stderr=err, text=True)
        end = time.time() + TOKENIZE_TIMEOUT_S
        try:
            while True:
                try:
                    p.wait(timeout=1.0)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if stop is not None and stop.is_set():
                    raise RuntimeError(f"tokenizing {job['path']} stopped")
                if time.time() > end:
                    raise RuntimeError(f"tokenizing {job['path']} took more than {TOKENIZE_TIMEOUT_S} s (hung?)")
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()
        out.seek(0)
        err.seek(0)
        if p.returncode != 0:
            raise RuntimeError(f"tokenizing {job['path']} failed (exit {p.returncode}):\n{err.read()[-4000:]}")
        return json.loads(out.read().strip().splitlines()[-1])


def _split(total: int, room: List[int], weight: List[float]) -> List[int]:
    """``total`` tokens taken from files with ``room`` tokens left, in proportion to ``weight`` (each file's whole
    stream: every document of the open files equally likely); a file with too little room gives all it has and the
    others share the rest in proportion; integers summing to ``total``."""
    out, left, todo = [0] * len(room), int(total), list(range(len(room)))
    while todo and left > 0:
        w = sum(weight[i] for i in todo)
        full = [i for i in todo if room[i] - out[i] <= left * weight[i] / w]
        if not full:
            break
        for i in full:                                               # files that cannot take their share: all of it
            left -= room[i] - out[i]
            out[i] = room[i]
        todo = [i for i in todo if i not in full]
    if todo and left > 0:
        w = sum(weight[i] for i in todo)
        exact = {i: left * weight[i] / w for i in todo}
        for i in todo:
            out[i] += int(exact[i])
        rest = left - sum(int(exact[i]) for i in todo)
        for i in sorted(todo, key=lambda i: (int(exact[i]) - exact[i], i))[:rest]:
            out[i] += 1
        left = 0
    if left:
        raise RuntimeError(f"{total} tokens asked of open files holding {sum(room)}")
    return out


def _plan_files(total: int, room: List[int], weight: List[float], keep: int) -> List[int]:
    """What a cycle reads of each open file of a stratum (``total`` tokens): all of every file but the ``keep`` with
    the most room (the files earlier cycles read from have the least), and the rest from those ``keep`` in
    proportion to their streams (``_split``) -- so at most ``keep`` files stay open, and a file left open keeps its
    whole stream on disk for one more cycle.  Ties go to the earlier file.  When the other files hold more than
    ``total`` (more files are open than ``keep``: an estimate was low, or a branch's n_open is below its parent's),
    the files are read whole in the order of their room and the last one read in part."""
    order = sorted(range(len(room)), key=lambda i: (room[i], -i))
    done, kept = order[:max(0, len(order) - keep)], order[max(0, len(order) - keep):]
    out = [0] * len(room)
    full = sum(room[i] for i in done)
    if full <= total:
        for i in done:
            out[i] = room[i]
        for i, x in zip(kept, _split(total - full, [room[i] for i in kept], [weight[i] for i in kept])):
            out[i] = x
        return out
    left = int(total)
    for i in order:
        out[i] = min(room[i], left)
        left -= out[i]
        if not left:
            break
    return out


def _doc_end(bin_path: str, pos: int, limit: int, eos: int, reach: int = 1 << 16) -> int:
    """The end (one past the EOS) of the document running at stream position ``pos - 1``, if it ends before
    ``limit`` and within ``reach`` tokens; else ``pos`` (a cut inside a document)."""
    import numpy as np                                               # (not at import: the supervisor imports this)
    mm = np.memmap(bin_path, dtype=np.uint32, mode="r")
    a, b = max(pos - 1, 0), min(limit, pos - 1 + reach)
    hit = np.nonzero(np.asarray(mm[a:b]) == eos)[0] if b > a else []
    return int(a + hit[0] + 1) if len(hit) else pos


class NeedsSpace(RuntimeError):
    """The disk cannot take the files a cycle needs now (the preparer waits)."""


class Work:
    def __init__(self, root: str):
        self.root = Path(root).resolve()                              # every stream path in the state is absolute
        self.state_path = self.root / "state.json"
        self._by: Dict[str, Dict[str, List[Dict]]] = {}

    # ---------------------------------------------------------------- state
    @contextlib.contextmanager
    def lock(self):
        """The work directory's writer lock: state.json has one writer at a time (prepare, release, rebuild, append,
        fork, calibrate, val and each pass of the preparer loop (nstrain.data.preparer) hold it; a second writer
        waits for it)."""
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / "lock", "a") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f"{self.root}: waiting for the lock of another writer", flush=True)
                fcntl.flock(f, fcntl.LOCK_EX)
            yield

    def in_training(self) -> int:
        """The cycle in training (progress.json: the trainer writes {"cycle": k, "done": false} when it starts or
        resumes in cycle k and {"cycle": k, "done": true} when cycle k is consumed, k + 1 is then in training)."""
        try:
            prog = json.loads((self.root / "progress.json").read_text())
        except FileNotFoundError:
            return 0
        return int(prog.get("cycle", 0)) + (1 if prog.get("done") else 0)

    def init(self, tokenizer_json: str, eos_id: int, cycle_tokens: float, total_tokens: Optional[float] = None,
             schedule: Optional[List[dict]] = None, line: str = "trunk", open_files: int = 16,
             pool_cycles: float = 4.0, seed: int = 2026, max_epochs: int = 1, max_strata: int = 16) -> dict:
        import tokenizers
        self.root.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            raise RuntimeError(f"{self.state_path} exists")
        # an earlier line's streams and manifests: the new line would read them as its own
        left = [n for n in ("cycles", "pool", "manifests", "val", "progress.json") if (self.root / n).exists()]
        if left:
            raise RuntimeError(f"{self.root} holds {left} of an earlier line; init a new line in an empty directory")
        repos = sorted({s.repo for s in SOURCES.values()})
        st = {"format": FORMAT, "line": line, "parent": None, "tokenizer": tokenizer_json,
              "tokenizer_sha256": _sha256(tokenizer_json), "tokenizers_version": tokenizers.__version__,
              "eos_id": eos_id, "total_tokens": total_tokens, "cycle_tokens": cycle_tokens,
              "schedule": schedule or [mixture.entry("stable", 0)], "seed": seed, "open_files": open_files,
              "pool_cycles": pool_cycles, "max_epochs": max_epochs, "max_strata": max_strata,
              "revisions": {r: hf.pin(r) for r in repos},
              "cursor": {}, "bytes_per_token": {}, "pool": {}, "cycles": {}}
        self.save(st)
        return st

    def prepared_end(self, st: dict) -> float:
        done = [int(k) for k, v in st["cycles"].items() if v["status"] in ("prepared", "released")]
        return (max(done) + 1) * st["cycle_tokens"] if done else 0.0

    def append(self, recipe: str, from_token: float) -> dict:
        """Append a schedule entry; it may not start before the end of the last prepared cycle."""
        st = self.load()
        if from_token < self.prepared_end(st):
            raise RuntimeError(f"tokens before {self.prepared_end(st):.0f} are prepared; the schedule only grows")
        if from_token <= max(e["from"] for e in st["schedule"]):
            raise RuntimeError("entries must start after the last entry")
        e = mixture.entry(recipe, from_token)

        def current(s: str) -> bool:                                 # held-out rows, of the source's text mode
            idx = self.root / "val" / s / "index.json"
            return idx.exists() and json.loads(idx.read_text()).get("text") == text_mode(SOURCES[s].control)
        held = [s for s in e["weights"] if not current(s)]
        if (self.root / "val").exists() and held:
            raise RuntimeError(f"recipe {recipe}: no held-out rows of {held} in {self.root / 'val'}, or of another "
                               f"text mode (`nstrain data val --sources {' '.join(held)}` first: the trainer "
                               "validates every source of its recipe)")
        st["schedule"].append(e)
        self.save(st)
        return st

    def load(self) -> dict:
        st = json.loads(self.state_path.read_text())
        if st.get("format") != FORMAT:
            raise RuntimeError(f"{self.state_path}: data line format {st.get('format', 1)}, this nstrain reads "
                               f"format {FORMAT} (strata, open files, shuffled and held-out documents, documents "
                               "encoded as plain text); prepare a new line")
        if _sha256(st["tokenizer"]) != st["tokenizer_sha256"]:
            raise RuntimeError("the tokenizer file changed since init")
        import tokenizers
        if tokenizers.__version__ != st["tokenizers_version"]:
            raise RuntimeError(f"tokenizers {tokenizers.__version__} != {st['tokenizers_version']} at init")
        return st

    def save(self, st: dict) -> None:
        _write_atomic(self.state_path, json.dumps(st, indent=1))

    def cycle_dir(self, k: int) -> Path:
        return self.root / "cycles" / f"cycle_{k:05d}"

    def _files(self, st: dict, name: str) -> List[Dict]:
        src = SOURCES[name]
        files = hf.list_files(src, self.root / "listings", st["revisions"][src.repo])
        stems = [Path(f["path"]).name.rsplit(".", 1)[0] for f in files]
        if len(set(stems)) != len(stems):                            # streams are named by the file's stem
            raise RuntimeError(f"{name}: two files share a name at revision {st['revisions'][src.repo]}")
        return files

    # ---------------------------------------------------------------- strata and files
    def strata(self, st: dict, name: str) -> Dict[str, List[Dict]]:
        """The source's files by stratum (hf.strata)."""
        if name not in self._by:
            self._by[name] = hf.strata(SOURCES[name], self._files(st, name), st["max_strata"])
        return self._by[name]

    @staticmethod
    def bpt(st: dict, name: str, stratum: str) -> float:
        """File bytes per training token of a stratum: measured on its files tokenized so far (and estimated from its
        held-out documents by ``build_val``), else the source's calibration."""
        return st.get("stratum_bpt", {}).get(name, {}).get(stratum) or st["bytes_per_token"][name]

    def plan_source(self, st: dict, name: str, tokens: int) -> Dict[str, Tuple[int, int]]:
        """{stratum: (tokens of the cycle, open files)}: ``tokens`` split over the strata in proportion to their
        tokens (their file bytes over their bytes per token, ``bpt``); ``open_files`` x the stratum's share files open
        (at least one, at most the stratum's files), fewer -- never below one per stratum -- while the open files
        could hold more than ``pool_cycles`` x ``tokens`` (the disk they take: an open file keeps its whole stream,
        and the files left open are the ones with the most room, so the n largest of a stratum are counted)."""
        by = self.strata(st, name)
        size = {t: float(sum(f["size"] for f in fs)) / self.bpt(st, name, t) for t, fs in by.items()}   # tokens
        total = sum(size.values())
        quota = hf.largest_remainder(int(tokens), size)
        n_open = {t: min(len(by[t]), max(1, round(st["open_files"] * size[t] / total))) for t in by}
        largest = {t: sorted((f["size"] / self.bpt(st, name, t) for f in fs), reverse=True) for t, fs in by.items()}

        def held(t: str) -> float:
            return sum(largest[t][:n_open[t]])
        cap = st["pool_cycles"] * tokens
        while sum(held(t) for t in by) > cap:
            t = max((t for t in by if n_open[t] > 1), key=lambda t: (held(t), t), default=None)
            if t is None:
                break
            n_open[t] -= 1
        return {t: (quota[t], n_open[t]) for t in by}

    def _epoch_opened(self, st: dict, name: str, stratum: str) -> bool:
        """Every file of the stratum's current epoch has been opened."""
        cur = st["cursor"].get(name, {}).get(stratum, {"epoch": 0, "index": 0})
        return cur["index"] >= len(self.strata(st, name)[stratum])

    def _next_file(self, st: dict, name: str, stratum: str) -> dict:
        """The stratum's next file.  Past its last file the stratum's next epoch reads its files again -- only
        within ``max_epochs`` (default 1: a line trains every token once): beyond it, an error."""
        files = self.strata(st, name)[stratum]
        cur = st["cursor"].setdefault(name, {}).setdefault(stratum, {"epoch": 0, "index": 0})
        if cur["index"] >= len(files):
            if cur["epoch"] + 1 >= st["max_epochs"]:
                raise RuntimeError(f"{name}/{stratum or '(flat)'}: all {len(files)} files are used (epoch "
                                   f"{cur['epoch']}); a further epoch would train them again (max_epochs "
                                   f"{st['max_epochs']} in {self.state_path})")
            cur["epoch"] += 1
            cur["index"] = 0
            print(f"{name}/{stratum or '(flat)'}: epoch {cur['epoch']} begins (its files are read again)",
                  flush=True)
        f = hf.order(files, f"{name}/{stratum}", cur["epoch"], seed=st["seed"])[cur["index"]]
        cur["index"] += 1
        return dict(f, epoch=cur["epoch"], stratum=stratum)

    @staticmethod
    def doc_seed(st: dict, name: str, f: dict) -> str:
        return f"{st['seed']}:{name}:{f['path']}:{f.get('epoch', 0)}"

    @staticmethod
    def stem(f: dict) -> str:
        """A stream's name: the file's name and its epoch (a validation stream: ".val")."""
        e = f.get("epoch", 0)
        return Path(f["path"]).name.rsplit(".", 1)[0] + (".val" if e < 0 else f".e{e}")

    def staging(self, name: str, f: dict) -> Path:
        """The directory a file is downloaded to: one per source, file and epoch (never shared by two jobs)."""
        return self.root / "staging" / name / self.stem(f)

    @staticmethod
    def holdout(name: str) -> dict:
        return {"share": SOURCES[name].holdout, "salt": HOLDOUT_SALT}

    def _fetch(self, st: dict, jobs: List[Tuple[str, dict, Path]], dl_workers: int, tok_workers: int,
               rayon: int) -> Dict:
        """Download + tokenize [(source, file, out_dir)] (epoch -1: the file's held-out documents, else its training
        documents of that epoch); returns {(source, path, epoch): meta}.  At most ``dl_workers + tok_workers`` files
        are downloaded and not yet tokenized at a time (a download waits until a tokenization has ended and deleted
        its download), so the disk in flight is bounded (``fetch_disk``).  The first failure ends the fetch: no
        further download or try starts, the tokenizers under way are killed, and it is raised."""
        keys = [(s, f["path"], f.get("epoch", 0)) for s, f, _ in jobs]
        if len(set(keys)) != len(keys):
            twice = sorted({k for k in keys if keys.count(k) > 1})
            raise RuntimeError(f"a file is asked for twice in one fetch: {twice}")
        results: Dict = {}
        todo = []
        for s, f, out in jobs:                                       # a stream finished before a crash is reused
            prefix = out / self.stem(f)
            try:
                meta = json.loads(Path(str(prefix) + ".json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                meta = None
            prov = {"path": f["path"], "revision": st["revisions"][SOURCES[s].repo], "epoch": f.get("epoch", 0),
                    "doc_seed": self.doc_seed(st, s, f), "holdout": self.holdout(s),
                    "text": text_mode(SOURCES[s].control)}
            b = Path(str(prefix) + ".bin")
            if (meta is not None and b.exists() and b.stat().st_size == 4 * meta.get("tokens", -1)
                    and all(meta.get(k) == v for k, v in prov.items())):
                meta["bin"] = str(b)
                results[(s, f["path"], f.get("epoch", 0))] = meta
                shutil.rmtree(self.staging(s, f), ignore_errors=True)  # (a crash before its download went)
            else:
                todo.append((s, f, out))
        jobs = todo
        slots, abort = threading.Semaphore(dl_workers + tok_workers), threading.Event()

        def download(s: str, f: dict) -> Path:
            while not slots.acquire(timeout=1.0):
                if abort.is_set():
                    raise RuntimeError("the fetch ended")
            if abort.is_set():                                       # (a slot freed while the fetch ends)
                slots.release()
                raise RuntimeError("the fetch ended")
            try:
                return hf.download(SOURCES[s], f["path"], self.staging(s, f), st["revisions"][SOURCES[s].repo],
                                   stop=abort)
            except BaseException:
                slots.release()
                raise

        def tokenize(job: dict, staging: Path) -> dict:
            meta = _tokenize_subprocess(job, rayon, stop=abort)
            shutil.rmtree(staging, ignore_errors=True)               # the download's directory
            slots.release()                                          # (a failed file's download stays: its slot too)
            return meta

        with cf.ThreadPoolExecutor(dl_workers) as dl, cf.ThreadPoolExecutor(tok_workers) as tp:
            try:
                running = {dl.submit(download, s, f): (s, f, out) for s, f, out in jobs}
                tokenizing = set()
                while running:                                       # downloads and tokenizations as they end
                    done, _ = cf.wait(running, return_when=cf.FIRST_COMPLETED)
                    for fut in done:
                        s, f, out = running.pop(fut)
                        if fut in tokenizing:
                            meta = fut.result()
                            meta["bin"] = str(out / self.stem(f)) + ".bin"
                            results[(s, f["path"], f.get("epoch", 0))] = meta
                            continue
                        local = fut.result()
                        out.mkdir(parents=True, exist_ok=True)
                        prefix = str(out / self.stem(f))
                        src = SOURCES[s]
                        e = f.get("epoch", 0)
                        prov = {"source": s, "repo": src.repo, "path": f["path"], "revision": st["revisions"][src.repo],
                                "epoch": e}
                        job = {"path": str(local), "fields": list(src.fields or (src.field,)),
                               "tokenizer_json": st["tokenizer"], "eos_id": st["eos_id"], "out_prefix": prefix,
                               "provenance": prov, "doc_seed": self.doc_seed(st, s, f), "holdout": self.holdout(s),
                               "keep": "val" if e < 0 else "train", "max_tokens": VAL_KEEP_TOKENS if e < 0 else 0,
                               "markers": list(src.markers), "control": list(src.control)}
                        t = tp.submit(tokenize, job, self.staging(s, f))
                        tokenizing.add(t)
                        running[t] = (s, f, out)
            except BaseException:
                abort.set()                                          # (downloads waiting for a slot end)
                dl.shutdown(wait=False, cancel_futures=True)
                tp.shutdown(wait=False, cancel_futures=True)
                raise
        return results

    def fetch_disk(self, st: dict, jobs: List[Tuple[str, dict, Path]], dl_workers: int, tok_workers: int) -> float:
        """Disk (bytes) a fetch of ``jobs`` takes at its peak: every file's stream (4 B per estimated training token)
        and the files in flight -- at most ``dl_workers + tok_workers`` downloaded and not yet tokenized (``_fetch``),
        each with its download and, while it is tokenized, its copy in file order (4 B per token) -- counted for the
        largest files."""
        est = [(f["size"], f["size"] / self.bpt(st, s, f["stratum"])) for s, f, _ in jobs]
        flight = sorted((b + 4 * n for b, n in est), reverse=True)[:dl_workers + tok_workers]
        return sum(4 * n for _, n in est) + sum(flight)

    @staticmethod
    def _piece(meta: dict, start: int, count: int, stratum: str) -> dict:
        return dict({k: meta[k] for k in PROV_KEYS}, bin=meta["bin"], start=start, count=count, stratum=stratum,
                    text=meta["text"])

    # ---------------------------------------------------------------- calibration / validation
    def calibrate(self, names: List[str], dl_workers: int = 8, tok_workers: int = 8, rayon: int = 8) -> Dict:
        """Tokenize the next file of each source's largest stratum; record file bytes per token; the stream becomes
        one of that stratum's open files.  The cursors move when the streams are registered, so a failed calibration
        takes the same files again (and reuses their finished streams)."""
        st = self.load()
        trial = json.loads(json.dumps(st))                           # the cursors advanced on a copy
        jobs, strat = [], {}
        for n in names:
            by = self.strata(trial, n)
            t = max(by, key=lambda t: (sum(f["size"] for f in by[t]), t))
            f = self._next_file(trial, n, t)
            jobs.append((n, f, self.root / "pool" / n))
            strat[(n, f["path"], f["epoch"])] = t
        res = self._fetch(st, jobs, dl_workers, tok_workers, rayon)
        st = self.load()
        for n in names:
            st["cursor"][n] = trial["cursor"][n]
        for (n, path, e), m in res.items():
            st["bytes_per_token"][n] = m["file_bytes"] / max(1, m["tokens"])
            st["pool"].setdefault(n, []).append(self._piece(m, 0, m["tokens"], strat[(n, path, e)]))
        self.save(st)
        return {n: m for (n, _, _), m in res.items()}

    def build_val(self, names: List[str], dl_workers: int = 8, tok_workers: int = 8, rayon: int = 8,
                  files_per_source: int = 8, rows: int = 64, max_len: int = 32768) -> Dict:
        """Each source's validation data into val/<source>/: per stratum, the held-out documents (never trained on,
        nstrain.data.tokenize.held) of its first files in the order of epoch 0 -- ``files_per_source`` x the
        stratum's share of them, at least one, more (up to ``VAL_FILES``) until they hold the stratum's rows at
        ``max_len`` (``rows`` x its share and a margin of 8 rows) --, each file's shuffled and kept up to
        ``VAL_KEEP_TOKENS``; val/<source>/index.json (strata with their shares of the source's tokens, files,
        revision, holdout, text mode) is written last; a source whose index is of this text mode is left as it is.
        A stratum's bytes per token not yet measured is estimated from its held-out documents (``bpt``)."""
        st = self.load()
        for d in (self.root / "val").glob("*") if (self.root / "val").exists() else []:
            for f in _temporaries(d):
                f.unlink()                                           # writers of a crashed build
        plan, got, held = {}, {}, {}
        for n in names:
            idx = self.root / "val" / n / "index.json"
            if idx.exists() and json.loads(idx.read_text()).get("text") == text_mode(SOURCES[n].control):
                continue                                             # (an index of another text mode is remade)
            by = self.strata(st, n)
            size = {t: float(sum(f["size"] for f in fs)) / self.bpt(st, n, t) for t, fs in by.items()}
            total = sum(size.values())
            plan[n] = {t: (size[t] / total, hf.order(fs, f"{n}/{t}", 0, seed=st["seed"])) for t, fs in by.items()}
            for t, (share, files) in plan[n].items():
                k = min(VAL_FILES, len(files), max(1, round(files_per_source * share)))
                held[(n, t)] = files[:k]
        while True:                                                  # fetch, then more files where too few rows
            jobs = [(n, dict(f, epoch=-1), self.root / "val" / n) for (n, t), fs in sorted(held.items()) for f in fs
                    if (n, f["path"]) not in got]
            if not jobs:
                break
            for (n, path, _), m in self._fetch(st, jobs, dl_workers, tok_workers, rayon).items():
                got[(n, path)] = m
            for (n, t), fs in sorted(held.items()):
                share, files = plan[n][t]
                want = (max(1, math.ceil(rows * share)) + 8) * (max_len + 1)
                have = sum(got[(n, f["path"])]["tokens"] for f in fs if (n, f["path"]) in got)
                if have < want and len(fs) < min(VAL_FILES, len(files)):
                    held[(n, t)] = files[:min(VAL_FILES, 2 * len(fs))]        # twice as many files
        st = self.load()
        for n in plan:
            h = SOURCES[n].holdout
            for t in plan[n]:
                ms = [got[(n, f["path"])] for f in held[(n, t)]]
                b, tok = sum(m["file_bytes"] for m in ms), sum(m["tokens_before_cap"] for m in ms)
                if tok and t not in st.get("stratum_bpt", {}).get(n, {}):    # estimated: the held-out documents
                    st.setdefault("stratum_bpt", {}).setdefault(n, {})[t] = b * h / (tok * (1 - h))
            by = self.strata(st, n)                                  # the shares in tokens, with those estimates
            size = {t: float(sum(f["size"] for f in fs)) / self.bpt(st, n, t) for t, fs in by.items()}
            items = []
            for t in sorted(plan[n]):
                files = [{"bin": Path(m["bin"]).name, "path": f["path"], "tokens": m["tokens"],
                          "blake2b": m["blake2b"]} for f in held[(n, t)] for m in [got[(n, f["path"])]] if m["tokens"]]
                items.append({"stratum": t, "share": size[t] / sum(size.values()), "files": files})
            index = {"source": n, "revision": st["revisions"][SOURCES[n].repo], "holdout": self.holdout(n),
                     "text": text_mode(SOURCES[n].control), "strata": items}
            _write_atomic(self.root / "val" / n / "index.json", json.dumps(index, indent=1))
        self.save(st)
        return {n: m for (n, _), m in got.items()}

    # ---------------------------------------------------------------- one cycle
    def prepare(self, k: int, dl_workers: int = 8, tok_workers: int = 8, rayon: int = 8, margin: float = 0.03,
                min_free_gb: float = 0.0) -> dict:
        """Cycle k (see the module documentation).  A file not yet tokenized counts at its estimate (file bytes over
        bytes per training token) less ``margin`` (the estimate may be high); before a batch of files is fetched the
        disk must hold their streams and the files in flight (``fetch_disk``) plus ``min_free_gb``, else NeedsSpace
        (nothing moved: the cursors are saved only with the batch)."""
        st = self.load()
        C = st["cycle_tokens"]
        segs = mixture.quotas(st["schedule"], k * C, C, st["total_tokens"])
        if not segs:
            raise RuntimeError(f"cycle {k} lies beyond the line's total")
        mine_status = st["cycles"].get(str(k), {}).get("status")
        if mine_status is not None:
            raise RuntimeError(f"cycle {k} is {mine_status}: a cycle is prepared once (`rebuild` re-creates a "
                               "released one from its manifest)")
        prev = st["cycles"].get(str(k - 1), {}).get("status") if k > 0 else "released"
        if prev not in ("prepared", "released"):
            raise RuntimeError(f"cycle {k - 1} must be prepared before cycle {k} (open files carry over in order)")
        need: Dict[str, int] = {}
        for _, q in segs:
            for s, v in q.items():
                need[s] = need.get(s, 0) + int(v)
        missing = [s for s in need if s not in st["bytes_per_token"]]
        if missing:
            raise RuntimeError(f"calibrate first: {missing}")
        cdir = self.cycle_dir(k)
        if cdir.exists():                                            # writers of a crashed prepare
            for f in _temporaries(cdir):
                f.unlink()
        plan = {s: self.plan_source(st, s, q) for s, q in need.items()}     # the open files of each stratum
        measured: Dict[Tuple[str, str], Tuple[int, int]] = {}                  # bytes, tokens of the new files
        # every segment (stable / decay / long ...) takes each source's strata at their shares on its own
        split = [{s: {t: x for t, (x, _) in self.plan_source(st, s, int(v)).items()} for s, v in q.items()}
                 for _, q in segs]
        want = {s: {t: sum(sp.get(s, {}).get(t, 0) for sp in split) for t in plan[s]} for s in need}
        pool: Dict[str, List[dict]] = {s: list(st["pool"].get(s, [])) for s in need}
        for s in need:                                               # an open file encodes as the source does now
            for p in pool[s]:                                        # (a piece without "text": plain text, FORMAT 4)
                if p.get("text", TEXT) != text_mode(SOURCES[s].control):
                    raise RuntimeError(f"the open file {p['path']} of {s} (epoch {p['epoch']}) was encoded with text "
                                       f"mode {p.get('text', TEXT)!r}, the source now encodes with "
                                       f"{text_mode(SOURCES[s].control)!r}: the line's open files are of another "
                                       "encoding (a new data line, `nstrain data init`)")
        inflight = st.get("inflight") or {}
        if inflight and inflight.get("cycle") != k:
            raise RuntimeError(f"cycle {inflight.get('cycle')} was being prepared; finish it first")
        jobs = [(e["source"], e["file"], cdir / e["source"]) for e in inflight.get("files", [])]
        while True:                                                  # open files until every stratum is covered
            if jobs:                                                 # (the files of a crashed prepare first)
                opened = {(s, f["path"], f["epoch"]): f for s, f, _ in jobs}
                for key, m in self._fetch(st, jobs, dl_workers, tok_workers, rayon).items():
                    t = opened[key]["stratum"]
                    pool[key[0]].append(self._piece(m, 0, m["tokens"], t))
                    b, n = measured.get((key[0], t), (0, 0))
                    measured[(key[0], t)] = (b + m["file_bytes"], n + m["tokens"])
                st = self.load()
            jobs = []
            for s, strata in plan.items():
                for t, (_, n_open) in sorted(strata.items()):
                    q = want[s][t]
                    if q <= 0:
                        continue
                    mine = [p for p in pool[s] if p["stratum"] == t]
                    n, have = len(mine), float(sum(p["count"] for p in mine))
                    # files are opened while the open ones hold too few tokens (a further epoch then, if allowed), or
                    # are fewer than n_open and the epoch has files left (never a further epoch to have more open)
                    opened = 0
                    while have < q or (n < n_open and not self._epoch_opened(st, s, t)):
                        if opened and self._epoch_opened(st, s, t):          # the files just opened are fetched
                            break                                            # first: their estimate may be low
                        f = self._next_file(st, s, t)
                        have += f["size"] / self.bpt(st, s, t) * (1 - margin)       # (bpt: per training token)
                        n += 1
                        opened += 1
                        jobs.append((s, f, cdir / s))
            if not jobs:
                break
            disk = self.fetch_disk(st, jobs, dl_workers, tok_workers)
            free = shutil.disk_usage(self.root).free
            if free < disk + min_free_gb * 1e9:
                raise NeedsSpace(f"cycle {k}: the next {len(jobs)} files need about {disk / 1e9:.0f} GB (streams, "
                                 f"downloads and temporaries in flight) and {min_free_gb:.0f} GB must stay free; "
                                 f"{free / 1e9:.0f} GB are free")
            st["inflight"] = {"cycle": k, "files": (st.get("inflight") or {}).get("files", []) +
                              [{"source": s, "file": f} for s, f, _ in jobs]}
            self.save(st)                                            # cursors and the files they cover, together
        # exact quotas: a stratum's tokens of the cycle are read from its open files so that at most its n_open stay
        # open (``_plan_files``); each segment takes its part from those files in proportion to what the cycle reads
        # of them, cut at document ends (but the last cut of a segment; within what the cycle reads of a file, but
        # past it at the cycle's end, in a file left open); the unread rest stays open
        pieces: Dict[str, List[dict]] = {s: [] for s in need}
        rest: Dict[str, List[dict]] = {}
        for s in need:
            mine = sorted(pool[s], key=lambda p: (p["stratum"], p["epoch"], p["path"], p["start"]))
            for t, (_, n_open) in sorted(plan[s].items()):
                js = [j for j, sp in enumerate(split) if sp.get(s, {}).get(t, 0) > 0]
                if not js:
                    continue
                idx = [i for i, p in enumerate(mine) if p["stratum"] == t and p["count"] > 0]
                x = _plan_files(sum(split[j][s][t] for j in js), [mine[i]["count"] for i in idx],
                                [float(mine[i]["tokens"]) for i in idx], n_open)
                rank = sorted(range(len(idx)), key=lambda n: (x[n] < mine[idx[n]]["count"], n))  # read whole first
                idx, x = [idx[n] for n in rank], [x[n] for n in rank]
                left = list(x)                                       # of what the cycle reads of each file
                for r, j in enumerate(js):
                    use = [n for n in range(len(idx)) if left[n] > 0]
                    if r == len(js) - 1:                             # the rest, cut where the files stay open
                        xs, ps = [left[n] for n in use], [mine[idx[n]] for n in use]
                    else:
                        xs = _split(split[j][s][t], [left[n] for n in use], [float(x[n]) for n in use])
                        ps = [dict(mine[idx[n]], count=left[n]) for n in use]
                    for n, y in zip(use, self._cuts(ps, xs, st["eos_id"])):
                        if y:
                            p = mine[idx[n]]
                            pieces[s].append(dict(p, bin=self._link(p["bin"], cdir / s), count=y, segment=j))
                            mine[idx[n]] = dict(p, start=p["start"] + y, count=p["count"] - y)
                            left[n] -= y
            pieces[s].sort(key=lambda p: p["segment"])
            rest[s] = [dict(p, bin=self._link(p["bin"], self.root / "pool" / s)) for p in mine if p["count"] > 0]
        for s in st["pool"]:
            if s not in need:
                rest.setdefault(s, st["pool"][s])
        consumed = self._audit(k, st, segs, split, pieces, rest)    # before anything is committed
        manifest = {"format": FORMAT, "order": ORDER, "line": st["line"], "parent": st["parent"], "cycle": k,
                    "start_token": k * C,
                    "tokens": int(sum(sum(q.values()) for _, q in segs)),
                    "tokenizer_sha256": st["tokenizer_sha256"], "tokenizers_version": st["tokenizers_version"],
                    "revisions": st["revisions"],
                    "segments": [{"phase": e["recipe"], "seq_len": e["seq_len"], "weights": e["weights"],
                                  "quota": {s: int(v) for s, v in q.items()}} for e, q in segs],
                    "sources": pieces}
        text = json.dumps(manifest, indent=1)
        (self.root / "manifests").mkdir(exist_ok=True)
        _write_atomic(self.root / "manifests" / f"cycle_{k:05d}.json", text)
        st["pool"] = rest
        st["consumed"] = consumed
        st["inflight"] = None
        for (s, t), (b, n) in sorted(measured.items()):              # the strata's bytes per token: measured on
            sums = st.setdefault("stratum_sums", {}).setdefault(s, {})    # every file they opened so far
            b0, n0 = sums.get(t, (0, 0))
            sums[t] = (b0 + b, n0 + n)
            if n0 + n:
                st.setdefault("stratum_bpt", {}).setdefault(s, {})[t] = (b0 + b) / (n0 + n)
        st["cycles"][str(k)] = {"status": "prepared", "manifest_sha256": hashlib.sha256(text.encode()).hexdigest()}
        _write_atomic(self.root / "manifests" / f"cycle_{k:05d}.state.json", json.dumps(st, indent=1))
        self.save(st)                                                # the commit point
        _write_atomic(cdir / "manifest.json", text)                  # the trainer's signal, last
        return manifest

    @staticmethod
    def _audit(k: int, st: dict, segs, split, pieces: Dict[str, List[dict]], rest: Dict[str, List[dict]]) -> dict:
        """The cycle's invariants, checked before it is committed (a violation is an error: the cycle is not
        prepared, nothing is trained on it): every segment holds each source's quota exactly and each stratum's
        share of it; every file's stream is read once, from its start, without a gap or an overlap -- each piece
        starts where the line's reading of that stream stopped (``consumed``: per stream, the end of what earlier
        cycles took), the pieces of a stream follow each other, and an open file's rest starts where the last piece
        ended.  Returns the new ``consumed``."""
        bad = []
        for j, (_, q) in enumerate(segs):
            for s, v in q.items():
                got = [p for p in pieces.get(s, []) if p["segment"] == j]
                if sum(p["count"] for p in got) != int(v):
                    bad.append(f"segment {j} {s}: {sum(p['count'] for p in got)} tokens, quota {int(v)}")
                per: Dict[str, int] = {}
                for p in got:
                    per[p["stratum"]] = per.get(p["stratum"], 0) + p["count"]
                want = {t: x for t, x in split[j].get(s, {}).items() if x}
                if per != want:
                    bad.append(f"segment {j} {s}: strata {per}, planned {want}")
        consumed = json.loads(json.dumps(st.get("consumed", {})))
        for s, ps in pieces.items():
            mark = consumed.setdefault(s, {})
            for p in sorted(ps, key=lambda p: (p["path"], p["epoch"], p["start"])):
                key = f"{p['path']}@{p['epoch']}"
                at = mark.get(key, 0)
                if p["start"] != at or p["count"] <= 0 or p["start"] + p["count"] > p["tokens"]:
                    bad.append(f"{s} {key}: a piece at {p['start']} (+{p['count']}) where the stream's reading "
                               f"stopped at {at} (of {p['tokens']})")
                mark[key] = p["start"] + p["count"]
        for s, ps in rest.items():
            for p in ps:
                key = f"{p['path']}@{p['epoch']}"
                if consumed.get(s, {}).get(key, 0) != p["start"]:
                    bad.append(f"{s} {key}: the open rest starts at {p['start']}, the reading stopped at "
                               f"{consumed.get(s, {}).get(key, 0)}")
        if bad:
            raise RuntimeError(f"cycle {k} fails its audit ({len(bad)} problems), nothing committed: " +
                               "; ".join(bad[:10]))
        return consumed

    @staticmethod
    def _cuts(ps: List[dict], xs: List[int], eos: int) -> List[int]:
        """The takes ``xs`` of the pieces ``ps`` moved to document ends (one past an EOS, within 64K tokens), the
        difference carried to the next take; the last take ends exactly at the total (unchanged takes when that
        does not fit)."""
        nz = [n for n, x in enumerate(xs) if x]
        out, carry = list(xs), 0
        for r, n in enumerate(nz):
            p, want = ps[n], xs[n] + carry
            if r == len(nz) - 1:
                if not 0 <= want <= p["count"]:
                    return list(xs)
                out[n] = want
                break
            want = max(0, min(want, p["count"]))
            cut = _doc_end(p["bin"], p["start"] + want, p["start"] + p["count"], eos) - p["start"] if want else 0
            out[n] = cut
            carry = want - cut + (xs[n] + carry - want)
        if sum(out) != sum(xs) or any(not 0 <= x <= p["count"] for x, p in zip(out, ps)):
            return list(xs)
        return out

    def ensure_visible(self, k: int) -> None:
        """The cycle directory's manifest.json of a prepared cycle (restored from the permanent copy after a crash
        between the commit and its write)."""
        st = self.load()
        if st["cycles"].get(str(k), {}).get("status") != "prepared":
            return
        m = self.cycle_dir(k) / "manifest.json"
        if not m.exists():
            text = (self.root / "manifests" / f"cycle_{k:05d}.json").read_text()
            if hashlib.sha256(text.encode()).hexdigest() != st["cycles"][str(k)]["manifest_sha256"]:
                raise RuntimeError(f"cycle {k}: the permanent manifest does not match the state")
            _write_atomic(m, text)

    # ---------------------------------------------------------------- branches
    def fork(self, parent_root: str, k: int, schedule: List[dict], total: float, cycle_tokens: float, line: str,
             dl_workers: int = 8, tok_workers: int = 8, rayon: int = 8) -> dict:
        """A branch from the parent's state right after its cycle k (cursors and open files continue from there)."""
        if self.state_path.exists():
            raise RuntimeError(f"{self.state_path} exists")
        parent = Path(parent_root)
        snap = json.loads((parent / "manifests" / f"cycle_{k:05d}.state.json").read_text())
        self.root.mkdir(parents=True, exist_ok=True)
        st = dict(snap, line=line, total_tokens=total, cycle_tokens=cycle_tokens, schedule=schedule, cycles={},
                  parent={"root": str(parent), "line": snap["line"], "cycle": k,
                          "tokens": (k + 1) * snap["cycle_tokens"]})
        # the open files: hard-linked from the parent first (a link keeps the bytes when the parent's preparer
        # releases its name meanwhile), the ones already gone re-created from their provenance; state.json is written
        # last, so an interrupted fork is completed by running it again
        pool, jobs = {}, []
        for s, ps in snap["pool"].items():
            for p in ps:
                mine = self.root / "pool" / s / Path(p["bin"]).name
                if mine.exists() and _blake2b_file(str(mine)) == p["blake2b"]:
                    continue
                if mine.exists():
                    mine.unlink()                                    # not the parent's stream (an interrupted fork)
                try:
                    if _blake2b_file(p["bin"]) != p["blake2b"]:
                        raise FileNotFoundError(p["bin"])            # the name now holds another stream
                    self._link(p["bin"], self.root / "pool" / s)
                except FileNotFoundError:
                    jobs.append((s, {"path": p["path"], "epoch": p["epoch"], "stratum": p["stratum"]},
                                 self.root / "pool" / s))
        got = self._fetch(st, jobs, dl_workers, tok_workers, rayon) if jobs else {}
        for s, ps in snap["pool"].items():
            pool[s] = []
            for p in ps:
                m = got.get((s, p["path"], p["epoch"]))
                if m is not None and m["blake2b"] != p["blake2b"]:
                    raise RuntimeError(f"re-created {p['path']} differs from the parent's stream")
                pool[s].append(dict(p, bin=str(self.root / "pool" / s / Path(p["bin"]).name)))
        st["pool"] = pool
        self.save(st)
        return st

    @staticmethod
    def _link(path: str, dest: Path) -> str:
        """The same stream under ``dest`` (hard link; the bytes are shared, deleting one name keeps the other); a
        different file under that name is an error."""
        p = Path(path)
        if p.parent == dest:
            return str(p)
        dest.mkdir(parents=True, exist_ok=True)
        q = dest / p.name
        if not q.exists():
            os.link(p, q)
            fd = os.open(dest, os.O_RDONLY)                          # (the name durably there before a manifest
            try:                                                     # that lists it)
                os.fsync(fd)
            finally:
                os.close(fd)
        elif not os.path.samefile(p, q):
            raise RuntimeError(f"{q} is another stream than {p}")
        return str(q)

    # ---------------------------------------------------------------- after training / recovery
    def release(self, k: int) -> None:
        """Delete a consumed cycle (its permanent manifest and the open files' streams stay).  The status is
        "releasing" while the files go, so a crash in between never leaves a "prepared" cycle without its files; its
        manifest.json goes first, so a crash never leaves a manifest a trainer would read without its streams.  A
        cycle the trainer has not consumed is refused."""
        if k >= self.in_training():
            raise RuntimeError(f"cycle {k} is not consumed (the cycle in training is {self.in_training()})")
        st = self.load()
        st["cycles"][str(k)]["status"] = "releasing"
        self.save(st)
        d = self.cycle_dir(k)
        if (d / "manifest.json").exists():
            (d / "manifest.json").unlink()
            fd = os.open(d, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        if d.exists():
            shutil.rmtree(d)
        st["cycles"][str(k)]["status"] = "released"
        live = {os.path.realpath(p["bin"]) for ps in st["pool"].values() for p in ps}
        hold = self.root / "pool"
        if hold.exists():
            for f in _temporaries(hold):
                f.unlink()                                           # writers of a crashed calibrate / fork
            for f in hold.rglob("*.bin"):
                if os.path.realpath(f) not in live:
                    f.unlink()
                    j = Path(str(f)[:-len(".bin")] + ".json")
                    if j.exists():
                        j.unlink()
        self.save(st)

    def rebuild(self, k: int, dl_workers: int = 8, tok_workers: int = 8, rayon: int = 8) -> dict:
        """Re-create a released cycle from its permanent manifest; every stream's blake2b must match.  Its
        manifest.json (a half-released cycle's) goes first and comes back last, when the streams are whole.  (A
        rollback, ``nstrain rollback``, moves the cycle in training back; the loop then rebuilds the released cycles
        of its window.)"""
        st = self.load()
        text = (self.root / "manifests" / f"cycle_{k:05d}.json").read_text()
        man = json.loads(text)
        if man["tokenizer_sha256"] != st["tokenizer_sha256"]:
            raise RuntimeError("the manifest was built with another tokenizer")
        cdir = self.cycle_dir(k)
        if cdir.exists():                                            # writers of a crashed or aborted fetch
            if (cdir / "manifest.json").exists():
                (cdir / "manifest.json").unlink()
            for f in _temporaries(cdir):
                f.unlink()
        jobs, seen, checked = [], set(), set()
        for s, ps in man["sources"].items():
            for p in ps:
                if (s, p["path"], p["epoch"]) in seen:
                    continue
                if p["bin"] in checked:                              # an earlier piece of the same stream
                    continue
                checked.add(p["bin"])
                if Path(p["bin"]).exists():                          # an open file, or an interrupted rebuild's
                    if _blake2b_file(p["bin"]) == p["blake2b"]:
                        continue
                    os.unlink(p["bin"])                              # partial / foreign stream: re-create it
                kept = self.root / "pool" / s / Path(p["bin"]).name    # the same stream still open
                if kept.exists() and _blake2b_file(str(kept)) == p["blake2b"]:
                    self._link(str(kept), Path(p["bin"]).parent)
                    continue
                seen.add((s, p["path"], p["epoch"]))
                jobs.append((s, {"path": p["path"], "epoch": p["epoch"]}, Path(p["bin"]).parent))
        got = self._fetch(st, jobs, dl_workers, tok_workers, rayon)
        for s, ps in man["sources"].items():
            for p in ps:
                m = got.get((s, p["path"], p["epoch"]))
                if m is not None and (m["blake2b"] != p["blake2b"] or m["tokens"] != p["tokens"]):
                    raise RuntimeError(f"rebuild mismatch {s} {p['path']}: {m['blake2b']} != {p['blake2b']}")
        st["cycles"][str(k)]["status"] = "prepared"
        self.save(st)
        _write_atomic(cdir / "manifest.json", text)
        return man
