"""Cycles: the budget is consumed in fixed-size cycles; each cycle is downloaded, tokenized, trained on, and deleted.

Work directory layout::

    state.json                 tokenizer (+ sha256, library version), pinned dataset revisions, budget, cycle size,
                               per-source cursor / calibration / carry-over, cycle status
    listings/*.json            file listings of the sources at their pinned revisions
    staging/                   downloads in flight (deleted after tokenization)
    cycles/cycle_NNNNN/        <source>/<file>.bin|.json token streams + manifest.json (the training input)
    manifests/cycle_NNNNN.json permanent copies of the manifests (kept after the cycle is released)
    manifests/cycle_NNNNN.state.json  the state right after cycle N was prepared (cursors, carry-over): forks start here
    carry/<source>/            tails of partly used files, carried into the next cycle (hard links)
    val/<source>.bin|.json     held-out validation streams (one file per source, never trained on)
    progress.json              written by the trainer: {"cycle": k, "done": true} when cycle k is consumed
    lock                       held by every writer of state.json (one at a time; see ``Work.lock``)

Lines: a work directory is one line -- the trunk (schedule ``stable`` from token 0, no total: extendable until the
corpora are exhausted) or a branch forked from a trunk cycle with its own schedule and total (``fork``).  Schedules are
append-only (``append``): an entry may only start at or after the end of the last prepared cycle.

Crash safety of ``prepare``: the files a cycle fetches are recorded in ``state.json`` (``inflight``) together with the
advanced cursors, so a restarted ``prepare`` of the same cycle takes the same files (a finished stream is reused); the
commit order is the permanent manifest, the state snapshot, the state (the commit point), then ``manifest.json`` in
the cycle directory -- the trainer's signal -- written atomically; ``ensure_visible`` restores that last file from
the permanent copy after a crash between the commit and it.

Determinism: every input is pinned -- dataset revisions (commit sha), the file order of each source (fixed shuffle per
(source, epoch), the held-out file excluded), the tokenizer file (sha256) and library version.  A cycle's manifest
lists, per source, pieces {bin, start, count} with their provenance (repo, path, revision, epoch, tokens, blake2b of
the whole stream); ``rebuild`` re-creates a released cycle from its permanent manifest and checks every blake2b.
"""
from __future__ import annotations

import concurrent.futures as cf
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from nstrain.data import hf, mixture
from nstrain.data.sources import SOURCES

PROV_KEYS = ("repo", "path", "revision", "epoch", "tokens", "blake2b")


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


def _blake2b_file(path: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _tokenize_subprocess(job: dict, rayon: int) -> dict:
    """Tokenize one file in its own interpreter: independent of the caller's ``__main__`` (unlike multiprocessing
    spawn / forkserver) and of the caller's threads (unlike fork); a failure is an exception here, never a hang; the
    tokenizer dies with this process (nstrain.proc)."""
    root = str(Path(__file__).resolve().parents[2])                   # the directory holding the nstrain package
    env = dict(os.environ, RAYON_NUM_THREADS=str(rayon), TOKENIZERS_PARALLELISM="true",
               NSTRAIN_PREPARER_PID=str(os.getpid()),
               PYTHONPATH=os.pathsep.join(x for x in (root, os.environ.get("PYTHONPATH", "")) if x))
    p = subprocess.run([sys.executable, "-m", "nstrain.data.tokenize", json.dumps(job)], env=env, capture_output=True,
                       text=True)
    if p.returncode != 0:
        raise RuntimeError(f"tokenizing {job['path']} failed (exit {p.returncode}):\n{p.stderr[-4000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


class Work:
    def __init__(self, root: str):
        self.root = Path(root)
        self.state_path = self.root / "state.json"

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
             schedule: Optional[List[dict]] = None, line: str = "trunk") -> dict:
        import tokenizers
        self.root.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            raise RuntimeError(f"{self.state_path} exists")
        repos = sorted({s.repo for s in SOURCES.values()})
        st = {"line": line, "parent": None, "tokenizer": tokenizer_json, "tokenizer_sha256": _sha256(tokenizer_json),
              "tokenizers_version": tokenizers.__version__, "eos_id": eos_id,
              "total_tokens": total_tokens, "cycle_tokens": cycle_tokens,
              "schedule": schedule or [mixture.entry("stable", 0)],
              "revisions": {r: hf.pin(r) for r in repos},
              "cursor": {}, "bytes_per_token": {}, "carry": {}, "cycles": {}}
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
        st["schedule"].append(mixture.entry(recipe, from_token))
        self.save(st)
        return st

    def load(self) -> dict:
        st = json.loads(self.state_path.read_text())
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

    # ---------------------------------------------------------------- files
    def _next_file(self, st: dict, name: str) -> dict:
        n = len(self._files(st, name)) - 1                         # the training files (held-out file excluded)
        cur = st["cursor"].setdefault(name, {"epoch": 0, "index": 0})
        if cur["index"] >= n:
            cur["epoch"] += 1
            cur["index"] = 0
        f = hf.order(self._files(st, name), name, cur["epoch"])[cur["index"]]
        cur["index"] += 1
        return dict(f, epoch=cur["epoch"])

    def _fetch(self, st: dict, jobs: List[Tuple[str, dict, Path]], dl_workers: int, tok_workers: int,
               rayon: int) -> Dict:
        """Download + tokenize [(source, file, out_dir)]; returns {(source, path): meta}."""
        staging = self.root / "staging"
        staging.mkdir(parents=True, exist_ok=True)
        results: Dict = {}
        todo = []
        for s, f, out in jobs:                                       # a stream finished before a crash is reused
            prefix = out / Path(f["path"]).name.rsplit(".", 1)[0]
            try:
                meta = json.loads(prefix.with_suffix(".json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                meta = None
            prov = {"path": f["path"], "revision": st["revisions"][SOURCES[s].repo], "epoch": f.get("epoch", 0)}
            if (meta is not None and Path(str(prefix) + ".bin").exists()
                    and all(meta.get(k) == v for k, v in prov.items())):
                meta["bin"] = str(prefix) + ".bin"
                results[(s, f["path"])] = meta
            else:
                todo.append((s, f, out))
        jobs = todo
        with cf.ThreadPoolExecutor(dl_workers) as dl, cf.ThreadPoolExecutor(tok_workers) as tp:
            try:
                dls = {dl.submit(hf.download, SOURCES[s], f["path"], staging, st["revisions"][SOURCES[s].repo]):
                       (s, f, out) for s, f, out in jobs}
                pending = {}
                for fut in cf.as_completed(dls):
                    s, f, out = dls[fut]
                    local = fut.result()
                    out.mkdir(parents=True, exist_ok=True)
                    prefix = str(out / Path(f["path"]).name.rsplit(".", 1)[0])
                    src = SOURCES[s]
                    prov = {"source": s, "repo": src.repo, "path": f["path"], "revision": st["revisions"][src.repo],
                            "epoch": f.get("epoch", 0)}
                    job = {"path": str(local), "fields": list(src.fields or (src.field,)),
                           "tokenizer_json": st["tokenizer"], "eos_id": st["eos_id"], "out_prefix": prefix,
                           "provenance": prov}
                    pending[tp.submit(_tokenize_subprocess, job, rayon)] = (s, f, prefix)
                for fut in cf.as_completed(pending):
                    s, f, prefix = pending[fut]
                    meta = fut.result()
                    meta["bin"] = prefix + ".bin"
                    results[(s, f["path"])] = meta
            except BaseException:
                dl.shutdown(wait=False, cancel_futures=True)
                tp.shutdown(wait=False, cancel_futures=True)
                raise
        return results

    @staticmethod
    def _piece(meta: dict, start: int, count: int) -> dict:
        return dict({k: meta[k] for k in PROV_KEYS}, bin=meta["bin"], start=start, count=count)

    # ---------------------------------------------------------------- calibration / validation
    def calibrate(self, names: List[str], dl_workers: int = 8, tok_workers: int = 8, rayon: int = 8) -> Dict:
        """Tokenize the next file of each source; record file bytes per token; keep the stream as carry-over."""
        st = self.load()
        jobs = [(n, self._next_file(st, n), self.root / "carry" / n) for n in names]
        self.save(st)
        res = self._fetch(st, jobs, dl_workers, tok_workers, rayon)
        st = self.load()
        for (n, _), m in res.items():
            st["bytes_per_token"][n] = m["file_bytes"] / max(1, m["tokens"])
            st["carry"].setdefault(n, []).append(self._piece(m, 0, m["tokens"]))
        self.save(st)
        return {n: m for (n, _), m in res.items()}

    def build_val(self, names: List[str], dl_workers: int = 8, tok_workers: int = 8, rayon: int = 8) -> Dict:
        """Tokenize each source's held-out file into val/<source>/ (never part of any cycle)."""
        st = self.load()
        jobs = []
        for n in names:
            done = self.root / "val" / n
            if any(done.glob("*.bin")) and any(done.glob("*.json")):
                continue
            f = dict(hf.held_out(self._files(st, n), n), epoch=-1)
            jobs.append((n, f, self.root / "val" / n))
        return {n: m for (n, _), m in self._fetch(st, jobs, dl_workers, tok_workers, rayon).items()}

    # ---------------------------------------------------------------- one cycle
    def prepare(self, k: int, dl_workers: int = 8, tok_workers: int = 8, rayon: int = 8, margin: float = 0.03) -> dict:
        st = self.load()
        C = st["cycle_tokens"]
        segs = mixture.quotas(st["schedule"], k * C, C, st["total_tokens"])
        if not segs:
            raise RuntimeError(f"cycle {k} lies beyond the line's total")
        prev = st["cycles"].get(str(k - 1), {}).get("status") if k > 0 else "released"
        if prev not in ("prepared", "released"):
            raise RuntimeError(f"cycle {k - 1} must be prepared before cycle {k} (carry-over order)")
        need: Dict[str, int] = {}
        for _, q in segs:
            for s, v in q.items():
                need[s] = need.get(s, 0) + int(v)
        missing = [s for s in need if s not in st["bytes_per_token"]]
        if missing:
            raise RuntimeError(f"calibrate first: {missing}")
        cdir = self.cycle_dir(k)
        if cdir.exists():                                            # writers of a crashed prepare
            for f in cdir.rglob("*.bin.part*"):
                f.unlink()
        have: Dict[str, List[dict]] = {s: list(st["carry"].get(s, [])) for s in need}
        inflight = st.get("inflight") or {}
        jobs = [(e["source"], e["file"], cdir / e["source"]) for e in inflight.get("files", [])
                if inflight.get("cycle") == k]
        if inflight and inflight.get("cycle") != k:
            raise RuntimeError(f"cycle {inflight.get('cycle')} was being prepared; finish it first")
        while True:                                                  # fetch until every quota is covered
            if jobs:                                                 # the files of a crashed prepare first
                for (s, _), m in self._fetch(st, jobs, dl_workers, tok_workers, rayon).items():
                    have[s].append(self._piece(m, 0, m["tokens"]))
                have = {s: sorted(ps, key=lambda p: (p["epoch"], p["path"]) if p.get("start", 0) == 0 else (-1, ""))
                        for s, ps in have.items()}
                st = self.load()
            jobs = []
            for s, q in need.items():
                avail = sum(p["count"] for p in have[s])
                est = 0.0
                while avail + est < q * (1 + margin):
                    f = self._next_file(st, s)
                    est += f["size"] / st["bytes_per_token"][s]
                    jobs.append((s, f, cdir / s))
            st["inflight"] = {"cycle": k, "files": (st.get("inflight") or {}).get("files", []) +
                              [{"source": s, "file": f} for s, f, _ in jobs]}
            self.save(st)                                            # cursors and the files they cover, together
            if not jobs:
                break
            for (s, _), m in self._fetch(st, jobs, dl_workers, tok_workers, rayon).items():
                have[s].append(self._piece(m, 0, m["tokens"]))
            have = {s: sorted(ps, key=lambda p: (p["epoch"], p["path"]) if p.get("start", 0) == 0 else (-1, ""))
                    for s, ps in have.items()}
            st = self.load()
        # exact quotas: carried tails first, then the new files in path order; the rest is the next carry-over
        pieces: Dict[str, List[dict]] = {}
        carry: Dict[str, List[dict]] = {}
        for s, q in need.items():
            left, use, rest = q, [], []
            for p in have[s]:
                if left <= 0:
                    rest.append(p)
                    continue
                take = min(p["count"], left)
                use.append(dict(p, bin=self._link(p["bin"], cdir / s), count=take))
                left -= take
                if take < p["count"]:
                    rest.append(dict(p, start=p["start"] + take, count=p["count"] - take))
            pieces[s] = use
            carry[s] = [dict(p, bin=self._link(p["bin"], self.root / "carry" / s)) for p in rest]
        for s in st["carry"]:
            if s not in need:
                carry.setdefault(s, st["carry"][s])
        manifest = {"line": st["line"], "parent": st["parent"], "cycle": k, "start_token": k * C,
                    "tokens": int(sum(sum(q.values()) for _, q in segs)),
                    "tokenizer_sha256": st["tokenizer_sha256"], "tokenizers_version": st["tokenizers_version"],
                    "revisions": st["revisions"],
                    "segments": [{"phase": e["recipe"], "seq_len": e["seq_len"], "weights": e["weights"],
                                  "quota": {s: int(v) for s, v in q.items()}} for e, q in segs],
                    "sources": pieces}
        text = json.dumps(manifest, indent=1)
        (self.root / "manifests").mkdir(exist_ok=True)
        _write_atomic(self.root / "manifests" / f"cycle_{k:05d}.json", text)
        st["carry"] = carry
        st["inflight"] = None
        st["cycles"][str(k)] = {"status": "prepared", "manifest_sha256": hashlib.sha256(text.encode()).hexdigest()}
        _write_atomic(self.root / "manifests" / f"cycle_{k:05d}.state.json", json.dumps(st, indent=1))
        self.save(st)                                                # the commit point
        _write_atomic(cdir / "manifest.json", text)                  # the trainer's signal, last
        return manifest

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
        """A branch from the parent's state right after its cycle k (cursors and carried tails continue from there)."""
        if self.state_path.exists():
            raise RuntimeError(f"{self.state_path} exists")
        parent = Path(parent_root)
        snap = json.loads((parent / "manifests" / f"cycle_{k:05d}.state.json").read_text())
        self.root.mkdir(parents=True, exist_ok=True)
        st = dict(snap, line=line, total_tokens=total, cycle_tokens=cycle_tokens, schedule=schedule, cycles={},
                  parent={"root": str(parent), "line": snap["line"], "cycle": k,
                          "tokens": (k + 1) * snap["cycle_tokens"]})
        # the carried tails: hard-linked from the parent first (a link keeps the bytes when the parent's preparer
        # releases its name meanwhile), the ones already gone re-created from their provenance; state.json is written
        # last, so an interrupted fork is completed by running it again
        carry, jobs = {}, []
        for s, ps in snap["carry"].items():
            for p in ps:
                mine = self.root / "carry" / s / Path(p["bin"]).name
                if mine.exists():
                    continue
                try:
                    self._link(p["bin"], self.root / "carry" / s)
                except FileNotFoundError:
                    jobs.append((s, {"path": p["path"], "epoch": p["epoch"]}, self.root / "carry" / s))
        got = self._fetch(st, jobs, dl_workers, tok_workers, rayon) if jobs else {}
        for s, ps in snap["carry"].items():
            carry[s] = []
            for p in ps:
                m = got.get((s, p["path"]))
                if m is not None and m["blake2b"] != p["blake2b"]:
                    raise RuntimeError(f"re-created {p['path']} differs from the parent's stream")
                carry[s].append(dict(p, bin=str(self.root / "carry" / s / Path(p["bin"]).name)))
        st["carry"] = carry
        self.save(st)
        return st

    @staticmethod
    def _link(path: str, dest: Path) -> str:
        """The same stream under ``dest`` (hard link; the bytes are shared, deleting one name keeps the other)."""
        p = Path(path)
        if p.parent == dest:
            return str(p)
        dest.mkdir(parents=True, exist_ok=True)
        q = dest / p.name
        if not q.exists():
            os.link(p, q)
        return str(q)

    # ---------------------------------------------------------------- after training / recovery
    def release(self, k: int) -> None:
        """Delete a consumed cycle (its permanent manifest and carried tails stay).  The status is "releasing" while
        the files go, so a crash in between never leaves a "prepared" cycle without its files."""
        st = self.load()
        st["cycles"][str(k)]["status"] = "releasing"
        self.save(st)
        d = self.cycle_dir(k)
        if d.exists():
            shutil.rmtree(d)
        st["cycles"][str(k)]["status"] = "released"
        live = {p["bin"] for ps in st["carry"].values() for p in ps}
        hold = self.root / "carry"
        if hold.exists():
            for f in hold.rglob("*.bin"):
                if str(f) not in live:
                    f.unlink()
                    j = f.with_suffix(".json")
                    if j.exists():
                        j.unlink()
        self.save(st)

    def rebuild(self, k: int, dl_workers: int = 8, tok_workers: int = 8, rayon: int = 8) -> dict:
        """Re-create a released cycle from its permanent manifest; every stream's blake2b must match.  A cycle before
        the one in training (a rollback) becomes the cycle in training (progress.json), so the loop keeps it and the
        cycles after it until the trainer moves on."""
        st = self.load()
        if k < self.in_training():
            _write_atomic(self.root / "progress.json", json.dumps({"cycle": k, "done": False}))
            print(f"cycle {k} is now the cycle in training (progress.json); the preparer keeps cycles {k}.. "
                  "until the trainer moves on", flush=True)
        text = (self.root / "manifests" / f"cycle_{k:05d}.json").read_text()
        man = json.loads(text)
        if man["tokenizer_sha256"] != st["tokenizer_sha256"]:
            raise RuntimeError("the manifest was built with another tokenizer")
        cdir = self.cycle_dir(k)
        jobs, seen, checked = [], set(), set()
        for s, ps in man["sources"].items():
            for p in ps:
                if p["path"] in seen:
                    continue
                if p["bin"] in checked:                              # an earlier piece of the same stream
                    continue
                checked.add(p["bin"])
                if Path(p["bin"]).exists():                          # carried tail, or left by an interrupted rebuild
                    if _blake2b_file(p["bin"]) == p["blake2b"]:
                        continue
                    os.unlink(p["bin"])                              # partial / foreign stream: re-create it
                kept = self.root / "carry" / s / Path(p["bin"]).name    # the same stream still carried over
                if kept.exists() and _blake2b_file(str(kept)) == p["blake2b"]:
                    self._link(str(kept), Path(p["bin"]).parent)
                    continue
                seen.add(p["path"])
                jobs.append((s, {"path": p["path"], "epoch": p["epoch"]}, Path(p["bin"]).parent))
        got = self._fetch(st, jobs, dl_workers, tok_workers, rayon)
        for s, ps in man["sources"].items():
            for p in ps:
                m = got.get((s, p["path"]))
                if m is not None and (m["blake2b"] != p["blake2b"] or m["tokens"] != p["tokens"]):
                    raise RuntimeError(f"rebuild mismatch {s} {p['path']}: {m['blake2b']} != {p['blake2b']}")
        st["cycles"][str(k)]["status"] = "prepared"
        self.save(st)
        _write_atomic(cdir / "manifest.json", text)
        return man
