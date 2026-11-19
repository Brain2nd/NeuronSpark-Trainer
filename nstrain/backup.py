"""A line's backup on Hugging Face: one repository holds what a resume needs, overwritten at every validation.

A Hugging Face key turns it on: ``nstrain new ... --hf-key KEY`` or ``nstrain run RUN --hf-key KEY`` (``enable``) keeps
the key in ``<run_dir>/backup/hf_key`` (mode 600; never uploaded, never in run.json) and sets run.json's ``backup``
(``{"repo": "<the key's account>/<line>-backup", "token_file": that file, "private": false}``: public).  Then the
supervisor starts ``nstrain backup RUN`` (nstrain.supervise.BackupSupervisor) whenever a checkpoint is due
(``pending``): the first checkpoint of every validation interval (``validation.every_tokens``), which is the one written
at the validation (the trainer writes both at the update that reaches it); the checkpoint ``latest`` names after
``nstrain rollback`` (the line went back); the newest one while the repository holds nothing of the line.  The
checkpoint the repository holds is sent again when its trainer.json changed (marked kept).  ``nstrain backup RUN
--latest`` sends the newest checkpoint whatever its interval (before a planned move of the line).  An upload replaces
the repository's content in one commit, so the repository always holds one whole state; its history is then squashed and
the stored files it no longer references (the checkpoint it held before, the streams of released cycles) are deleted,
so it holds one copy.  A file the repository already holds is not sent again.  The repository::

    BACKUP.json                     what it holds: line, checkpoint, update, tokens, time; run_dir, data_work, val_work
                                    and the tokenizer (the paths a resume needs); code and library versions; every
                                    file's size and sha256; ``links``: the names of a file held under another name
                                    (on the host one file: a cycle's stream is a hard link to the open file's)
    README.md
    run/run.json, run/wandb_run.json, run/branches.json, run/rollback.json        (those the line has)
    run/checkpoints/latest, run/checkpoints/u<N>/{model,optimizer}.safetensors, trainer.json
    data/state.json, data/progress.json, data/manifests/*.json, data/listings/*.json      the data line's state
    data/pool/<source>/*            the open files' streams and their metas
    data/cycles/cycle_<K>/...       every prepared cycle's streams, their metas and its manifest.json
    val/<source>/...                the held-out streams (val_work)
    tokenizer/<file>                the tokenizer of the data line

That is what the line reads when it continues on its own host: the trainer the checkpoint and the cycles from its
position on, the preparer the state, the open files and the listings for the cycles after them.  A cycle the trainer
finished and released before the upload read the state (the trainer goes on during an upload) is not in it: the
preparer rebuilds it from its permanent manifest when the trainer needs it, as after a rollback.
``data/progress.json`` holds the checkpoint's cycle (what the trainer marks when it resumes).

Before a commit the checkpoint's files are checked against their blake2b (a damaged checkpoint never replaces the
repository's), after it every file the repository holds against the upload (sha256 of a stored file, git blob id of a
small one).  A repository that holds another line's backup, or files no backup wrote, is refused: an upload replaces
a repository's whole content.  The key goes to huggingface.co only (``ENDPOINT``; a download mirror set in
``HF_ENDPOINT`` never sees it) and is never printed; it is written only to ``hf_key``.  Every request has a timeout
(nstrain.data.hf.hub_timeouts): one that stalls fails the upload, which the supervisor starts again.

``nstrain restore REPO --hf-key KEY`` puts a line back on another host at the same absolute paths (the checkpoints and
the data state name them; they must be free there): every file checked against BACKUP.json and put where it was, the
hard links made again, the key kept for the line's backup; then ``nstrain run RUN``.

In ``<run_dir>/backup/``: ``hf_key``, ``record.json`` (what the repository holds, written after each upload), ``lock``
(one upload at a time), ``upload.log``, ``hashes.json`` (the sha256 of the files sent, by inode, size and time: a file
unchanged since is not read again), ``stage/`` (the files of the upload under way).
"""
from __future__ import annotations

import errno
import fcntl
import fnmatch
import hashlib
import json
import math
import os
import platform
import re
import shutil
import tempfile
import time
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from nstrain import config as C
from nstrain.data.hf import hub_timeouts
from nstrain.data.work import _write_atomic

ENDPOINT = "https://huggingface.co"
DIR = "backup"
RUN_FILES = ("run.json", "wandb_run.json", "branches.json", "rollback.json")
TEMPORARY = ("*.bin.part*", "*.bin.raw*", "*.json.tmp*")     # a crashed writer's files (nstrain.data.work)
REGULAR_MAX = 10 << 20          # larger files are stored files (LFS) on the Hub; a smaller one may be a regular one
LIBRARIES = ("torch", "triton", "numpy", "tokenizers", "safetensors", "pyarrow", "wandb", "huggingface_hub", "hf_xet")
FORMAT = 2                      # 2: the data line's streams in the repository
STAGE_TRIES = 5                 # readings of the data state whose streams were released meanwhile, at most
KEY_FILE = "hf_key"             # <run_dir>/backup/hf_key: the key given with --hf-key (mode 600; never uploaded)

Digest = Tuple[int, str, Optional[str]]          # size, sha256, git blob id (files up to REGULAR_MAX)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _say(msg: str) -> None:
    print(f"{_now()} backup: {msg}", flush=True)


# ---------------------------------------------------------------------- the token, the Hub
def token(token_file: Optional[str]) -> Optional[str]:
    """The token in ``token_file``; None (no file): the host's login, huggingface_hub's own lookup."""
    if token_file is None:
        return None
    t = Path(token_file).read_text().strip()
    if not t:
        raise RuntimeError(f"{token_file} holds no token")
    return t


def _client(key: Optional[str]):
    """The backup's Hugging Face client: huggingface.co with ``key``, every request under a timeout."""
    from huggingface_hub import HfApi
    hub_timeouts()
    return HfApi(endpoint=ENDPOINT, token=key)


def hub(token_file: Optional[str]):
    return _client(token(token_file))


def account(key: str) -> str:
    """The Hugging Face account of ``key`` (the key checked)."""
    try:
        return _client(key).whoami()["name"]
    except Exception as e:
        raise RuntimeError(f"--hf-key: Hugging Face does not take the key ({type(e).__name__}: "
                           f"{str(e).replace(key, '***')[:300]})") from None


def repo_name(line: str) -> str:
    """The repository name of a line's backup: ``<line>-backup`` (letters, digits, '-', '_', '.')."""
    s = re.sub(r"[-.]{2,}", "-", re.sub(r"[^A-Za-z0-9._-]+", "-", line)).strip("-.")
    return f"{s or 'line'}-backup"


def enable(run, key: str, owner: Optional[str] = None) -> C.RunConfig:
    """``--hf-key KEY`` (nstrain new, run, restore): the line backs up with ``key``.  The key goes to
    ``<run_dir>/backup/hf_key`` (mode 600), run.json's ``backup`` names that file and the line's repository: the one it
    has, else ``<owner or the key's account>/<line>-backup``, public."""
    key = key.strip()
    if not key:
        raise ValueError("--hf-key: no key given")
    cfg = C.load(run)
    if cfg.backup is not None:
        repo, private = cfg.backup.repo, cfg.backup.private
    else:
        repo, private = f"{owner or account(key)}/{repo_name(cfg.name)}", False
    d = cfg.run_dir / DIR
    d.mkdir(exist_ok=True)
    tmp = d / (KEY_FILE + ".tmp")
    if tmp.exists():
        tmp.unlink()
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(key + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, d / KEY_FILE)
    new = C.Backup(repo=repo, token_file=str(d / KEY_FILE), private=private)
    if cfg.backup != new:                                            # (a restored line's run.json stays as it was)
        cfg.backup = new
        C.save(cfg)
    return cfg


def _tree(api, repo: str, revision: str) -> dict:
    """{path: RepoFile} of the repository at ``revision``."""
    from huggingface_hub.hf_api import RepoFile
    return {f.path: f for f in api.list_repo_tree(repo, recursive=True, revision=revision) if isinstance(f, RepoFile)}


# ---------------------------------------------------------------------- where the backup stands
def latest(cfg: C.RunConfig) -> Optional[str]:
    try:
        return (cfg.ckpt_dir / "latest").read_text().strip() or None
    except FileNotFoundError:
        return None


def record(cfg: C.RunConfig) -> Optional[dict]:
    """What the repository holds, as the last upload of this run directory left it (None: nothing uploaded)."""
    try:
        return json.loads((cfg.run_dir / DIR / "record.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _trainer_sha(cfg: C.RunConfig, name: str) -> Optional[str]:
    try:
        return hashlib.sha256((cfg.ckpt_dir / name / "trainer.json").read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def _tokens(cfg: C.RunConfig, name: str) -> Optional[float]:
    try:
        return float(json.loads((cfg.ckpt_dir / name / "trainer.json").read_text())["tokens"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def pending(cfg: C.RunConfig, newest: bool = False) -> Optional[str]:
    """The checkpoint due for the repository (``latest``, when due; see the module's doc); None when none is due or
    the line has no backup.  ``newest``: the newest checkpoint whatever its validation interval."""
    if cfg.backup is None:
        return None
    name = latest(cfg)
    if name is None:
        return None
    r = record(cfg)
    if r is None or r.get("repo") != cfg.backup.repo:
        return name                                                  # nothing of the line held
    if r.get("checkpoint") == name:
        return None if r.get("trainer_sha256") == _trainer_sha(cfg, name) else name
    t, held = _tokens(cfg, name), r.get("tokens")
    if newest or t is None or held is None or t < held:              # (t < held: the line went back)
        return name
    every = cfg.validation.every_tokens
    return name if math.floor(t / every) > math.floor(held / every) else None


def running(cfg: C.RunConfig) -> bool:
    """An upload of this line runs (its lock is held)."""
    from nstrain.proc import flock_held
    return flock_held(cfg.run_dir / DIR / "lock")


# ---------------------------------------------------------------------- hashes
def _digest(path: Path, size: int) -> Digest:
    """sha256, and the git blob id of a file up to REGULAR_MAX (one read)."""
    sha = hashlib.sha256()
    git = hashlib.sha1(b"blob %d\0" % size) if size <= REGULAR_MAX else None
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            sha.update(b)
            if git is not None:
                git.update(b)
    return size, sha.hexdigest(), git.hexdigest() if git is not None else None


def _digest_bytes(body: bytes) -> Digest:
    git = hashlib.sha1(b"blob %d\0" % len(body) + body).hexdigest() if len(body) <= REGULAR_MAX else None
    return len(body), hashlib.sha256(body).hexdigest(), git


def _blake2b(path: Path) -> str:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


class _Hashes:
    """The digests of the files sent before, by device, inode, size and modification time (the checkpoint's
    tensors are hard links: the same inode as in the checkpoint), so a file unchanged since is not read again.  Saved
    with the files of the last upload only."""

    def __init__(self, path: Path):
        self.path = path
        try:
            self.known = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            self.known = {}
        self.used: Dict[str, list] = {}

    def get(self, p: Path) -> Digest:
        st = os.stat(p)
        key = f"{st.st_dev}:{st.st_ino}"
        e = self.known.get(key)
        if e is None or e[0] != st.st_size or e[1] != st.st_mtime_ns:
            size, sha, git = _digest(p, st.st_size)
            e = [size, st.st_mtime_ns, sha, git]
        self.used[key] = e
        return e[0], e[2], e[3]

    def put(self, p: Path, sha256: str) -> Digest:
        """A sha256 computed by the upload itself; the git blob id of a small file computed here."""
        st = os.stat(p)
        git = _digest(p, st.st_size)[2] if st.st_size <= REGULAR_MAX else None
        self.used[f"{st.st_dev}:{st.st_ino}"] = [st.st_size, st.st_mtime_ns, sha256, git]
        return st.st_size, sha256, git

    def save(self) -> None:
        _write_atomic(self.path, json.dumps(self.used))


def _matches(f, d: Digest) -> bool:
    """The repository's file ``f`` (a RepoFile) holds the content of digest ``d``."""
    size, sha, git = d
    if f.size != size:
        return False
    if f.lfs is not None:
        return f.lfs.sha256 == sha
    return git is not None and f.blob_id == git


# ---------------------------------------------------------------------- an upload
class Gone(RuntimeError):
    """The checkpoint was removed (retention) before it was staged."""


def _temporary(name: str) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in TEMPORARY)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _readme(cfg: C.RunConfig) -> str:
    b = cfg.backup
    return (f"# {cfg.name}\n\nA checkpoint of the training line `{cfg.name}` and what continuing it needs: its "
            "configuration, its data line (state and streams), its held-out streams and its tokenizer.  Written by "
            "`nstrain backup` at every validation, each time replacing the content; `BACKUP.json` lists what it "
            "holds.  To continue the line on another host (the same paths free there):\n\n"
            f"```\nnstrain restore {b.repo} --hf-key <key>\nnstrain run {cfg.run_dir}\n```\n")


def _streams(work: Path, st: dict) -> Tuple[List[str], List[str]]:
    """The streams the data line in state ``st`` reads (paths in the data line): every open file's and every prepared
    cycle's (named by its permanent manifest); and what goes with them where present: each stream's meta, each
    prepared cycle's manifest.json."""
    bins, cycles = {p["bin"] for ps in st["pool"].values() for p in ps}, []
    for k, v in st["cycles"].items():
        if v["status"] == "prepared":
            man = json.loads((work / "manifests" / f"cycle_{int(k):05d}.json").read_text())
            bins |= {p["bin"] for ps in man["sources"].values() for p in ps}
            cycles.append(int(k))
    names = []
    for b in sorted(bins):
        try:
            names.append(Path(b).relative_to(work).as_posix())
        except ValueError:
            raise RuntimeError(f"{b}: a stream outside the data line {work}") from None
    extra = [n[:-len(".bin")] + ".json" for n in names if n.endswith(".bin")]
    return names, extra + [f"cycles/cycle_{k:05d}/manifest.json" for k in sorted(cycles)]


def _data_state(work: Path, stage: Path) -> Tuple[dict, Dict[str, Path]]:
    """The data line's state (copied into ``stage``) and its streams (``_streams``) as {path in the data line: local
    file}: hard-linked into ``stage`` (the preparer may release them during the upload: the link keeps the bytes), or
    read where they are (another filesystem).  A stream the state names that is gone (released after the state was
    read): the state is read again."""
    for _ in range(STAGE_TRIES):
        for sub in ("pool", "cycles"):
            shutil.rmtree(stage / "data" / sub, ignore_errors=True)
        _copy(work / "state.json", stage / "data/state.json")
        st = json.loads((stage / "data/state.json").read_text())
        names, extra = _streams(work, st)
        out, gone = {}, []
        for n in names + extra:
            src, dst = work / n, stage / "data" / n
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, dst)
                out[n] = dst
            except FileNotFoundError:
                if n in names:
                    gone.append(n)
            except OSError as e:
                if e.errno != errno.EXDEV:
                    raise
                if src.exists():
                    out[n] = src
                elif n in names:
                    gone.append(n)
        if not gone:
            return st, out
    raise RuntimeError(f"{work}: streams its state names were gone {STAGE_TRIES} times ({', '.join(gone[:3])})")


def _stage(cfg: C.RunConfig, name: str, stage: Path) -> Tuple[Dict[str, Path], dict, dict, Dict[str, str]]:
    """{path in the repository: local file} of an upload of checkpoint ``name``, its trainer.json, the data line's
    state, the links ({path: the path of the same file}: a stream under several names, uploaded once).  Copied into
    ``stage``: what is rewritten while the line runs (trainer.json, the run's files, state.json -- first: it names only
    manifests already written); hard-linked: the checkpoint's tensors (retention may remove the checkpoint meanwhile,
    the link keeps the bytes) and the data line's streams (``_data_state``); read where they are: files written once
    (manifests, listings, held-out streams, the tokenizer)."""
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    ck, base = cfg.ckpt_dir / name, f"run/checkpoints/{name}"
    try:
        _copy(ck / "trainer.json", stage / base / "trainer.json")
        meta = json.loads((stage / base / "trainer.json").read_text())
        for f in meta["files"]:
            try:
                os.link(ck / f, stage / base / f)
            except OSError as e:
                if e.errno != errno.EXDEV:
                    raise
                _copy(ck / f, stage / base / f)                       # (checkpoints on another filesystem)
    except FileNotFoundError:
        raise Gone(name) from None
    for f, h in meta["files"].items():                               # a damaged checkpoint is never sent
        got = _blake2b(stage / base / f)
        if got != h:
            raise RuntimeError(f"{ck / f}: blake2b {got}, trainer.json says {h}: not uploaded")
    files = {f"{base}/{f}": stage / base / f for f in [*meta["files"], "trainer.json"]}
    (stage / "run/checkpoints/latest").write_text(name + "\n")
    files["run/checkpoints/latest"] = stage / "run/checkpoints/latest"
    for f in RUN_FILES:
        if (cfg.run_dir / f).exists():
            _copy(cfg.run_dir / f, stage / "run" / f)
            files[f"run/{f}"] = stage / "run" / f
    work = Path(cfg.data_work)
    st, streams = _data_state(work, stage)
    files["data/state.json"] = stage / "data/state.json"
    (stage / "data/progress.json").write_text(json.dumps({"cycle": int(meta["data"]["cycle"]), "done": False}))
    files["data/progress.json"] = stage / "data/progress.json"
    for sub in ("manifests", "listings"):
        if (work / sub).is_dir():
            files.update({f"data/{sub}/{p.name}": p for p in sorted((work / sub).glob("*.json"))})
    same: Dict[Tuple[int, int], List[str]] = {}                      # the names of one file (hard links)
    for n, p in streams.items():
        s = os.stat(p)
        same.setdefault((s.st_dev, s.st_ino), []).append(n)
    links = {}
    for ns in same.values():
        ns.sort(key=lambda n: (not n.startswith("pool/"), n))        # an open file's name: it stays the longest
        files[f"data/{ns[0]}"] = streams[ns[0]]
        links.update({f"data/{n}": f"data/{ns[0]}" for n in ns[1:]})
    val = Path(cfg.val_work) / "val"
    if val.is_dir():
        files.update({f"val/{p.relative_to(val).as_posix()}": p for p in sorted(val.rglob("*"))
                      if p.is_file() and not _temporary(p.name)})
    tok = Path(st["tokenizer"])
    files[f"tokenizer/{tok.name}"] = tok
    (stage / "README.md").write_text(_readme(cfg))
    files["README.md"] = stage / "README.md"
    return files, meta, st, links


def _check_owner(cfg: C.RunConfig, revision: str, remote: dict, stage: Path) -> None:
    """The repository is this line's (its BACKUP.json names this line and run directory) or holds nothing a backup
    would replace (but .gitattributes and a README)."""
    repo = cfg.backup.repo
    if "BACKUP.json" not in remote:
        foreign = sorted(set(remote) - {".gitattributes", "README.md"})
        if foreign:
            more = " ..." if len(foreign) > 5 else ""
            raise RuntimeError(f"{repo} holds files no backup wrote ({', '.join(foreign[:5])}{more}): an upload "
                               "replaces a repository's whole content; give the line a new or empty repository "
                               "(backup.repo)")
        return
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo, "BACKUP.json", revision=revision, local_dir=stage / "remote", endpoint=ENDPOINT,
                        token=token(cfg.backup.token_file))
    old = json.loads(Path(p).read_text())
    if old.get("line") != cfg.name or old.get("run_dir") != str(cfg.run_dir):
        raise RuntimeError(f"{repo} holds the backup of the line {old.get('line')} ({old.get('run_dir')}): each line "
                           "needs its own repository (backup.repo)")


def _versions() -> dict:
    out = {}
    for k in LIBRARIES:
        try:
            out[k] = metadata.version(k)
        except metadata.PackageNotFoundError:
            out[k] = None
    return out


def _write_record(cfg: C.RunConfig, rec: dict) -> None:
    _write_atomic(cfg.run_dir / DIR / "record.json", json.dumps(rec, indent=1))


def upload(cfg: C.RunConfig, name: str, say=_say) -> dict:
    """Make the repository hold checkpoint ``name`` and what a resume from it needs (see the module documentation);
    returns the record written.  Raises Gone when the checkpoint was removed before it was staged."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    b, d = cfg.backup, cfg.run_dir / DIR
    stage, t0 = d / "stage", time.time()
    try:
        files, meta, st, links = _stage(cfg, name, stage)
        api = hub(b.token_file)
        api.create_repo(b.repo, private=b.private, exist_ok=True)
        head = api.repo_info(b.repo).sha
        remote = _tree(api, b.repo, head)
        _check_owner(cfg, head, remote, stage)
        hashes = _Hashes(d / "hashes.json")
        ops, digests, sent = [], {}, 0
        for path, local in sorted(files.items()):
            if path in remote:
                dg = hashes.get(local)
                if _matches(remote[path], dg):                        # the repository holds it: not sent again
                    digests[path] = dg
                    continue
            op = CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(local))
            digests[path] = hashes.put(local, op.upload_info.sha256.hex())
            sent += digests[path][0]
            ops.append(op)
        rec = {"format": FORMAT, "line": cfg.name, "run_dir": str(cfg.run_dir), "checkpoint": name,
               "update": meta["update"], "tokens": meta["tokens"], "time": _now(),
               "data_work": cfg.data_work, "val_work": cfg.val_work, "data_line": st["line"],
               "data": {k: meta["data"][k] for k in ("cycle", "sample")},
               "tokenizer": {"path": st["tokenizer"], "sha256": st["tokenizer_sha256"],
                             "tokenizers": st["tokenizers_version"]},
               "code": meta.get("code"), "python": platform.python_version(), "libraries": _versions(),
               "files": {p: {"size": s, "sha256": h} for p, (s, h, _) in sorted(digests.items())},
               "links": dict(sorted(links.items()))}
        body = (json.dumps(rec, indent=1) + "\n").encode()
        ops.append(CommitOperationAdd(path_in_repo="BACKUP.json", path_or_fileobj=body))
        keep = set(digests) | {"BACKUP.json", ".gitattributes"}
        ops += [CommitOperationDelete(path_in_repo=p) for p in sorted(remote) if p not in keep]
        total = sum(s for s, _, _ in digests.values())
        say(f"{name} ({meta['tokens'] / 1e9:.2f}B tokens) to {b.repo}: {len(files)} files, {total / 1e9:.2f} GB, "
            f"{sent / 1e9:.2f} GB of them to send")
        info = api.create_commit(b.repo, ops, commit_message=f"{cfg.name}: {name}, {meta['tokens'] / 1e9:.2f}B tokens",
                                 parent_commit=head)
        tree = _tree(api, b.repo, info.oid)
        want = dict(digests, **{"BACKUP.json": _digest_bytes(body)})
        extra, missing = set(tree) - set(want) - {".gitattributes"}, set(want) - set(tree)
        bad = sorted(p for p in set(want) & set(tree) if not _matches(tree[p], want[p]))
        if extra or missing or bad:
            raise RuntimeError(f"{b.repo} at {info.oid} does not hold the upload: other files {sorted(extra)[:5]}, "
                               f"missing {sorted(missing)[:5]}, other contents {bad[:5]}")
        hashes.save()
        trainer = (stage / f"run/checkpoints/{name}/trainer.json").read_bytes()
        out = {"repo": b.repo, "checkpoint": name, "update": meta["update"], "tokens": meta["tokens"],
               "trainer_sha256": hashlib.sha256(trainer).hexdigest(), "commit": info.oid, "time": rec["time"],
               "bytes": total, "sent": sent, "seconds": round(time.time() - t0), "clean": False}
        _write_record(cfg, out)
    finally:
        if stage.exists():
            shutil.rmtree(stage)                                     # (the linked checkpoint's bytes with it)
    say(f"{b.repo} holds {name} (commit {info.oid[:12]}; {sent / 1e9:.2f} GB sent in {time.time() - t0:.0f} s)")
    clean(cfg, api, say)
    return record(cfg)


def clean(cfg: C.RunConfig, api=None, say=_say) -> List[str]:
    """Squash the repository's history into its one commit, then delete the stored files no commit references any
    more (the checkpoints it held before): the repository keeps one copy.  Marks the record clean; returns the names
    of the files deleted."""
    b, rec = cfg.backup, record(cfg)
    api = api or hub(b.token_file)
    api.super_squash_history(b.repo, commit_message=f"{cfg.name}: {rec['checkpoint']}, "
                                                    f"{rec['tokens'] / 1e9:.2f}B tokens")
    head = api.repo_info(b.repo).sha
    refs = {f.lfs.sha256 for f in _tree(api, b.repo, head).values() if f.lfs is not None}
    stored = list(api.list_lfs_files(b.repo))
    unlisted = refs - {f.file_oid for f in stored}
    if unlisted:                                                     # (another form of id: delete nothing)
        raise RuntimeError(f"{b.repo}: the listing of stored files lacks {len(unlisted)} the repository references; "
                           "nothing deleted")
    gone = [f for f in stored if f.file_oid not in refs]
    if gone:
        api.permanently_delete_lfs_files(b.repo, gone, rewrite_history=False)
        say(f"deleted {len(gone)} stored files no longer referenced ({sum(f.size for f in gone) / 1e9:.2f} GB)")
    _write_record(cfg, dict(rec, commit=head, clean=True))
    return [f.filename for f in gone]


def run(run_dir, newest: bool = False) -> int:
    """``nstrain backup RUN [--latest]``: upload until no checkpoint is due (``pending``; run.json is read again before
    each upload); one upload of a line at a time."""
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()                                          # (a log file, not a terminal)
    cfg = C.load(run_dir)
    if cfg.backup is None:
        raise ValueError(f"{cfg.run_dir / C.RUN_FILE} has no backup")
    (cfg.run_dir / DIR).mkdir(exist_ok=True)
    lock = open(cfg.run_dir / DIR / "lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError(f"another upload of {cfg.name} runs") from None
    try:
        sent, gone = set(), set()
        while True:
            cfg = C.load(run_dir)
            if cfg.backup is None:
                return 0
            name = pending(cfg, newest)
            if name is None:
                rec = record(cfg)
                if rec is not None and rec.get("repo") == cfg.backup.repo and not rec.get("clean"):
                    clean(cfg)                                       # (cut off after the commit)
                return 0
            key = (cfg.backup.repo, name, _trainer_sha(cfg, name))
            if key in sent:
                raise RuntimeError(f"{name} was uploaded, yet the record does not show it")
            try:
                upload(cfg, name)
            except Gone:
                if name in gone:
                    raise RuntimeError(f"`latest` names {name}, which has no complete checkpoint directory") from None
                gone.add(name)
                _say(f"{name} was removed before it was staged; the newest checkpoint next")
                continue
            sent.add(key)
    finally:
        lock.close()


# ---------------------------------------------------------------------- restore
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _destination(path: str, rec: dict) -> Optional[Path]:
    """Where a file of the backup goes (None: README.md)."""
    for prefix, root in (("run/", Path(rec["run_dir"])), ("data/", Path(rec["data_work"])),
                         ("val/", Path(rec["val_work"]) / "val")):
        if path.startswith(prefix):
            return root / path[len(prefix):]
    if path.startswith("tokenizer/"):
        return Path(rec["tokenizer"]["path"])
    return None


def _order(path: str) -> int:
    """Placing order: the held-out streams and the tokenizer; the data line: its streams, the cycles' manifest.json
    (a cycle is visible once its streams are in place), its manifests, listings, progress, state.json; the checkpoint,
    its `latest`, the run's files, run.json last (a run directory without run.json is a restore not finished)."""
    if path.startswith("data/cycles/") and path.endswith("/manifest.json"):
        return 4
    for i, p in enumerate(("val/", "tokenizer/", "data/pool/", "data/cycles/", "", "data/manifests/", "data/listings/",
                           "data/progress.json", "data/state.json", "run/checkpoints/u", "run/checkpoints/latest")):
        if p and path.startswith(p):
            return i
    return 12 if path == "run/run.json" else 11


def _place(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(src, dst)
    except OSError as e:
        if e.errno != errno.EXDEV:
            raise
        part = dst.with_name(dst.name + ".restore-part")             # another filesystem: copied, then renamed
        shutil.copyfile(src, part)
        with open(part, "rb") as f:
            os.fsync(f.fileno())
        os.replace(part, dst)


def restore(repo: str, key: Optional[str] = None, revision: Optional[str] = None, say=_say) -> dict:
    """``nstrain restore REPO [--hf-key KEY]``: the line of a backup back at its paths (see the module documentation);
    returns BACKUP.json.  ``key``: read with it (a private repository), and the line backs up with it (``enable``);
    without it, ``nstrain run RUN --hf-key KEY`` gives the key back to the line."""
    from huggingface_hub import hf_hub_download
    key = key.strip() if key else None
    api = _client(key)
    sha = api.repo_info(repo, revision=revision).sha
    with tempfile.TemporaryDirectory() as t:
        rec = json.loads(Path(hf_hub_download(repo, "BACKUP.json", revision=sha, local_dir=t, endpoint=ENDPOINT,
                                              token=key)).read_text())
    if rec.get("format") != FORMAT:
        raise RuntimeError(f"{repo}: a backup of format {rec.get('format')}; this nstrain restores format {FORMAT} "
                           "(with the data line's streams)")
    run_dir = Path(rec["run_dir"])
    if (run_dir / C.RUN_FILE).exists():
        raise FileExistsError(f"{run_dir} holds a line: a restore puts the line {rec['line']} back at the paths its "
                              "checkpoints and data state name, into a run directory without run.json")
    links = rec["links"]
    held = dict(rec["files"], **{p: rec["files"][t] for p, t in links.items()})   # what each path holds
    dest = {p: _destination(p, rec) for p in held}
    clash = [str(q) for p, q in dest.items() if q is not None and not p.startswith("run/") and q.exists()
             and (q.stat().st_size != held[p]["size"] or _sha256(q) != held[p]["sha256"])]
    if clash:
        raise FileExistsError(f"{len(clash)} files of the backup exist here with other contents ({', '.join(clash[:5])}"
                              f"{' ...' if len(clash) > 5 else ''}): another data line?  Move them away and restore "
                              "again")
    work = Path(rec["data_work"])                                   # downloaded beside where they go (renamed there)
    tmps = {"data/": work.parent / f".{work.name}.restore", "": run_dir.parent / f".{run_dir.name}.restore"}
    say(f"{repo} at {sha[:12]}: line {rec['line']}, {rec['checkpoint']} ({rec['tokens'] / 1e9:.2f}B tokens, "
        f"{rec['time']}); downloading {sum(f['size'] for f in rec['files'].values()) / 1e9:.2f} GB")
    for prefix, tmp in tmps.items():
        tmp.mkdir(parents=True, exist_ok=True)
        if prefix:
            api.snapshot_download(repo, revision=sha, local_dir=tmp, allow_patterns=[f"{prefix}*"])
        else:
            api.snapshot_download(repo, revision=sha, local_dir=tmp, ignore_patterns=["data/*"])

    def got(p: str) -> Path:
        return tmps["data/" if p.startswith("data/") else ""] / p
    for p, f in rec["files"].items():
        q = got(p)
        if not q.is_file() or q.stat().st_size != f["size"] or _sha256(q) != f["sha256"]:
            raise RuntimeError(f"{p}: the download differs from BACKUP.json")
    for p in sorted(held, key=lambda p: (_order(p), p in links, p)):    # a link after the file it names
        q = dest[p]
        if q is None or (q.exists() and not p.startswith("run/")):
            continue
        if p in links:
            q.parent.mkdir(parents=True, exist_ok=True)
            os.link(dest[links[p]], q)
        else:
            _place(got(p), q)
    for tmp in tmps.values():
        shutil.rmtree(tmp)
    if key:
        enable(run_dir, key)
    say(f"the line is at {run_dir}, its data line at {rec['data_work']}")
    return rec


def differences(rec: dict) -> List[str]:
    """The libraries of this host that differ from the line's at the backup (the sample order depends on numpy: a
    resume with another order is refused)."""
    here = _versions()
    return [f"{k} {here.get(k)} here, {v} in the line" for k, v in sorted(rec.get("libraries", {}).items())
            if here.get(k) != v]
