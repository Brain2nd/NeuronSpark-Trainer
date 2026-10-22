"""Hugging Face side: pinned revisions, the file listing of a source at its revision (cached), the training order of
its files (fixed shuffle per epoch, the held-out file excluded), and single-file downloads at the pinned revision."""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Dict, List

from nstrain.data.sources import Source

DATA_EXT = (".parquet", ".jsonl")


def pin(repo: str) -> str:
    """The dataset repository's current commit sha (the run then reads only this revision)."""
    from huggingface_hub import HfApi
    return HfApi().dataset_info(repo).sha


def list_files(src: Source, cache_dir: Path, revision: str) -> List[Dict]:
    """[{"path", "size"}] of the source's data files at ``revision``; cached as JSON per revision."""
    key = f"{src.repo.replace('/', '__')}__{src.prefix.replace('/', '__')}__{revision[:12]}.json"
    cache = cache_dir / key
    if cache.exists():
        return json.loads(cache.read_text())
    from huggingface_hub import HfApi
    files = []
    for e in HfApi().list_repo_tree(src.repo, repo_type="dataset", path_in_repo=src.prefix, recursive=True,
                                    revision=revision):
        if getattr(e, "size", None) is not None and e.path.endswith(DATA_EXT):
            files.append({"path": e.path, "size": int(e.size)})
    files.sort(key=lambda f: f["path"])
    if not files:
        raise RuntimeError(f"no data files under {src.repo}/{src.prefix}@{revision}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(files))
    return files


def held_out(files: List[Dict], name: str) -> Dict:
    """The source's validation file: the one whose path hashes lowest (fixed, independent of the listing order)."""
    return min(files, key=lambda f: hashlib.blake2b(f"{name}:{f['path']}".encode(), digest_size=8).digest())


def order(files: List[Dict], name: str, epoch: int, seed: int = 2026) -> List[Dict]:
    """The source's training files in consumption order: a fixed shuffle per (source, epoch), the validation file
    excluded."""
    val = held_out(files, name)["path"]
    out = [f for f in files if f["path"] != val]
    random.Random(f"{seed}:{name}:{epoch}").shuffle(out)
    return out


def download(src: Source, path: str, dest_dir: Path, revision: str, retries: int = 20) -> Path:
    """One file at ``revision`` into ``dest_dir`` (HF_ENDPOINT honoured); retried with back-off."""
    from huggingface_hub import hf_hub_download
    last = None
    for i in range(retries):
        try:
            return Path(hf_hub_download(src.repo, filename=path, repo_type="dataset", revision=revision,
                                        local_dir=str(dest_dir)))
        except Exception as e:                                       # network / 5xx: back off and retry
            last = e
            time.sleep(min(300, 5 * 2 ** i) + random.random() * 5)
    raise RuntimeError(f"download failed after {retries} tries: {src.repo}/{path}@{revision}: {last}")
