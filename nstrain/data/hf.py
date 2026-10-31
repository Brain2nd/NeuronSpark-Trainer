"""Hugging Face side: pinned revisions, the file listing of a source at its revision (cached), the source's strata
(its subdirectories: languages, crawl dumps, L3 sub-types ...; or its files, ``file_strata``), the order of a
stratum's files (fixed shuffle per epoch), and single-file downloads at the pinned revision.  Validation data are
held-out documents of the files (nstrain.data.tokenize.held), not held-out files."""
from __future__ import annotations

import json
import os
import random
import time
from itertools import accumulate
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
    tmp = cache.with_name(f"{cache.name}.tmp{os.getpid()}")          # a cut write never becomes the listing
    tmp.write_text(json.dumps(files))
    os.replace(tmp, cache)
    return files


def stratum_of(src: Source, path: str) -> str:
    """The stratum of a file: its directory below the source's prefix ("" for a flat source), or the file itself
    (its name without extension) for a source whose files are kinds of data (``file_strata``)."""
    rel = path[len(src.prefix):].lstrip("/") if path.startswith(src.prefix) else path
    if src.file_strata:
        return rel.rsplit(".", 1)[0]
    return rel.rsplit("/", 1)[0] if "/" in rel else ""


def largest_remainder(total: int, weights: Dict[str, float]) -> Dict[str, int]:
    """``total`` split over the keys in proportion to ``weights`` (integers summing to ``total``; ties by key)."""
    w = sum(weights.values())
    if total <= 0 or w <= 0:
        return {k: 0 for k in weights}
    exact = {k: total * v / w for k, v in weights.items()}
    out = {k: int(x) for k, x in exact.items()}
    left = total - sum(out.values())
    for k in sorted(exact, key=lambda k: (out[k] - exact[k], k))[:left]:
        out[k] += 1
    return out


def strata(src: Source, files: List[Dict], max_strata: int = 16) -> Dict[str, List[Dict]]:
    """The source's files by stratum, each list in path order.  A stratum is a subdirectory, or a file
    (``file_strata``: kinds of data, never merged; ``max_strata`` 1 makes them one stratum); more than ``max_strata``
    subdirectories (crawl dumps) are merged, adjacent ones in path order (neighbouring dumps), into ``max_strata``
    groups of about equal size, named "<first>..<last>"."""
    by: Dict[str, List[Dict]] = {}
    for f in sorted(files, key=lambda f: f["path"]):
        by.setdefault(stratum_of(src, f["path"]), []).append(f)
    names = sorted(by)
    if src.file_strata:
        return by if max_strata > 1 else {"": sorted(files, key=lambda f: f["path"])}
    if len(names) <= max_strata:
        return by
    size = [float(sum(f["size"] for f in by[t])) for t in names]
    cum = list(accumulate(size))
    group = [min(int((c - 0.5 * x) / cum[-1] * max_strata), max_strata - 1) for c, x in zip(cum, size)]
    out: Dict[str, List[Dict]] = {}
    for g in sorted(set(group)):
        members = [t for t, x in zip(names, group) if x == g]
        out[f"{members[0]}..{members[-1]}" if len(members) > 1 else members[0]] = [f for t in members for f in by[t]]
    return out


def order(files: List[Dict], key: str, epoch: int, seed: int = 2026) -> List[Dict]:
    """A stratum's training files in consumption order: a fixed shuffle per (source/stratum, epoch)."""
    out = sorted(files, key=lambda f: f["path"])
    random.Random(f"{seed}:{key}:{epoch}").shuffle(out)
    return out


def download(src: Source, path: str, dest_dir: Path, revision: str, retries: int = 20, stop=None) -> Path:
    """One file at ``revision`` into ``dest_dir`` (HF_ENDPOINT honoured); retried with back-off.  ``stop`` (a
    threading.Event): set, no further try starts (a try under way runs to its end)."""
    from huggingface_hub import hf_hub_download
    last = None
    for i in range(retries):
        if stop is not None and stop.is_set():
            raise RuntimeError(f"download of {src.repo}/{path} stopped")
        try:
            return Path(hf_hub_download(src.repo, filename=path, repo_type="dataset", revision=revision,
                                        local_dir=str(dest_dir)))
        except Exception as e:                                       # network / 5xx: back off and retry
            last = e
            pause = min(300, 5 * 2 ** i) + random.random() * 5
            if stop is not None:
                stop.wait(pause)
            else:
                time.sleep(pause)
    raise RuntimeError(f"download failed after {retries} tries: {src.repo}/{path}@{revision}: {last}")
