"""One downloaded file -> a uint32 token stream (documents concatenated, each followed by EOS); the file is then
deleted.

Output per input file: ``<out>.bin`` (uint32 tokens) and ``<out>.json`` (provenance: repo / path / revision / epoch;
tokens, documents, text bytes, blake2b of the .bin, seconds).  The tokenizer is a ``tokenizer.json`` (Hugging Face
``tokenizers``, batch encoding on RAYON threads; deterministic for a given tokenizer file and library version).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np

_TOK = {}


def _tokenizer(path: str):
    if path not in _TOK:
        from tokenizers import Tokenizer
        _TOK[path] = Tokenizer.from_file(path)
    return _TOK[path]


def _texts(path: Path, fields: Sequence[str], batch_rows: int) -> Iterator[List[str]]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(path))
        for rb in pf.iter_batches(columns=list(fields), batch_size=batch_rows):
            cols = [rb.column(f).to_pylist() for f in fields]
            if len(cols) == 1:
                yield [x or "" for x in cols[0]]
            else:
                yield ["\n\n".join(x for x in row if x) for row in zip(*cols)]
    else:                                                            # jsonl
        buf: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                buf.append("\n\n".join(str(r[k]) for k in fields if r.get(k)))
                if len(buf) == batch_rows:
                    yield buf
                    buf = []
        if buf:
            yield buf


def tokenize_file(path: str, fields: Sequence[str], tokenizer_json: str, eos_id: int, out_prefix: str,
                  provenance: dict, batch_rows: int = 4096, delete: bool = True) -> dict:
    t0 = time.time()
    tok = _tokenizer(tokenizer_json)
    src = Path(path)
    out_bin = Path(out_prefix + ".bin")
    tmp = out_bin.with_suffix(f".bin.part{os.getpid()}")       # a leftover writer of the same file never interleaves
    digest = hashlib.blake2b(digest_size=16)
    n_tok = n_doc = n_bytes = 0
    with open(tmp, "wb") as f:
        for texts in _texts(src, fields, batch_rows):
            texts = [t for t in texts if t]
            if not texts:
                continue
            parts = []
            for e in tok.encode_batch(texts, add_special_tokens=False):
                parts.append(np.asarray(e.ids, dtype=np.uint32))
                parts.append(np.asarray([eos_id], dtype=np.uint32))
            arr = np.concatenate(parts)
            b = arr.tobytes()
            f.write(b)
            digest.update(b)
            n_tok += int(arr.size)
            n_doc += len(texts)
            n_bytes += sum(len(t.encode("utf-8")) for t in texts)
    os.replace(tmp, out_bin)
    meta = dict(provenance, file_bytes=src.stat().st_size, tokens=n_tok, docs=n_doc, text_bytes=n_bytes,
                blake2b=digest.hexdigest(), seconds=round(time.time() - t0, 2))
    Path(out_prefix + ".json").write_text(json.dumps(meta))
    if delete:
        src.unlink()
    return meta


if __name__ == "__main__":
    # one job per process (``Work._fetch`` runs these as subprocesses): argv[1] = JSON of tokenize_file's arguments;
    # the last stdout line is the meta
    import signal
    from nstrain.proc import PREPARER, die_with_parent
    die_with_parent(PREPARER, signal.SIGKILL)       # a killed preparer leaves no writer behind
    job = json.loads(sys.argv[1])
    print(json.dumps(tokenize_file(**job)), flush=True)
