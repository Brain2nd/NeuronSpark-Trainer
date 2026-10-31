"""One downloaded file -> a uint32 token stream (documents in a seeded random order, each followed by EOS); the file
is then deleted.

The documents of a corpus file are often grouped (by site, forum thread, crawl segment, language ...); the stream holds
them in a random order fixed by ``doc_seed``, so any stretch of it -- a training window, the part of the file a cycle
takes (nstrain.data.work), a validation row -- is a uniform sample of the file's documents.

The documents are plain text (TEXT): a token of the tokenizer's added vocabulary written in a document ("</s>",
"<param", "/think", "<think>", "<unused_token_3>" ... in code, markup and paths) encodes as its characters, never as
that token -- every added token is marked special and tokenizers' ``encode_special_tokens`` set, so none is matched in
a text; the only added token of a stream is the EOS added after each document.  A source's literal row markers
(``markers``, e.g. "<s>" / "</s>" around each row) are removed first: they are the corpus's framing, not text.

Held-out documents: a document whose text hashes below ``holdout["share"]`` (blake2b of ``holdout["salt"]`` and the
text; the same in every file, revision, epoch and line) is validation data.  A training stream (``keep`` "train")
leaves it out; a validation stream (``keep`` "val") holds only such documents -- of the whole file, shuffled, then
the first ones up to ``max_tokens``.

Output per input file: ``<out>.bin`` (uint32 tokens) and ``<out>.json`` (provenance: repo / path / revision / epoch /
doc_seed, text; tokens, documents, text bytes, added-token ids in the encoded text (0 with plain text), blake2b of
the .bin, seconds), both fsynced before the file's download is deleted: a crash of the host leaves a stream whole or
absent.  The tokenizer is a ``tokenizer.json`` (Hugging Face ``tokenizers``, batch encoding on RAYON threads;
deterministic for a given tokenizer file and library version).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator, List, Sequence

TEXT = "added-tokens-as-text"   # how documents are encoded (in every stream's meta and validation index; see above)
_TOK = {}


def _tokenizer(path: str):
    """The tokenizer as documents are encoded (TEXT): its added tokens all special, none matched in a text (their ids
    are unchanged; the file itself is not touched)."""
    if path not in _TOK:
        from tokenizers import Tokenizer
        spec = json.loads(Path(path).read_text())
        for a in spec.get("added_tokens", []):
            a["special"] = True
        t = Tokenizer.from_str(json.dumps(spec))
        t.encode_special_tokens = True
        _TOK[path] = t
    return _TOK[path]


def _fsync(path, directory: bool = False) -> None:
    fd = os.open(path, os.O_RDONLY | (os.O_DIRECTORY if directory else 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


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
                if not line.strip():
                    continue
                r = json.loads(line)
                buf.append("\n\n".join(str(r[k]) for k in fields if r.get(k)))
                if len(buf) == batch_rows:
                    yield buf
                    buf = []
        if buf:
            yield buf


def seed_words(text: str) -> list:
    """A seed for numpy's generators from a string (four 32-bit words of its blake2b)."""
    d = hashlib.blake2b(text.encode(), digest_size=16).digest()
    return [int.from_bytes(d[i:i + 4], "little") for i in range(0, 16, 4)]


def _unmark(t: str, markers: Sequence[str]) -> str:
    begin, end = markers
    t = t.strip()
    if begin and t.startswith(begin):
        t = t[len(begin):]
    if end and t.endswith(end):
        t = t[:-len(end)]
    return t.strip()


def held(text: str, holdout: dict) -> bool:
    """A held-out (validation) document: its text's blake2b with the salt below ``share`` of the hash range."""
    h = hashlib.blake2b(holdout["salt"].encode() + text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") < holdout["share"] * 2.0 ** 64


def tokenize_file(path: str, fields: Sequence[str], tokenizer_json: str, eos_id: int, out_prefix: str,
                  provenance: dict, doc_seed: str, holdout: dict, keep: str = "train",
                  markers: Sequence[str] = ("", ""), max_tokens: int = 0, batch_rows: int = 4096,
                  delete: bool = True) -> dict:
    import numpy as np                                               # (not at import: work.py imports TEXT)
    if keep not in ("train", "val"):
        raise ValueError(f"keep must be 'train' or 'val', got {keep!r}")
    t0 = time.time()
    tok = _tokenizer(tokenizer_json)
    special = np.asarray(sorted(tok.get_added_tokens_decoder()), dtype=np.uint32)      # every added token's id
    n_special = 0
    src = Path(path)
    out_bin = Path(out_prefix + ".bin")
    raw = out_bin.with_suffix(f".bin.raw{os.getpid()}")       # file order; a leftover writer never interleaves
    tmp = out_bin.with_suffix(f".bin.part{os.getpid()}")
    try:                                                             # a failed file leaves no temporaries
        lens: List[int] = []
        n_bytes = n_other = 0
        with open(raw, "wb") as f:
            for texts in _texts(src, fields, batch_rows):
                if any(markers):
                    texts = [_unmark(t, markers) for t in texts]
                texts = [t for t in texts if t]
                mine = [t for t in texts if held(t, holdout) == (keep == "val")]
                n_other += len(texts) - len(mine)
                texts = mine
                if not texts:
                    continue
                parts = []
                for e in tok.encode_batch(texts, add_special_tokens=False):
                    ids = np.asarray(e.ids, dtype=np.uint32)
                    n_special += int(np.isin(ids, special).sum())
                    parts.append(ids)
                    parts.append(np.asarray([eos_id], dtype=np.uint32))
                    lens.append(len(e.ids) + 1)
                f.write(np.concatenate(parts).tobytes())
                n_bytes += sum(len(t.encode("utf-8")) for t in texts)
        size = np.asarray(lens, dtype=np.int64)
        first = np.concatenate(([0], np.cumsum(size)[:-1])) if lens else size
        order = np.random.default_rng(seed_words(doc_seed)).permutation(len(lens))
        all_tokens = int(size.sum())
        if max_tokens:                                                   # the first shuffled documents up to it
            order = order[:int(np.searchsorted(np.cumsum(size[order]), max_tokens, side="right")) + 1]
        n_doc, n_tok = int(order.size), int(size[order].sum())
        digest = hashlib.blake2b(digest_size=16)
        with open(tmp, "wb") as f:
            if n_tok:
                mm = np.memmap(raw, dtype=np.uint32, mode="r")
                for chunk in np.array_split(order, max(1, n_doc // 4096)):
                    b = np.concatenate([mm[first[d]:first[d] + size[d]] for d in chunk]).tobytes()
                    f.write(b)
                    digest.update(b)
                del mm
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out_bin)
        os.unlink(raw)
    except BaseException:
        for x in (raw, tmp):
            if x.exists():
                x.unlink()
        raise
    meta = dict(provenance, doc_seed=doc_seed, keep=keep, holdout=holdout, text=TEXT, file_bytes=src.stat().st_size,
                tokens=n_tok, tokens_before_cap=all_tokens, docs=n_doc, other_docs=n_other, text_bytes=n_bytes,
                added_in_text=n_special,
                blake2b=digest.hexdigest(), seconds=round(time.time() - t0, 2))
    j = Path(out_prefix + ".json")
    jt = j.with_name(f"{j.name}.tmp{os.getpid()}")
    with open(jt, "w") as f:
        f.write(json.dumps(meta))
        f.flush()
        os.fsync(f.fileno())
    os.replace(jt, j)
    _fsync(j.parent, directory=True)                                 # the stream and its meta, durably
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
