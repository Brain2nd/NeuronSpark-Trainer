"""Training checkpoints: everything needed to continue a line exactly where it stopped.

A checkpoint is a directory ``<run_dir>/checkpoints/u<update:08d>/``::

    model.safetensors       fp32 parameters and buffers (the repository's state-dict keys)
    optimizer.safetensors   AdamW state per parameter name: ``<name>::exp_avg`` / ``::exp_avg_sq`` / ``::step``
    trainer.json            update, tokens, line, data progress (work dir, cycle, sample, manifest sha256, seed),
                            shape (rows, seq_len, accum), run config and run_dir, model config, code versions,
                            tokenizer sha256, W&B run, skipped updates, the files' blake2b

Writing: the tensors are copied to host memory first (the training continues meanwhile), written into
``u<N>.tmp/``, fsynced, hashed, renamed to ``u<N>/``; then ``latest`` (one line: the directory name) is replaced
atomically; the directory is fsynced after each rename.  A background write that failed is reported at the next
:meth:`poll` (after every update).  Reading checks every blake2b.

Space is bounded by the line itself: after every checkpoint written (:meth:`_retain`) only these stay --

* the checkpoint ``latest`` names;
* checkpoints marked ``keep``: the start of a line, every cycle end (the fork points of branches), every stage end;
* the newest ``keep_last`` rolling checkpoints up to ``latest``.

Everything else is removed: older rolling checkpoints, directories of updates after ``latest`` (an abandoned
continuation), the checkpoints a rollback set aside (``rolled-back-*``, once the line has written a checkpoint after
the rollback).  A directory of the same update renamed into place by a run that died before pointing ``latest`` at it
is replaced when the line reaches that update again.

Recovery at start (:meth:`recover`): directories of interrupted writes (``*.tmp``) are removed; a root whose only
complete checkpoint is not yet named by ``latest`` (its first write cut off between the rename and ``latest``) gets
``latest`` pointed at it.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file, save_file


def _blake2b(path: Path) -> str:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _fsync(path: Path) -> None:
    """fsync a file or a directory."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def snapshot(model, opt_state: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[Dict[str, torch.Tensor],
                                                                             Dict[str, torch.Tensor]]:
    """Host copies of the model state and the optimizer state (flattened ``<name>::<key>``)."""
    ms = {k: v.detach().to("cpu", copy=True).contiguous() for k, v in model.state_dict().items()}
    os_ = {f"{n}::{k}": v.detach().to("cpu", copy=True).contiguous() for n, st in opt_state.items()
           for k, v in st.items()}
    return ms, os_


class Checkpoints:
    def __init__(self, root: str, keep_last: int = 3):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self._writer: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None

    # ------------------------------------------------------------------ write
    def save(self, update: int, model_state: Dict[str, torch.Tensor], opt_state: Dict[str, torch.Tensor],
             meta: dict, keep: bool = False, background: bool = True) -> None:
        """Write a host snapshot (see :func:`snapshot`); in a background thread unless ``background`` is False."""
        self.wait()
        args = (update, model_state, opt_state, dict(meta, keep=bool(keep)))
        if not background:
            self._write(*args)
            return
        self._writer = threading.Thread(target=self._run, args=args, daemon=False)
        self._writer.start()

    def _run(self, *args) -> None:
        try:
            self._write(*args)
        except BaseException as e:
            self._error = e

    def poll(self) -> None:
        """Re-raise the error of a finished background write (non-blocking)."""
        if self._writer is not None and not self._writer.is_alive():
            self.wait()

    def wait(self) -> None:
        """Wait for the background write; re-raise its error."""
        if self._writer is not None:
            self._writer.join()
            self._writer = None
        if self._error is not None:
            e, self._error = self._error, None
            raise RuntimeError("checkpoint write failed") from e

    def _write(self, update: int, model_state, opt_state, meta: dict) -> None:
        name = f"u{update:08d}"
        tmp, final = self.root / f"{name}.tmp", self.root / name
        if final.exists():
            cur = self.root / "latest"
            if cur.exists() and cur.read_text().strip() == name:
                raise FileExistsError(final)
            # renamed into place by a run that died before pointing ``latest`` at it: the line resumed from an
            # earlier checkpoint and reached this update again, so that directory is superseded by this write
            shutil.rmtree(final)
            print(f"checkpoint {name}: replaced an earlier directory of this update that `latest` never named",
                  flush=True)
        if tmp.exists():
            shutil.rmtree(tmp)                                       # a write interrupted before its rename
        tmp.mkdir()
        save_file(model_state, str(tmp / "model.safetensors"))
        save_file(opt_state, str(tmp / "optimizer.safetensors"))
        files = {}
        for f in ("model.safetensors", "optimizer.safetensors"):
            _fsync(tmp / f)
            files[f] = _blake2b(tmp / f)
        (tmp / "trainer.json").write_text(json.dumps(dict(meta, files=files), indent=1))
        _fsync(tmp / "trainer.json")
        os.replace(tmp, final)
        _fsync(self.root)
        self._point_latest(name)
        self._retain()

    def _point_latest(self, name: str) -> None:
        lt = self.root / "latest.tmp"
        lt.write_text(name + "\n")
        _fsync(lt)
        os.replace(lt, self.root / "latest")
        _fsync(self.root)

    def keep(self, update: int) -> None:
        """Mark the checkpoint of ``update`` kept (retention never removes it); trainer.json is rewritten atomically
        (it is not among the hashed files)."""
        d = self.root / f"u{update:08d}"
        meta = json.loads((d / "trainer.json").read_text())
        if meta.get("keep"):
            return
        tmp = d / "trainer.json.tmp"
        tmp.write_text(json.dumps(dict(meta, keep=True), indent=1))
        _fsync(tmp)
        os.replace(tmp, d / "trainer.json")
        _fsync(d)

    def _retain(self) -> None:
        """After a checkpoint was written and ``latest`` names it (no other writer): remove what the line no longer
        needs (see the module documentation)."""
        cur = self.latest()
        if cur is None:
            return
        rolling = []
        for d in sorted(self.root.glob("u[0-9]*")):
            if d.suffix == ".tmp" or d == cur:
                continue
            if d.name > cur.name:                                    # after latest: an abandoned continuation
                shutil.rmtree(d)
                continue
            if not (d / "trainer.json").exists():
                continue
            if not json.loads((d / "trainer.json").read_text()).get("keep"):
                rolling.append(d)
        for d in rolling[:max(0, len(rolling) - (self.keep_last - 1))]:
            shutil.rmtree(d)
        for d in self.root.glob("rolled-back-*"):                    # the line has moved on after a rollback
            shutil.rmtree(d)

    # ------------------------------------------------------------------ read
    def latest(self) -> Optional[Path]:
        p = self.root / "latest"
        return self.root / p.read_text().strip() if p.exists() else None

    def complete(self) -> List[Path]:
        """The checkpoint directories whose write finished (renamed into place, with trainer.json)."""
        return [d for d in sorted(self.root.glob("u[0-9]*")) if d.suffix != ".tmp" and (d / "trainer.json").exists()]

    def recover(self, owner: Path) -> Optional[Path]:
        """Clean up after a writer that died (call with the line's lock held, before any write); returns ``latest``.
        Raises when complete checkpoints exist that ``latest`` does not name and they are not the first checkpoint of
        the line whose run directory is ``owner``."""
        for d in self.root.glob("u[0-9]*.tmp"):
            shutil.rmtree(d)
        cur = self.latest()
        if cur is not None:
            return cur
        done = self.complete()
        if not done:
            return None
        if len(done) == 1 and Path(json.loads((done[0] / "trainer.json").read_text())["run_dir"]) == Path(owner):
            self.verify(done[0])
            self._point_latest(done[0].name)
            print(f"checkpoint {done[0].name}: the line's first checkpoint, written before a crash that came ahead "
                  f"of `latest`; `latest` now names it", flush=True)
            return done[0]
        raise RuntimeError(f"{self.root} holds checkpoints {[d.name for d in done]} but no `latest`: write the name "
                           "of the newest complete checkpoint into it (or move the directories away) before training "
                           "this line")

    @staticmethod
    def verify(path: Path) -> dict:
        """The checkpoint's trainer.json after checking every file's blake2b."""
        meta = json.loads((path / "trainer.json").read_text())
        for f, h in meta["files"].items():
            got = _blake2b(path / f)
            if got != h:
                raise RuntimeError(f"{path / f}: blake2b {got} != {h}")
        return meta

    @staticmethod
    def load(path: Path, verify: bool = True) -> Tuple[dict, Dict[str, torch.Tensor],
                                                       Dict[str, Dict[str, torch.Tensor]]]:
        meta = Checkpoints.verify(path) if verify else json.loads((path / "trainer.json").read_text())
        ms = load_file(str(path / "model.safetensors"))
        flat = load_file(str(path / "optimizer.safetensors"))
        opt: Dict[str, Dict[str, torch.Tensor]] = {}
        for k, v in flat.items():
            n, key = k.rsplit("::", 1)
            opt.setdefault(n, {})[key] = v
        return meta, ms, opt
