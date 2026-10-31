"""Child processes that end with their parent: the trainer and the preparer with the supervisor, the tokenizers with
the preparer, a tune's measurements with the tune (Linux PR_SET_PDEATHSIG; the parent's pid comes in
an environment variable, so a child started after its parent died exits at once).  ``flock_held``: whether another
process holds a lock file, without taking the lock (a probe that took it for a moment would make a real acquirer at
that instant fail).  ``honor_stop``: a line's STOP file renamed when honored."""
from __future__ import annotations

import fcntl
import os
import signal
import sys
from pathlib import Path

SUPERVISOR = "NSTRAIN_SUPERVISOR_PID"
PREPARER = "NSTRAIN_PREPARER_PID"
TUNE = "NSTRAIN_TUNE_PID"


def die_with_parent(var: str, sig: int = signal.SIGTERM) -> None:
    parent = os.environ.get(var)
    if parent is None or not sys.platform.startswith("linux"):
        return
    import ctypes
    if ctypes.CDLL(None, use_errno=True).prctl(1, int(sig), 0, 0, 0) != 0:    # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
    if os.getppid() != int(parent):
        raise SystemExit(f"the parent process {parent} exited before this one started")


DEVICE_FS = ("ext2", "ext3", "ext4", "xfs", "tmpfs", "f2fs", "jfs")   # st_dev is the device /proc/locks names
                                                                    # (not btrfs subvolumes, overlayfs, network fs)


def _fs_type(path: Path) -> str:
    """The type of the filesystem holding ``path`` (its longest mount point in /proc/self/mounts)."""
    real, best, kind = os.path.realpath(path), "", ""
    try:
        with open("/proc/self/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    mnt = parts[1].replace("\\040", " ")
                    if (real == mnt or real.startswith(mnt.rstrip("/") + "/")) and len(mnt) > len(best):
                        best, kind = mnt, parts[2]
    except OSError:
        return ""
    return kind


def flock_held(path: Path) -> bool:
    """Some open file holds an flock on ``path`` (as a failing non-blocking attempt would tell, also one of this
    process's): read from /proc/locks on Linux filesystems whose st_dev /proc/locks names (``DEVICE_FS``), the lock is
    not touched; elsewhere a non-blocking probe.  /proc/locks lists the locks of the processes in this pid namespace
    only: the lines of one host (or container) are seen, a holder in another container on the same files is not."""
    path = Path(path)
    if not path.exists():
        return False
    if sys.platform.startswith("linux") and _fs_type(path) in DEVICE_FS:
        st = os.stat(path)
        dev = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
        with open("/proc/locks") as f:
            for line in f:
                parts = line.split()
                if "->" in parts:                                    # a waiter, not a holder
                    continue
                if len(parts) >= 6 and parts[1] == "FLOCK" and parts[5] == dev:
                    return True
        return False
    with open(path, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(f, fcntl.LOCK_UN)
        return False


def honor_stop(root: Path, tag) -> Path:
    """Rename ``root``/STOP to STOP.honored-<tag> (.2, .3, ... after an earlier one of the same tag: every honored stop
    leaves a file of its own, which the supervisor counts).  Only one process honors a line's STOP at a time: its
    trainer while it runs, else its supervisor."""
    dst, n = Path(root) / f"STOP.honored-{tag}", 1
    while dst.exists():
        n += 1
        dst = Path(root) / f"STOP.honored-{tag}.{n}"
    os.replace(Path(root) / "STOP", dst)
    return dst
