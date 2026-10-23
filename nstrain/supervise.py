"""Supervision of a training line: ``nstrain run RUN`` keeps the trainer and the line's data preparer running.

``nstrain run RUN`` starts the supervisor in the background (``start_background``): its own session without a
controlling terminal, stdin /dev/null, output appended to ``<run_dir>/supervisor.out``, so the end of the ssh login
or a closed terminal reaches neither it nor the trainer; it returns once the supervisor serves the line (its pid and
this start's token in supervisor.json).  Where systemd-logind would end the process with the login anyway
(``logind_reason``: a session scope it kills at logout or when idle, a user manager without linger) the start is
refused with the command that runs the supervisor as a system service.  ``nstrain run RUN --foreground`` keeps it in
the terminal (tests, debugging): a hangup of that terminal is a stop request like SIGTERM.  ``nstrain data
run-preparer`` starts the supervised preparer alone the same way (``start_preparer_background``).

Trainer (``python -m nstrain train RUN``, own session, output appended to ``<run_dir>/run.log``):

* exact resume after any abnormal exit: the trainer continues from the line's newest checkpoint;
* a failure counts toward ``supervisor.max_fails`` only when the line made no progress since the previous launch
  (``latest`` did not move); at max_fails the supervisor stops with the trainer's exit code (a deterministic fault
  needs a human); exit 0 (done, stopped, data exhausted) ends the line;
* exit 3 (another process on the GPU, or the GPU lease held): wait until the GPU is free (nvidia-smi on the training
  device; when nvidia-smi itself fails the wait ends and the trainer's own check decides, a counted failure), pause,
  relaunch; any other abnormal exit while another process is on the GPU (a kernel fault under time-slicing, see
  nstrain.gpu) is handled the same way; exit 4 (another trainer still holds the line, e.g. one writing its last
  checkpoint): wait for it; none of these is a failure;
* exit 5 (not enough disk space for the next checkpoints): stop, the line needs the operator;
* watchdog: a heartbeat older than ``supervisor.stall_s`` (and older than this launch) means a stalled trainer (e.g. a
  GPU hang): SIGKILL by PID, restart like a crash; a trainer that does not exit within 120 s of the kill is reported
  ("stuck in the driver") and the supervisor stops;
* SIGTERM / SIGINT / SIGHUP: passed to the trainer as SIGTERM (it stops at the next update boundary with a
  checkpoint); between trainers (restart pause, waits, before a launch) the supervisor ends at once.  The trainer gets
  SIGTERM if the supervisor dies (PR_SET_PDEATHSIG, nstrain.proc).

Preparer (``supervisor.prepare``): the supervisor serves the data line (``<data_work>/preparer.lock``, see
nstrain.data.preparer) and keeps ``nice -n 10 python -m nstrain data loop`` running: restarted after any exit with a
pause of 60 s doubling to 1800 s (back to 60 s after an hour of uptime; a Hugging Face outage can last hours), output
appended to ``<data_work>/preparer.log``; ended (SIGTERM, SIGKILL after 60 s) with the supervisor, which then releases
the cycles the trainer consumed.  ``nstrain data run-preparer`` runs only this part (``run_preparer``).

Logs over 256 MB are rotated (one previous file kept) before a launch.  Events that need attention (the line
finished, failed, stuck, stopped for disk space; a watchdog kill; the trainer waiting for data longer than stall_s;
the preparer exiting three times in a row, or another process serving the data line) are written to supervisor.json
and sent to the host's ``notify`` command (nstrain.host).  ``supervisor.lock`` allows one supervisor per line.
"""
from __future__ import annotations

import fcntl
import json
import os
import pwd
import re
import secrets
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

from nstrain import config as C
from nstrain import exits, host
from nstrain.data.preparer import HELD_BY_PARENT, release_consumed, serve
from nstrain.gpu import GpuCheckFailed, lease_holder, other_processes
from nstrain.proc import SUPERVISOR

KILL_WAIT_S = 120
START_WAIT_S = 60.0
LOG_MAX = 256 << 20


class _Stop:
    """Set by SIGTERM / SIGINT / SIGHUP (the handler only sets the flag and signals the trainer); waits are slices of
    at most 1 s, so a stop is seen within a second wherever it lands."""

    def __init__(self):
        self.event = threading.Event()
        self.child: Optional[subprocess.Popen] = None

    def install(self) -> None:
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(s, self._on)

    def _on(self, *_):
        self.event.set()
        c = self.child
        if c is not None and c.returncode is None:
            try:
                os.kill(c.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def __bool__(self) -> bool:
        return self.event.is_set()

    def sleep(self, s: float) -> bool:
        """Sleep up to ``s`` seconds; True if a stop came."""
        end = time.time() + s
        while not self.event.is_set():
            left = end - time.time()
            if left <= 0:
                return False
            self.event.wait(min(1.0, left))
        return True


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _rotate(p: Path) -> None:
    if p.exists() and p.stat().st_size > LOG_MAX:
        os.replace(p, p.with_name(p.name + ".1"))


def _code(rc: int) -> int:
    """The exit status of a child as a shell reports it (a signal N: 128 + N)."""
    return 128 - rc if rc < 0 else rc


def notify(event: dict) -> None:
    """The host's notify command with the event on stdin (failures are reported, never raised)."""
    cmd = host.get("notify")
    if not cmd:
        return
    try:
        subprocess.run(cmd, shell=isinstance(cmd, str), input=json.dumps(event), text=True, timeout=60,
                       capture_output=True, check=True)
    except Exception as e:
        print(f"{_now()} notify failed: {e}", flush=True)


class PreparerSupervisor(threading.Thread):
    BACKOFF = (60.0, 1800.0, 3600.0)                                 # first pause, longest pause, uptime that resets

    def __init__(self, work: str, prep: C.Preparer, stop: _Stop, say, trouble=None):
        super().__init__(daemon=True, name="preparer")
        self.work, self.prep, self.stop, self.say = work, prep, stop, say
        self.trouble = trouble or (lambda kind, msg: None)              # an event worth a notification
        self.proc: Optional[subprocess.Popen] = None
        self.ended = threading.Event()
        self.lock = None

    def cmd(self) -> List[str]:
        p = self.prep
        return ["nice", "-n", "10", sys.executable, "-u", "-m", "nstrain", "data", "loop", "--work", self.work,
                "--lookahead", str(p.lookahead), "--min-free-gb", str(p.min_free_gb), "--dl-workers",
                str(p.dl_workers), "--tok-workers", str(p.tok_workers), "--rayon", str(p.rayon)]

    def run(self) -> None:
        try:
            self.lock = serve(self.work)
        except RuntimeError as e:
            self.say(f"{e}; this line runs without its own preparer")
            self.trouble("preparer", f"{e}; this line runs without its own preparer")
            return
        first, longest, reset = self.BACKOFF
        pause = first
        failed = 0
        log_path = Path(self.work) / "preparer.log"
        try:
            while not self.stop and not self.ended.is_set():
                _rotate(log_path)
                t0 = time.time()
                with open(log_path, "a") as log:
                    self.proc = subprocess.Popen(self.cmd(), stdout=log, stderr=subprocess.STDOUT,
                                                 start_new_session=True,
                                                 env=dict(os.environ, **{SUPERVISOR: str(os.getpid()),
                                                                         HELD_BY_PARENT: "1"}))
                self.say(f"preparer started (pid {self.proc.pid})")
                while self.proc.poll() is None and not self.stop and not self.ended.is_set():
                    time.sleep(1.0)
                if self.proc.poll() is None or self.stop or self.ended.is_set():
                    break
                rc = _code(self.proc.returncode)
                if time.time() - t0 > reset:
                    pause, failed = first, 0
                failed += 1
                self.say(f"preparer exited {rc}; restarting in {pause:.0f} s")
                if failed == 3:
                    self.trouble("preparer", f"the preparer of {self.work} exited {failed} times in a row (last exit "
                                             f"{rc}; see {log_path})")
                end = time.time() + pause
                while time.time() < end and not self.stop and not self.ended.is_set():
                    time.sleep(1.0)
                pause = min(2 * pause, longest)
        finally:
            self.terminate()

    def terminate(self) -> None:
        p = self.proc
        if p is not None and p.poll() is None:
            p.send_signal(signal.SIGTERM)
            try:
                p.wait(60)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()

    def end(self) -> None:
        self.ended.set()
        self.terminate()
        self.join(timeout=90)
        if self.lock is not None:
            self.lock.close()
            self.lock = None


def gpu_state() -> str:
    """'busy' (another compute process, or the GPU lease held), 'free', or 'failed' (nvidia-smi failed)."""
    if lease_holder() is not None:
        return "busy"
    try:
        return "busy" if other_processes() else "free"
    except GpuCheckFailed:
        return "failed"


def line_held(root: Path) -> bool:
    """Another process holds the line's trainer.lock."""
    p = root / "trainer.lock"
    if not p.exists():
        return False
    with open(p, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(f, fcntl.LOCK_UN)
        return False


class LineSupervisor:
    def __init__(self, run):
        self.cfg = C.load(run)
        self.root = self.cfg.run_dir
        self.stop = _Stop()
        self.log = None
        self.log_lock = threading.Lock()                             # the preparer thread reports here too
        self.state_lock = threading.Lock()                           # and records events
        self.state = {}

    # ------------------------------------------------------------------ records
    def say(self, msg: str) -> None:
        line = f"{_now()} supervisor: {msg}"
        print(line, flush=True)
        with self.log_lock:
            if self.log is not None:
                self.log.write(line + "\n")
                self.log.flush()

    def record(self, **kw) -> None:
        with self.state_lock:
            self.state.update(kw)
            tmp = self.root / f"supervisor.json.tmp{os.getpid()}"
            tmp.write_text(json.dumps(self.state, indent=1))
            os.replace(tmp, self.root / "supervisor.json")

    def event(self, kind: str, msg: str) -> None:
        e = {"time": _now(), "line": self.cfg.name, "run_dir": str(self.root), "event": kind, "message": msg}
        self.record(last_event=e)
        notify(e)

    def trainer_cmd(self) -> List[str]:
        return [sys.executable, "-u", "-m", "nstrain", "train", str(self.root)]

    def latest(self) -> str:
        p = self.cfg.ckpt_dir / "latest"
        return p.read_text().strip() if p.exists() else "none"

    def heartbeat_time(self) -> float:
        try:
            return (self.root / "heartbeat").stat().st_mtime
        except FileNotFoundError:
            return 0.0

    def open_log(self) -> None:
        with self.log_lock:
            if self.log is not None:
                self.log.close()
            _rotate(self.root / "run.log")
            self.log = open(self.root / "run.log", "a", buffering=1)

    # ------------------------------------------------------------------ the line
    def run(self) -> int:
        lock = open(self.root / "supervisor.lock", "a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"{self.cfg.name}: another supervisor runs this line") from None
        self.stop.install()
        self.open_log()
        self.record(pid=os.getpid(), start_token=os.environ.get(START_TOKEN), host=socket.gethostname(),
                    started=_now(), trainer_pid=None, launches=0, fails=0, state="starting", last_event=None)
        prep = None
        if self.cfg.supervisor.prepare:
            prep = PreparerSupervisor(self.cfg.data_work, self.cfg.supervisor.preparer, self.stop, self.say,
                                      self.event)
            prep.start()
        try:
            rc = self._trainer_loop()
        finally:
            if prep is not None:
                prep.end()
                try:
                    gone = release_consumed(self.cfg.data_work)
                    if gone:
                        self.say(f"released the consumed cycles {gone}")
                except Exception as e:                               # the next preparer pass releases them
                    self.say(f"releasing the consumed cycles failed: {e}")
            final = self.state.get("state")
            self.record(state=final if final in ("finished", "failed", "stuck", "needs operator") else "ended",
                        trainer_pid=None, ended=_now())
            with self.log_lock:
                self.log.close()
                self.log = None
        return rc

    def _wait(self, why: str, busy, every: float) -> bool:
        """Wait while ``busy()``; True if a stop came."""
        self.say(why)
        while busy():
            if self.stop.sleep(every):
                return True
        return False

    def _trainer_loop(self) -> int:
        s = self.cfg.supervisor
        fails = 0
        while True:
            if self.stop:
                self.say("stopped on request")
                return 0
            self.open_log()
            start, before = time.time(), self.latest()
            proc = subprocess.Popen(self.trainer_cmd(), stdout=self.log, stderr=subprocess.STDOUT,
                                    start_new_session=True, env=dict(os.environ, **{SUPERVISOR: str(os.getpid())}))
            self.stop.child = proc
            if self.stop:                                            # a stop that came before child was set
                proc.send_signal(signal.SIGTERM)
            self.record(trainer_pid=proc.pid, trainer_started=_now(), launches=self.state["launches"] + 1,
                        state="training")
            self.say(f"trainer started (pid {proc.pid}, latest {before})")
            if self._watch(proc, start) == "stuck":
                msg = (f"trainer {proc.pid} does not exit {KILL_WAIT_S} s after SIGKILL (stuck in the driver?); "
                       "stopping")
                self.say(msg)
                self.record(state="stuck")
                self.event("stuck", msg)
                return 1
            rc = _code(proc.wait())
            self.stop.child = None
            self.record(trainer_pid=None, last_exit=rc)
            if self.stop:
                self.say(f"stopped on request (trainer exit {rc})")
                return rc
            if rc == exits.OK:
                self.say("trainer finished (exit 0)")
                self.record(state="finished")
                self.event("finished", f"latest {self.latest()}")
                return 0
            if rc == exits.NEEDS_OPERATOR:
                msg = f"trainer stopped for the operator (exit {rc}, see run.log; latest {self.latest()})"
                self.say(msg)
                self.record(state="needs operator")
                self.event("needs operator", msg)
                return rc
            if rc not in (exits.GPU_BUSY, exits.LINE_BUSY) and gpu_state() == "busy":
                self.say(f"trainer exited {rc} after {time.time() - start:.0f} s with another process on the GPU; "
                         f"not a failure (latest {self.latest()})")
                rc = exits.GPU_BUSY
            if rc == exits.GPU_BUSY:
                self.record(state="waiting for the GPU")
                failed = []

                def busy():
                    g = gpu_state()
                    if g == "failed":
                        failed.append(1)
                    return g == "busy"
                if self._wait("another process is using the GPU; waiting for it to finish", busy, s.gpu_wait_s):
                    self.say("stopped on request while waiting for the GPU")
                    return 0
                if failed:
                    self.say("nvidia-smi failed; relaunching (the trainer's own check decides)")
                self.stop.sleep(s.restart_pause_s)
                continue
            if rc == exits.LINE_BUSY:
                self.record(state="waiting for the line")
                if self._wait("another trainer holds the line; waiting for it to exit",
                              lambda: line_held(self.root), 5.0):
                    self.say("stopped on request while waiting for the line")
                    return 0
                continue
            fails = 1 if self.latest() != before else fails + 1
            self.record(fails=fails, state="restarting")
            self.say(f"trainer exited {rc} after {time.time() - start:.0f} s (failure {fails} of {s.max_fails} "
                     f"without progress; latest {self.latest()})")
            if fails >= s.max_fails:
                msg = f"{fails} failures without progress, stopping"
                self.say(msg)
                self.record(state="failed")
                self.event("failed", f"{msg} (last exit {rc}; latest {self.latest()})")
                return rc
            self.say(f"restarting in {s.restart_pause_s:.0f} s")
            self.stop.sleep(s.restart_pause_s)

    def _stage(self) -> str:
        try:
            return json.loads((self.root / "heartbeat").read_text()).get("stage", "")
        except (FileNotFoundError, json.JSONDecodeError):
            return ""

    def _watch(self, proc: subprocess.Popen, start: float) -> Optional[str]:
        """Until the trainer exits; SIGKILL when its heartbeat is older than stall_s; an event when it has been
        waiting for data longer than stall_s (once per wait).  'stalled' / 'stuck' / None."""
        stall = self.cfg.supervisor.stall_s
        waiting, told = None, False
        while True:
            try:
                proc.wait(1.0)                                       # also after a stop: no busy wait
                return None
            except subprocess.TimeoutExpired:
                pass
            if self._stage() == "waiting for data":
                waiting = waiting or time.time()
                if not told and time.time() - waiting > stall:
                    told = True
                    self.event("waiting for data", f"the trainer has waited {time.time() - waiting:.0f} s for "
                                                   f"prepared data (see {self.cfg.data_work}/preparer.log)")
            else:
                waiting, told = None, False
            hb = max(self.heartbeat_time(), start)                   # a heartbeat of an earlier process
            if time.time() - hb > stall and proc.poll() is None:
                try:
                    beat = (self.root / "heartbeat").read_text()
                except FileNotFoundError:
                    beat = "none"
                msg = f"no heartbeat for more than {stall:.0f} s ({beat}); killing trainer {proc.pid}"
                self.say(msg)
                self.event("watchdog", msg)
                proc.kill()
                try:
                    proc.wait(KILL_WAIT_S)
                except subprocess.TimeoutExpired:
                    return "stuck"
                return "stalled"


def run_line(run) -> int:
    return LineSupervisor(run).run()


IDLE_NEVER = "t 18446744073709551615"                 # StopIdleSessionUSec: infinity
START_TOKEN = "NSTRAIN_START_TOKEN"
SERVICE_ENV = ("NSTRAIN_HOST", "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "HF_ENDPOINT", "NSTRAIN_GPU_LOCK_DIR")


def _user_name(uid: int) -> str:
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def logind_reason(cgroup: str, prop: Callable[[str, str, str], str]) -> Optional[str]:
    """Why systemd-logind would end a process of ``cgroup`` (the text of /proc/PID/cgroup) together with a login, or
    None.  ``prop(object, interface, name)``: a login1 property as busctl prints it.  As logind decides:

    * a process of a user manager (``user@UID.service``, e.g. under ``systemd-run --user``) ends at the user's last
      logout unless the user lingers;
    * a login session's scope (``user-UID.slice/session-N.scope``) is stopped, its processes killed, when an idle
      session is stopped (``StopIdleSessionSec`` finite), and at the session's end when logind's kill rule holds for
      the session's owner: not in ``KillExcludeUsers`` (an unset list, printed as empty, excludes root), then in
      ``KillOnlyUsers`` when that list is set, else ``KillUserProcesses``."""
    mgr = ("/org/freedesktop/login1", "org.freedesktop.login1.Manager")
    m = re.search(r"/user@(\d+)\.service/", cgroup)
    if m:
        uid = m.group(1)
        if prop(f"/org/freedesktop/login1/user/_{uid}", "org.freedesktop.login1.User", "Linger") != "b true":
            return f"this process runs under user@{uid}.service, which stops at the user's last logout (no linger)"
        return None
    m = re.search(r"/user-(\d+)\.slice/session-[^/]+\.scope", cgroup)
    if not m:
        return None
    owner = _user_name(int(m.group(1)))
    idle = prop(*mgr, "StopIdleSessionUSec")
    if idle != IDLE_NEVER:
        return f"systemd-logind stops idle login sessions and their processes (StopIdleSessionUSec {idle.split()[-1]})"
    exclude = re.findall(r'"([^"]*)"', prop(*mgr, "KillExcludeUsers"))
    if owner in exclude or (not exclude and owner == "root"):
        return None
    only = re.findall(r'"([^"]*)"', prop(*mgr, "KillOnlyUsers"))
    kill = owner in only if only else prop(*mgr, "KillUserProcesses") == "b true"
    return f"systemd-logind ends the processes of {owner}'s login session at logout" if kill else None


def logout_reason() -> Optional[str]:
    """``logind_reason`` for this process; login1's properties are read through busctl (an unreadable one raises)."""
    try:
        cgroup = Path("/proc/self/cgroup").read_text()
    except OSError:
        return None

    def prop(obj: str, iface: str, name: str) -> str:
        p = subprocess.run(["busctl", "get-property", "org.freedesktop.login1", obj, iface, name], capture_output=True,
                           text=True, timeout=30)
        if p.returncode != 0:
            why = p.stderr.strip() or f"busctl exit {p.returncode}"
            raise RuntimeError(f"cannot read systemd-logind's {name}: {why}")
        return p.stdout.strip()

    return logind_reason(cgroup, prop)


def service_command(unit: str, cmd: List[str]) -> str:
    """The command that runs ``cmd`` (a foreground command) as a transient system service: this user, this working
    directory and the environment nstrain reads (credentials are not copied: the service user's own W&B / Hugging Face
    logins apply)."""
    unit = "nstrain-" + re.sub(r"[^A-Za-z0-9:_.-]", "_", unit)
    parts = ["systemd-run", f"--unit={unit}", "--collect", f"--working-directory={os.getcwd()}"]
    if os.getuid() != 0:
        parts = ["sudo"] + parts + [f"--uid={os.getuid()}", f"--gid={os.getgid()}"]
    parts += [f"--setenv={k}={os.environ[k]}" for k in SERVICE_ENV if k in os.environ]
    return shlex.join(parts + cmd)


def _start_background(cmd: List[str], foreground: List[str], out: Path, record: Path, what: str, unit: str) -> int:
    """Start ``cmd`` in its own session (no controlling terminal), stdin /dev/null, output appended to ``out``; return
    its pid once ``record`` carries its pid and this start's token.  Refused (with ``service_command`` of
    ``foreground``) where logind would end it with the login; a start that exits first raises with its output."""
    reason = logout_reason()
    if reason:
        raise RuntimeError(f"{reason}: a {what} started here would end with the login; start it as a system service: "
                           f"{service_command(unit, foreground)}")
    token = secrets.token_hex(8)
    with open(out, "ab") as log:
        start = log.tell()
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True, env=dict(os.environ, **{START_TOKEN: token}))
    deadline = time.time() + START_WAIT_S
    while time.time() < deadline:
        rc = p.poll()
        if rc is not None:
            with open(out, "rb") as f:
                f.seek(start)
                tail = f.read().decode(errors="replace").strip().splitlines()[-5:]
            raise RuntimeError(f"the {what} exited at its start (exit {rc}): " + " | ".join(tail))
        try:
            r = json.loads(record.read_text())
            if r.get("pid") == p.pid and r.get("start_token") == token:
                return p.pid
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(0.2)
    raise RuntimeError(f"the {what} (pid {p.pid}) did not report within {START_WAIT_S:.0f} s; its output: {out}")


def start_background(run, cmd: Optional[List[str]] = None) -> int:
    """``nstrain run RUN``: the line's supervisor in the background (see the module docstring); its pid once it serves
    the line.  ``cmd``: the supervisor's command (default ``nstrain run RUN --foreground``)."""
    cfg = C.load(run)
    fg = [sys.executable, "-m", "nstrain", "run", str(cfg.run_dir), "--foreground"]
    return _start_background(cmd or fg, fg, cfg.run_dir / "supervisor.out", cfg.run_dir / "supervisor.json",
                             f"supervisor of {cfg.name}", cfg.name)


def start_preparer_background(work: str, args: List[str], cmd: Optional[List[str]] = None) -> int:
    """``nstrain data run-preparer --work W``: the supervised preparer in the background, like ``start_background``;
    its pid once it serves the data line.  ``args``: its preparer options."""
    w = Path(work).resolve()
    fg = [sys.executable, "-m", "nstrain", "data", "run-preparer", "--work", str(w), *args, "--foreground"]
    return _start_background(cmd or fg, fg, w / "preparer-supervisor.out", w / "preparer-supervisor.json",
                             f"preparer of {w.name}", f"preparer-{w.name}")


def run_preparer(work: str, prep: C.Preparer) -> int:
    """The supervised preparer alone (until SIGTERM / SIGINT / SIGHUP); ``<work>/preparer-supervisor.json`` records it
    once it serves the data line.  Exit 1 when the data line cannot be served (another process serves it)."""
    stop = _Stop()
    stop.install()

    def say(msg):
        print(f"{_now()} preparer supervisor: {msg}", flush=True)

    t = PreparerSupervisor(work, prep, stop, say)
    t.start()
    while t.is_alive() and t.lock is None:
        t.join(0.2)
    if t.lock is None:
        return 1
    rec = Path(work) / "preparer-supervisor.json"
    tmp = rec.with_name(f"{rec.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps({"pid": os.getpid(), "start_token": os.environ.get(START_TOKEN),
                               "host": socket.gethostname(), "started": _now()}, indent=1))
    os.replace(tmp, rec)
    while t.is_alive():
        t.join(1.0)
    t.end()
    return 0
