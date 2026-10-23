# NeuronSpark-Trainer

The trainer of [NeuronSpark](https://github.com/Brain2nd/NeuronSpark) language models: streaming data lines built from
the open MiniCPM corpora, the training step as CUDA graphs, exact resume from any checkpoint, trunk and branch lines,
one continuous W&B record per line, and a supervisor that keeps a line running unattended.  One package, one command:
`nstrain`.

## Installation

Requirements: a Linux host with an NVIDIA Blackwell GPU, Python ≥ 3.10, and an environment with NeuronSpark's
requirements (PyTorch 2.9.1, Triton 3.5.1) plus `numpy`, `safetensors`, `wandb`, `tokenizers`, `pyarrow` and
`huggingface_hub`.  The package declares no dependencies of its own, so installing it never changes a pinned version.

```bash
pip install --no-deps -e .
```

Host settings go into `~/.nstrain/host.json` (or the file named by `$NSTRAIN_HOST`), read by every command before
anything else:

```json
{"neuronspark": "/path/to/NeuronSpark",
 "env": {"TRITON_PTXAS_PATH": "/usr/local/cuda/bin/ptxas"},
 "work_root": "/data/lines",
 "tokenizer": "/data/tokenizer/tokenizer.json",
 "calibration_from": "/data/lines/trunk/state.json",
 "notify": "/usr/local/bin/notify-training"}
```

`notify` (optional) is run with an event in JSON on stdin when a line finishes, fails, gets stuck, is restarted by the
watchdog, stops because the disk is full, waits for data longer than the watchdog interval, or when its data preparer
keeps failing.

## Concepts

* A **data line** is a directory of pre-tokenized data consumed in fixed-size **cycles**: each cycle is downloaded,
  tokenized, trained on and deleted.  Its mixture schedule is append-only; every input is pinned (dataset revisions,
  file order, tokenizer), every cycle has a permanent manifest, so any cycle can be rebuilt exactly.  The **trunk**
  has no length limit; a **branch** is forked from the state after a trunk cycle and has its own schedule and length.
* A **training line** is a directory: `run.json` (its configuration) and everything the line produces.  Every command
  takes this directory.

```
<run>/run.json          the configuration (every field is documented in nstrain/config.py)
<run>/checkpoints/      u<update>/ and latest
<run>/run.log           the trainer's and the supervisor's output
<run>/supervisor.out    the background supervisor's own output (start-up errors)
<run>/heartbeat         time, update, stage and pid of the trainer
<run>/wandb_run.json    the line's W&B run
<run>/supervisor.json   supervisor state: pids, launches, failures, last event
<run>/STOP              a stop request
<run>/branches.json     the branches forked from this line
```

## Usage

```bash
# a data line (trunk): pinned revisions, calibration, held-out streams
nstrain data init --work /data/lines/trunk --eos-id 1 --cycle-tokens 2e10
nstrain data val --work /data/lines/trunk

# a training line from a model preset (ns-2b, ns-1p3b, tiny-32k)
nstrain new runs/trunk --model ns-2b --data /data/lines/trunk --peak-lr 3e-4
nstrain tune runs/trunk                     # the fastest micro-batch shape of every length that fits the GPU
nstrain check runs/trunk                    # every start check, without a GPU

# run it: the trainer and the line's data preparer under one supervisor, in the background, independent of the
# login (closing the ssh session does not stop it); --foreground keeps the supervisor in the terminal
nstrain run runs/trunk
nstrain status runs/trunk

# stop with a checkpoint: at the next update, or exactly at the end of a data cycle
nstrain stop runs/trunk
nstrain stop runs/trunk --at-cycle-end 1

# a branch from the trunk's end of cycle 1: decay at 4K, then 32K long context
nstrain branch runs/b1 --from runs/trunk --at-cycle 1 --plan decay:10e9,long:2e9 --data /data/lines/b1
nstrain run runs/b1

# continue from an earlier checkpoint as a new W&B run (the old run is marked superseded)
nstrain rollback runs/trunk --to 30000
nstrain run runs/trunk

# throughput of one shape (and its kernel-time breakdown with --profile)
nstrain bench --model ns-2b --seq-len 32768 --rows 1 --accum 32 --ac dendritic
```

A setting a line keeps for its whole life (tokens per update, optimizer, seeds, data and held-out lines, W&B project)
is changed on purpose by editing `run.json` and listing the key in `allow_changes`; otherwise the line refuses to
start.  The schedule only grows at its end.

## Training

* **Updates.** Every optimizer update has the same number of tokens and one sequence length; the loss is the mean over
  the update's tokens.  Each length has its own micro-batch rows and activation recomputation (`lengths` in
  `run.json`), chosen by `nstrain tune` before the first update: the fastest setting whose peak memory fits the GPU
  with a small margin, so the GPU's memory is used, not left idle.  Updates never span a cycle end or a change of sequence length, so every cycle end is an update
  boundary and an exact fork point.
* **Step.** Three CUDA graphs per shape: micro-batch forward and backward (gradients accumulated into flat buffers),
  the global gradient norm, and the clipped AdamW update (one Triton kernel per parameter group).  An update whose
  gradient norm is not finite is skipped without touching weights or optimizer state.
* **Learning rate.** Warmup, constant and decay pieces over tokens trained; a branch continues the trunk's schedule up
  to the fork and decays from there.  The optimizer state continues across sequence-length changes.
* **Checkpoints.** Weights and optimizer state in fp32, the data position (cycle, sample, manifest sha256), the full
  configuration, the code versions and the W&B run; written in the background, hashed (blake2b) and renamed
  atomically.  Each cycle end, each change of sequence length and the start of a line are kept checkpoints.
* **W&B.** One run per line; a resumed line continues its run (the replayed updates are compared with the server's
  copy; updates the server never received before a crash are sent again from the checkpoint, so the curve has no
  hole), a rollback starts a new run backfilled from the old one, a branch starts a new run backfilled from its parent.
* **Determinism.** The same data and state give the same update: a resumed line reproduces the updates of an
  uninterrupted one.

## Recovery

| Event | What happens |
|---|---|
| The ssh login ends, the terminal closes | nothing: the supervisor (and `nstrain data run-preparer`) runs in its own session without a terminal; where systemd-logind would end the login's processes anyway (killing them at logout or when the session is idle, or a user manager without linger), the start is refused with the `systemd-run` command that runs it as a system service |
| The trainer crashes or is killed | the supervisor resumes the line from its newest checkpoint (data position, optimizer state, W&B run); it stops after 5 failures in a row without progress |
| The GPU hangs (no heartbeat for 30 min) | the trainer is killed and restarted; if it does not exit, the supervisor stops and reports it |
| Another process appears on the GPU | the trainer checkpoints and exits (checked every `guards.gpu_check_every` updates); a crash while the other process is there is not counted as a failure; the supervisor waits until the GPU is free and resumes the line |
| A checkpoint write is interrupted | the partial write is discarded at the next start |
| The data source is unreachable, the preparer crashes | the preparer restarts with backoff and resumes the cycle with the same files; the trainer waits; a long wait or repeated preparer failures are notified |
| The disk fills up | the preparer waits for space; the trainer checkpoints while there is room and stops for the operator |
| W&B is unreachable | calls are retried with backoff (2 h by default) |
| A run of non-finite updates | skipped one by one; the line stops after too many in a row |
| The supervisor dies | the trainer stops at the next update boundary with a checkpoint |

## Disk space

A line's disk use is bounded.  After every checkpoint only these stay: the checkpoint `latest` names, the kept
checkpoints (start of the line, cycle ends, stage ends: the points a branch or a later stage continues from), and the
newest `checkpoints.keep_last` rolling checkpoints.  Everything else is removed automatically: older rolling
checkpoints, checkpoints of an abandoned continuation, checkpoints set aside by a rollback (once the line has moved
on), partial writes.  Consumed data cycles are deleted by the preparer, also when a line stops.  The W&B client's local
files of cleanly finished runs are removed at the next start; logs over 256 MB are rotated.

## Code layout

```
nstrain/cli.py         the command line            nstrain/config.py      run.json
nstrain/trainer.py     the training loop           nstrain/supervise.py   the supervisor
nstrain/checks.py      start checks                nstrain/ops.py         new, branch, rollback, stop, check
nstrain/ckpt.py        checkpoints                 nstrain/wb.py          W&B continuity
nstrain/engine.py      the CUDA-graph step         nstrain/flat_adamw.py  AdamW over flat buffers
nstrain/schedule.py    learning-rate schedules     nstrain/model.py       model presets, construction, groups
nstrain/status.py      status                      nstrain/bench.py       throughput
nstrain/gpu.py         the GPU lease and checks    nstrain/host.py        host settings
nstrain/data/          data lines: sources, mixture, hf, tokenize, work, stream, batches, preparer
nstrain/presets/       model configurations
examples/minicpm-2b/   a 2B line on the open MiniCPM corpora: run.json and the commands
```

## Citation

```bibtex
@article{tang2026neuronspark,
  title={NeuronSpark: A Spiking Neural Network Language Model with Selective State Space Dynamics},
  author={Tang, Zhengzheng and Zhu, Hao and Chen, Zhewei and Tao, Dajiang},
  journal={arXiv preprint arXiv:2603.16148},
  year={2026}
}
```

## License

CC BY-NC-SA 4.0 (see `LICENSE`).
