# NeuronSpark-Trainer

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

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
watchdog or after a signal it did not send, stops because the disk is full, waits for data or for the GPU longer than
the watchdog interval, when its data preparer keeps failing, or when the supervisor itself fails.

## Concepts

* A **data line** is a directory of pre-tokenized data consumed in fixed-size **cycles**: each cycle is downloaded,
  tokenized, trained on and deleted.  Its mixture schedule is append-only; every input is pinned (dataset revisions,
  file and document order, tokenizer), every cycle has a permanent manifest, so any cycle can be rebuilt exactly.
  The **trunk** grows until a source's data are used up (an error, not a second pass: `max_epochs` 1, every token
  is trained once); a **branch** is forked from the state after a trunk cycle and has its own schedule and length.
* The **mix is representative at every level**: every update holds each source within two samples of its recipe
  share; every cycle and every segment of it hold every stratum of a source (its subdirectories: languages, crawl
  dumps, L3 sub-types; or its files, for a corpus whose files are kinds of data) at its share of the source's tokens,
  taken from the stratum's open files (all of each but the few left open for the next cycles, which give the rest in
  proportion to their sizes) and cut at document ends; a file's documents are
  shuffled at tokenization and encoded as plain text (a token of the tokenizer's added vocabulary written in a
  document, e.g. `</s>`, `<param` or `<think>` in code, is its characters, not that token; the exception is a source's
  own control markers, such as the `<|im_start|>` / `<|im_end|>` of the chats and the `<|fim_...|>` of the
  fill-in-the-middle code in the InfLLM-V2 data, which encode as those tokens.  Matching added tokens in a text,
  the default of the `tokenizers` library, encoded about 127 of every million tokens of UltraData-Code L2 and 126 of
  Stack v3 as control tokens -- `/// <param name=...>` in C#, `#include <functional>` in C++, `"<s>"` in Python strings,
  `/think` in paths -- and 770 per million in the InfLLM-V2 data, mostly its chat markers; measured over every
  held-out document of the 15 sources); the trainer reads each
  source's windows of a segment in a random order.  1 % of every file's documents (5 % of the InfLLM-V2 data's; by a hash of their text, the same
  in every line) are held out and never trained on as documents of that source (a text that is also in another
  source, e.g. quoted inside an L3 row, is held out or trained there by that source's own hash); each
  source's validation rows come from those of every stratum, and its loss weights its strata by their shares;
  `val_<L>/mix` weights the sources by the recipe.  Every validation also probes the dendritic neuron layers on
  its rows (`val_<L>/dendritic/...`, over every dendritic head and per layer): `retention_p50`, the median head's
  retention over the row (`exp(A Σ dt)`, what is left of a compartment's potential at the row's first token after its
  last); `retention_frac_gt_0.9`, the fraction of slow-leak heads; `retention_max` and `retention_frac_gt_0.5`, the slowest
  head's retention and the fraction of the heads above 0.5 (the tail below the slow-leak heads);
  `retention32k_frac_gt_0.5`, the fraction of the heads that would keep half of it over 32K tokens at the row's mean
  step; `tau_p50`, `tau_p90`, `tau_max`, the heads' e-fold lengths in tokens at the row's mean step (the row length
  over the row's decay exponent `-A Σ dt`; readable where the retention underflows); `firing`, the mean firing rate;
  and at the rows' document boundaries (an EOS followed by the next document's first token): `boundaries`, their
  count, and when it is not 0, `boundary_retention_p50` and `boundary_retention_max`, the median and the slowest head's
  retention across a boundary (`exp(A (dt_e + dt_e+1))`, what is left of the old document's potential after the new
  one's first token), `boundary_step_ratio_p50` and `boundary_step_ratio_max`, the heads' mean step over a boundary's
  two tokens over their mean step elsewhere (well above 1: a head that flushes its compartments at the boundaries).
  The hippocampal layers are probed on the same rows (`val_<L>/hippocampal/...`, per layer and over the model):
  `decay_tau_p50/p90/max`, the decay channels' e-fold lengths in tokens; `clock_tau_head<h>` (per layer) and
  `clock_tau_p50/max`, the tokens over which a head's pool clock runs one e-fold; `clock_boundary_step_ratio`, the
  clock's increment over a document boundary over its increment elsewhere; `write_head<h>` (per layer) and
  `write_p50/max`, the mean write tag of a head; the firing rates `gate_firing`, `ffn_in_firing`, `ffn_gate_firing`,
  `ffn_up_firing`; the copy arbitration's `copy_pi`, `novelty`, `read_gain`; the PKM read's `pkm_entropy`,
  `pkm_slots_used`, `pkm_gate`; `rope_ext_q_head<h>` / `rope_ext_k_head<h>` (per layer) and `rope_ext_q_p50/max`,
  `rope_ext_k_p50/max`, the share of a head's query / key energy in the RoPE pairs whose wavelength exceeds the
  training row length (against `rope_ext_pairs_frac`, the share of all pairs that are such).
  When the model's longest context (`rope_max_seq_len`) exceeds the training length, every validation also runs
  the long-row probe on each source's held-out rows of that length, one row at a time: `val_<P>/<source>`, `/mix`,
  `/mean` (the loss at the long length), `val_<P>/pos/seg<k>` (the recipe-weighted loss by position, in segments
  of the training length) and `/pos/last_over_first`, and the dendritic probe over the long rows (`val_<P>/dendritic/...`,
  with the firing rate by segment).  One row of each probe runs before every capture (its memory and kernels), and
  its readings are printed.  The row of an update that validated is printed to `run.log` whatever `print_every`.
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
* **Checkpoints.** Weights and optimizer state in fp32, the data position (cycle, sample, manifest sha256, the
  sample order's digest), the full
  configuration, the code versions and the W&B run; written in the background, hashed (blake2b) and renamed
  atomically.  Each cycle end, each change of sequence length and the start of a line are kept checkpoints.
* **W&B.** One run per line; a resumed line continues its run (the replayed updates are compared with the server's
  copy; updates the server never received before a crash are sent again from the checkpoint, so the curve has no
  hole), a rollback starts a new run backfilled from the old one, a branch starts a new run backfilled from its parent.
* **Determinism.** The same data, state and shape give the same update: a line resumed in the same GPU mode
  reproduces the updates of an uninterrupted one (the checkpoint records the sample order's digest; a resume with
  another order is refused).  Beside other processes (a reserve set, and they are there) the line trains with the lengths' `shared`
  shapes (another micro-batch split of the same update: the same data, equal up to rounding) and goes back to its
  own shapes once they have left.
* **GPU memory.** The trainer's arena (one memory pool) holds all of the GPU's memory in large blocks -- but a 4 GiB
  slack for what a capture creates outside the allocator, and, beside other processes, what they may still take --
  and every allocation of an engine's warm-up and capture and of the validation rows of any length is served from
  them (a pool that takes blocks from the driver as they are asked for holds none larger than the largest allocation
  so far: the long rows' allocations are larger than any of a training step's).
* **One line per GPU.** The GPU lease is the trainer's: between a trainer's crash and its relaunch (seconds) another
  line's supervisor waiting for the GPU may take it.  Lines that share a GPU wait for each other; none has priority.

## Recovery

| Event | What happens |
|---|---|
| The ssh login ends, the terminal closes | nothing: the supervisor (and `nstrain data run-preparer`) runs in its own session without a terminal; where systemd-logind would end the login's processes anyway (killing them at logout or when the session is idle, or a user manager without linger), the start is refused with the `systemd-run` command that runs it as a system service |
| The trainer crashes or is killed | the supervisor resumes the line from its newest checkpoint (data position, optimizer state, W&B run); it stops after 5 failures in a row without progress.  A trainer stopped by a SIGTERM that did not come from the supervisor checkpoints and is relaunched (`nstrain stop` stops the line) |
| The GPU hangs (no heartbeat for 30 min) | the trainer is killed and restarted; if it does not exit, the supervisor stops and reports it |
| Another process appears on the GPU | with `guards.share_reserve_gb` set the line goes on beside it, inside the same process (no restart: weights, optimizer state, the data position and the W&B run stay as they are): at the check that sees it (`guards.gpu_check_every` updates) the trainer releases its engine and arena, holds at once all of the GPU's memory but what the other processes may still take -- the reserve less what they use, adjusted at every check from then on: they cannot grow past the reserve (an allocation past it fails in them) -- and captures each length's `shared` shape (measured for that reserve by `nstrain tune --shared`; a start without one is refused) in the new arena (`nstrain.gpu.Arena`); what it holds stays held, but for a 4 GiB slack it gives back while it captures an engine; what they free is held again only past their reserve (the free memory is read between two readings of theirs).  The memory is free for anything to take only in the gap between the close of the old arena and the hold of the new one (one round of cudaFree / cudaMalloc).  Once they have been gone for 3 checks the line takes the GPU back the same way and trains with its own shapes again, so it runs at full speed whenever it is alone.  When they use more than the reserve and the shapes' margin -- at a start, at a check, or taken in that gap so that the capture beside them runs out of memory -- the trainer checkpoints and exits, and the supervisor relaunches it once they fit or the GPU is free.  Without a reserve the trainer checkpoints and exits, and the supervisor waits until the GPU is free.  A crash while another process is (or was at the trainer's last check) on the GPU is not counted as a failure; after 3 crashes beside them without progress the line waits for a free GPU |
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
