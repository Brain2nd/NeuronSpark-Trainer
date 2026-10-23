# Example: a 2B model on the open MiniCPM corpora, one GPU

The configuration and commands of a NeuronSpark pre-training line: the `ns-2b` preset with the MiniCPM5 tokenizer
(130,560 tokens), trained on the open MiniCPM corpora on a single NVIDIA GB300.  `run.json` is the line's
configuration as `nstrain new` and `nstrain tune` wrote it, with the shared-GPU settings (below) edited by hand; only
the data paths are placeholders.

## Data

* **Tokenizer:** `tokenizer.json` of the Hugging Face repository `openbmb/MiniCPM5-1B`; `</s>` is id 1.
* **Sources:** the 15 corpora in `nstrain/data/sources.py` (Ultra-FineWeb, Ultra-FineWeb-L3, UltraX, UltraData-Math,
  UltraData-Code, UltraData-Stack, InfLLM-V2 long documents), each with one held-out file for validation.
* **Recipes** (`nstrain/data/mixture.py`), domain shares web-en 50 / web-zh 25 / math 8 / code 17 as in the tiered-data
  experiment of arXiv 2602.09003 (Sec. 3.4):

  | Recipe | Length | Weights |
  |---|---|---|
  | `stable` (trunk) | 4K | web-en L2 40 + recent crawl 10, web-zh L2 25, math L2 6 + L1 2, code L2 12 + Stack L2 5 |
  | `decay` (branch) | 4K | the L3 corpora (rewritten / synthesised), with 30 % of the stable sources as replay |
  | `long` (branch) | 32K | 70 % long documents, 30 % the decay mix |

* **Cycles** of 20B tokens: each is downloaded, tokenized, trained on and deleted while the next one is prepared
  (80 GB of tokens per cycle).

## Commands

```bash
# the trunk's data line
nstrain data init --work /data/lines/trunk --tokenizer /data/tokenizers/minicpm5/tokenizer.json \
    --eos-id 1 --cycle-tokens 2e10
nstrain data calibrate --work /data/lines/trunk
nstrain data val --work /data/lines/trunk

# the training line: preset, then the fastest micro-batch shape that fits the GPU
nstrain new runs/trunk --model ns-2b --data /data/lines/trunk --peak-lr 3e-4
nstrain tune runs/trunk
nstrain run runs/trunk

# stop with a kept checkpoint after the first 20B-token cycle
nstrain stop runs/trunk --at-cycle-end 0

# the branch planned from that cycle end: 4B tokens of decay at 4K, then 0.8B at 32K
nstrain branch runs/b0 --from runs/trunk --at-cycle 0 --plan decay:4e9,long:0.8e9 --data /data/lines/b0
nstrain tune runs/b0
nstrain run runs/b0
```

## Settings in `run.json`

* **Update:** 1,048,576 tokens (256 rows of 4,096).  `nstrain tune` measured whole updates on the GB300 and chose
  8 rows per micro-batch with the first 3 dendritic neuron layers recomputed (peak 236.5 GiB):

  | Micro-batch | Recomputed layers | Tokens / s |
  |---|---|---|
  | 8 rows | first 3 | 48,591 |
  | 4 rows | none | 47,788 |
  | 16 rows | all | 43,219 |

  In training: about 48.2k tokens/s, 21.8 s per update.
* **Optimizer:** AdamW, β (0.9, 0.95), weight decay 0.1, gradient clip 1.0, peak learning rate 3e-4 (8× for the
  lexical memory tables); 2B tokens of warmup, then constant.  The trunk never decays; its branches do (1-sqrt to 5 %).
* **Checkpoints:** every 0.2B tokens, the last 3 kept (24.5 GB each: fp32 weights and AdamW state); every cycle end is
  kept permanently.
* **Validation:** every 2B tokens, 16 rows of 4,096 tokens from each source's held-out file (`val/<source>` and
  `val/mean` in W&B).  A held-out file is one file of its source, e.g. one programming language for a code source.
* **Shared GPU:** `guards.gpu_check_every: 1` (one `nvidia-smi` query per update) and supervisor waits of 5 s: when
  another process starts on the GPU, the trainer checkpoints at the end of the current update and exits; the
  supervisor resumes the line within seconds of the GPU becoming free.
