# Example: a 2B model on the open MiniCPM corpora, one GPU

The configuration and commands of a NeuronSpark pre-training line: the `ns-2b` preset with the MiniCPM5 tokenizer
(130,560 tokens), trained on the open MiniCPM corpora on a single NVIDIA GB300.  `run.json` is the line's
configuration as `nstrain new`, `nstrain tune` and `nstrain tune --shared` wrote it; only the data paths are
placeholders.

## Data

* **Tokenizer:** `tokenizer.json` of the Hugging Face repository `openbmb/MiniCPM5-1B`; `</s>` is id 1.
* **Sources:** the 15 corpora in `nstrain/data/sources.py` (Ultra-FineWeb, Ultra-FineWeb-L3, UltraX, UltraData-Math,
  UltraData-Code, UltraData-Stack, InfLLM-V2 long-context data); 1 % of every file's documents (5 % of the InfLLM-V2
  data's) are held out for validation, never trained on.
* **Recipes** (`nstrain/data/mixture.py`), domain shares web-en 50 / web-zh 25 / math 8 / code 17 as in the tiered-data
  experiment of arXiv 2602.09003 (Sec. 3.4):

  | Recipe | Length | Weights |
  |---|---|---|
  | `stable` (trunk) | 4K | web-en L2 46.7 + 2025-H2 crawls 3.3 (their shares of bytes), web-zh L2 25, math L2 6 + L1 2, code L2 12 + Stack L2 5 |
  | `decay` (branch) | 4K | the L3 corpora (rewritten / synthesised), with 30 % of the stable mix as replay |
  | `long` (branch) | 32K | 70 % InfLLM-V2 long-context data (MiniCPM's long stage mix; ~16 % of its tokens are in rows of 32K or more), 30 % the decay mix |

* **Cycles** of 20B tokens: each is downloaded, tokenized, trained on and deleted while the next one is prepared
  (80 GB of tokens per cycle).  A cycle holds every stratum of each source at its share of the source's bytes
  (languages, crawl dumps -- more than 16 of them grouped into 16 neighbouring groups --, L3 sub-types; each file of
  the InfLLM-V2 data, one kind of data each), taken from the stratum's open files: about 16 per source spread over
  the strata by size, at least one per stratum, fewer while they would hold more than 4 cycles' worth of the source
  -- but never fewer than one per stratum, so a source of many large strata (the code corpora) keeps more open.  A
  cycle reads all of every further file it opens and the rest from those open files in proportion to their sizes, so
  after a cycle no more than those are open (105 for the stable recipe; their whole streams, at most 130 GB, wait in
  `pool/` for the next cycles) -- a few more after a cycle whose bytes-per-token estimate of a stratum was low: the
  files it opened but did not need wait untouched, and the next cycle reads them first.

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
  8 rows per micro-batch with the first dendritic neuron layer recomputed (peak 236.8 GiB):

  | Micro-batch | Recomputed layers | Tokens / s | Peak reserved |
  |---|---|---|---|
  | 8 rows | first 1 | 49,576 | 236.8 GiB |
  | 4 rows | none | 47,860 | 153.9 GiB |
  | 16 rows | all 18 | 43,245 | 222.6 GiB |

  Beside other processes (`nstrain tune --shared`, 50 GB left to them, shapes below 192.1 GiB): 4 rows with no
  recomputation, 47,843 tokens/s at 153.9 GiB (8 rows would recompute 9 layers: 46,094 tokens/s at 191.7 GiB).
  In training with the shared shapes: about 47.6k tokens/s, 22.0 s per update.
* **Optimizer:** AdamW, β (0.9, 0.95), weight decay 0.1, gradient clip 1.0, peak learning rate 3e-4 (8× for the
  lexical memory tables); 2B tokens of warmup, then constant.  The trunk never decays; its branches do (1-sqrt to 5 %).
* **Checkpoints:** every 0.2B tokens, the last 3 kept (24.5 GB each: fp32 weights and AdamW state); every cycle end is
  kept permanently.
* **Validation:** every 2B tokens.  `nstrain data val` keeps the held-out documents (by a hash of their text) of each
  stratum's first files: 8 files per source spread over its strata, at least one per stratum, more until a stratum
  holds its rows at 32K (about 130 GB in 190 files for the 15 sources, once; cycle 0 downloads these files again,
  the first of each stratum, for their training documents).  Each stratum validates on 16 x its
  share rows, at least one micro-batch (4 rows of 4,096 for this line: the shared shape's), a multiple of it -- so a
  source of many strata validates on more than 16 rows; a source's loss weights its strata by their shares (`val_4096/<source>`); over the
  recipe's sources `val_4096/mix` (weighted by the recipe, the held-out estimate of the training loss) and
  `val_4096/mean` (unweighted); at 32K the same under `val_32768/`.
* **Shared GPU:** with `guards.share_reserve_gb: 50` and the `shared` shapes measured for it (`nstrain tune
  --shared`), the line always trains with those shapes and holds all of the GPU but 50 GB: another process may start
  at any time and take up to 50 GB, never more; `guards.gpu_check_every: 1` re-holds what it frees at every update.
  The run.json here is the trunk's: the reserve and the shared shapes as `nstrain tune --shared` measured them.
