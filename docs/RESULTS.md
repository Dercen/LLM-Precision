# PTQ Bench — results and findings

One paragraph per finding, with the numbers that support it. Per-model pages:
[`results/by-model/`](../results/by-model/README.md); tables:
[`results/tables/summary.md`](../results/tables/summary.md); figures: `results/plots/`; every row:
`results/raw/runs/*.json` (aggregated in `results/tables/results.csv`). Method and protocol:
[DESIGN.md](DESIGN.md). All numbers are fp16 (bf16 for Llama-3.1) on an RTX 4070 Laptop
GPU (8 GB), WikiText-2 / C4 / PTB test perplexity at 2048-token windows, unless stated.

## The pipeline reproduces the papers

Every fp16 baseline matches its published value to within 0.04% — opt-125m 27.6559
(27.65), opt-350m 22.0017 (22.00), opt-1.3b 14.6239 (14.63), opt-2.7b 12.4711 (12.47),
opt-6.7b 10.8603 (10.86), Llama-2-7b 5.4721 (5.47) — across 13 model/dataset cells.
RTN and GPTQ on opt-125m reproduce the GPTQ paper's Table 3 to +0.008% (RTN4 37.2831 vs
37.28) and +1.4% (GPTQ4 31.5475 vs 31.12; three calibration seeds give 31.24–31.87, so
that is seed variance). Across the 61 cells with a published number, GPTQ's mean
deviation is 1.2%.

Two prerequisites for those matches are easy to get wrong on a GPU and are enforced in
code: TF32 is disabled on every knob (torch 2.14 ships cuDNN with TF32 on), and the
quantizer's scale/zero divisions go through float64 because CUDA's fp32 scalar division
is one ULP short of correctly rounded — enough to make the same weight quantize
differently on GPU and CPU.

## Which dataset variants the GPTQ paper actually used

The GPTQ README does not say whether its PTB and C4 tables used `get_ptb`/`get_c4` or the
`--new-eval` variants. On opt-125m the `_new` variants reproduce the published numbers
(PTB fp16 38.9917 vs 38.99, RTN4 53.8840 vs 53.89; C4 26.5637 vs 26.56, RTN4 33.8850 vs
33.91) while the plain variants miss by 7–17%. Every OPT PTB/C4 comparison here uses
`ptb_new` and `c4_new`. The paper's Appendix A.2.1 describes the `"\n\n"` join; the
released code's `--new-eval` joins with a space, and the code is what the tables report.

## Two published cells do not reproduce

OPT-1.3B RTN 4-bit per-row: PTB 75.33 measured vs 57.30 published; C4 27.49 vs 24.51.
The same quantized weights reproduce the paper's WikiText-2 cell to 0.03% (48.18 vs
48.17); the model's fp16 and GPTQ cells match on all three datasets to under 1%; and the
measured values move by 0.05% under eager attention and fp32 instead of SDPA and fp16.
OPT-2.7B shows the same pattern mildly (PTB +4.4%, C4 +2.2%, WikiText-2 exact). The
cells are flagged `unreproduced` in `references/literature.yaml` and gate nothing.

## GPTQ on the Llama family needs act-order

On SmolLM2-135M, plain GPTQ 4-bit per-row is worse than RTN (27.91 vs 26.61) although its
per-layer reconstruction error is lower in 28 of 28 layers; with `act_order` it is 24.18,
and at g64 the pattern repeats (20.25 vs 19.95 vs 19.50 with act-order). Fixed column
order lets error feedback pile onto the outlier channels that Llama-style models
concentrate late in the block; quantizing the largest-Hessian-diagonal columns first
avoids it. Llama configurations default to `act_order: true`; OPT keeps the paper's
`false`.

## The resident matrix (five models, 436 rows)

opt-125m, opt-350m, SmolLM2-135M, opt-1.3b, opt-2.7b × fp16 / RTN / GPTQ / AWQ-lite /
HQQ × 8/4/3/2 bits × per-row and g128 (g64 at 2-bit) × WikiText-2, C4, PTB; 164 min plus
an 89 min rerun. GPTQ wins 4-bit g128 on opt-350m/1.3b/2.7b, AWQ-lite on opt-125m; AWQ-lite
beats RTN at the same grid on every model (opt-125m 3-bit g128: 51.20 → 36.97). At 3 bits
HQQ's defaults collapse on the OPT models (96–299, worse than RTN g128) — noted, not
investigated. At 2-bit g128 GPTQ stays near 75 on opt-2.7b while RTN and HQQ exceed 10⁴.
Per-configuration GPTQ cost on this GPU: 17 s (125m), 55 s (350m), 172 s (1.3b), 342 s (2.7b).

4-bit g128 on WikiText-2:

| model | fp16 | RTN | GPTQ | AWQ-lite | HQQ |
|---|---|---|---|---|---|
| opt-125m | 27.66 | 30.48 | 29.45 | **29.30** | 30.54 |
| opt-350m | 22.00 | 24.51 | **23.21** | 23.60 | 24.24 |
| opt-1.3b | 14.62 | 15.29 | **14.87** | 15.00 | 15.14 |
| opt-2.7b | 12.47 | 13.02 | **12.62** | 12.86 | 13.28 |

## 6.7B and 7B models on an 8 GB GPU

opt-6.7b (13.3 GB fp16) evaluates block-streamed at a peak of 1.95 GB VRAM and 13.6 GB
host RAM, matching the paper on all nine cells: WikiText-2 fp16 10.8603 / RTN4 12.0992 /
GPTQ4 11.4774 vs 10.86 / 12.10 / 11.39; PTB 15.7699 / 18.8578 / 16.4841 vs 15.77 / 18.84 /
16.56; C4 12.7121 / 14.3730 / 13.1715 vs 12.71 / 14.36 / 13.18. GPTQ on 6.7B takes 857 s.
Streamed and resident evaluation agree to 0.0 on every model where both are possible,
and offloaded GPTQ produces bit-identical weights to resident GPTQ.

## Llama-2-7b on an 8 GB GPU

Run on the NousResearch mirror (the official repo is gated; see README), streamed, GPTQ
with act-order as the family default. Against the AWQ paper's Table 4 and OmniQuant's
Table 1 on WikiText-2:

| config | ours | published |
|---|---|---|
| fp16 | **5.472** | 5.47 |
| RTN 4-bit g128 | **5.724** | 5.73 (AWQ) / 5.72 (OmniQuant) |
| GPTQ 4-bit g128 | **5.635** | 5.69 (AWQ) / 5.61 (OmniQuant) |
| AWQ-lite 4-bit g128 | 5.638 | 5.60 (AWQ, full method, pile_val calibration) |
| RTN 4-bit per-row | 6.116 | 6.11 |
| GPTQ 4-bit per-row | **5.830** | 5.83 |
| RTN 3-bit g128 | 6.664 | 6.66 |
| GPTQ 3-bit g128 | 6.362 | 6.29 |
| AWQ-lite 3-bit g128 | 6.333 | 6.24 (AWQ) |
| RTN 3-bit per-row | 542.7 | 539.48 |
| GPTQ 3-bit per-row | 8.545 | 8.37 |
| RTN 2-bit g64 | 432.9 | 431.97 |
| GPTQ 2-bit g64 | 26.8 | 20.85 (OmniQuant, protocol uncertain) |
| GPTQ 2-bit g128 | 54.8 | 36.77 (OmniQuant, protocol uncertain) |

Every RTN cell reproduces to 0.6% or better, which also vouches for the mirror weights
(RTN has no calibration to absorb a difference). GPTQ at 3–4 bits sits within 2% of
print; the exact 5.830 on W4 per-row identifies the AWQ paper's GPTQ baseline as the
act-order variant, and those literature rows now say so. The 2-bit GPTQ cells are 30–50%
above OmniQuant's numbers: 2-bit is where GPTQ is most sensitive to calibration and
column order, and OmniQuant does not state how its baselines were produced, so those
cells are descriptive. AWQ-lite — C4-calibrated, scoring group outputs rather than the
enclosing block — lands within 0.7% (4-bit) and 1.5% (3-bit) of full AWQ.

Each 7B GPTQ configuration took about 15 minutes streamed; peak VRAM stayed under 2 GB
for evaluation and under 5.5 GB while quantizing, with the hidden-state cache in host RAM.

## Llama-3.1-8B

_11 of the 15 `streamed_llama.yaml` cells are in (the run was resumed after a power loss on 2026-09-22; the four RTN cells at 3 and 2 bits remain). No verified published reference, so these rows are descriptive only. fp16 6.240; at 4 bits GPTQ g128 6.609, AWQ-lite g128 6.677, RTN g128 6.825, GPTQ per-row (act-order) 7.303._

## Size, grouping and which layers are sensitive (for the vault hardware)

Three questions the hardware side asked, answered on the two smallest models, the ones a
simulated vault system can hold. Every row now carries `bits_per_weight` and
`weight_bytes`, computed from the model's config the way the hand-off format stores
weights: packed codes, one 16-bit scale and one zero-point per group, everything else
at 16 bits (`src/ptqbench/analysis/footprint.py`). The charts are
`results/plots/<model>_frontier.png`, `_groupsize.png` and `_sensitivity.png`; the sweeps
are `configs/experiments/groupsize_smollm2.yaml` and `groupsize_opt125m.yaml` (109 cells,
25 minutes on the laptop GPU).

**The embedding table caps the compression.** SmolLM2-135M is 269 MB in fp16. At 4-bit
g32 the quantized layers bring it to 118 MB, not to a quarter, because the tied
49152 × 576 embedding (57 MB) and the norms stay at 16 bits; OPT-125m goes 250 → 126 MB
for the same reason (its 50272 × 768 embedding is 77 MB). Below about 2.2× the only lever
left is the embedding, and no setting of the linear layers changes that.

**Grouping is a tiling decision, and the two families disagree.** 4-bit on WikiText-2,
group size across (bits per weight: per-row 4.03, g192 4.10, g96 4.21, g64 4.31, g48 4.42,
g32 4.62):

| model | method | per-row | g192 | g96 | g64 | g48 | g32 |
|---|---|---|---|---|---|---|---|
| SmolLM2-135M (fp16 15.42) | RTN | 26.61 | 22.31 | 20.88 | 19.95 | 19.28 | 18.65 |
| | GPTQ, act-order | 24.18 | 21.56 | 20.36 | 19.50 | 18.63 | 17.97 |
| | AWQ-lite | — | 18.71 | 17.84 | 17.50 | 17.17 | 16.89 |
| OPT-125m (fp16 27.66) | RTN | 37.28 | 30.91 | 29.95 | 29.85 | 29.85 | 29.78 |
| | GPTQ | 31.55 | 29.91 | 29.29 | 29.53 | 28.82 | 28.85 |
| | AWQ-lite | — | 29.54 | 29.22 | 28.59 | 28.89 | 28.90 |

OPT-125m was also run at g256 and g128 (RTN 31.66 / 30.48, AWQ-lite 29.94 / 29.30). On
SmolLM2 every halving of the group keeps paying all the way down to 32, with every
method: RTN gains 8 points from per-row to g32, and AWQ-lite still gains 1.8 between
g192 and g32. On OPT-125m the curve flattens at g64: AWQ-lite 28.59 there is within 5%
of fp16, and g48 and g32 are no better. At 3 bits the grouping decides whether the model
works at all: from per-row to g32, SmolLM2 RTN goes 2180 → 44.9 and OPT-125m RTN
1277 → 36.6; AWQ-lite 3-bit g32 reaches 24.0 on SmolLM2 (+56%) and 31.8 on OPT-125m
(+15%). At 2 bits no grouping rescues either model (the best cells are GPTQ g32: 3607 on
SmolLM2, 110 on OPT-125m).

**The frontier.** The smallest setting within 10% of fp16 on SmolLM2 is AWQ-lite 4-bit
g32 at 118 MB (4.62 bits per weight, perplexity 16.89); within 5% only 8-bit qualifies
(163 MB). On OPT-125m, AWQ-lite 4-bit g64 at 126 MB (4.31 bits per weight, 28.59) is
within 5%, and AWQ-lite 3-bit g64 at 116 MB is within 25%. For a vault design: SmolLM2
wants groups of 32 to 48 with the scales co-located per tile, OPT-family models are
served by groups of 64, 3-bit is an option only with small groups and a calibrated
method (AWQ-lite or GPTQ), and 2-bit is not an option for models this small.

**Which layers are sensitive.** `scripts/sensitivity-sweep.sh` exported both models at
4, 3 and 2 bits with RTN and GPTQ, scoring every quantized layer's relative output error
‖Q(W)x − Wx‖ / ‖Wx‖ on 16 windows of real text (`results/sensitivity/<model>.md`, the
heatmaps in `results/plots/`). The pattern is a property of the architecture, not of the
bit width: the ranking of projections is the same at 4, 3 and 2 bits and under both
methods.

| model, RTN 4-bit | most sensitive | least sensitive | spread between layers |
|---|---|---|---|
| SmolLM2-135M (g64) | `v_proj` 0.145, then `up_proj` 0.100 | `k_proj` 0.063, `q_proj` 0.073 | 0.028 (block 11 `down_proj`) to 0.173 (block 12 `v_proj`), 6× |
| OPT-125m (g128) | `fc2` 0.134, then `v_proj` 0.118, `out_proj` 0.115 | `fc1` 0.048, `k_proj` 0.059 | 0.024 (block 2 `fc2`) to 0.245 (block 6 `fc2`), 10× |

On SmolLM2 the value projection is the layer to protect: 2.3× the error of the key
projection at the same bits, in every block. On OPT-125m it is the second feed-forward
matrix, and specifically in blocks 5 to 10, where `fc2` carries the ten most sensitive
layers of the model; blocks 0 to 2 and 11 are the most tolerant. Per block the mean
error varies much less (SmolLM2 0.076 to 0.103, OPT-125m 0.066 to 0.112), so mixed
precision by projection type is worth more than mixed precision by block. GPTQ lowers
every layer's output error against RTN by the same factor across the model (SmolLM2
1.5× at 4 and 3 bits, OPT-125m 1.3×) without changing which layers are sensitive. For
the vault compiler: give `v_proj` (Llama family) and `fc2` (OPT family) the extra bit or
the smaller group, and spend the savings on `k_proj`, `q_proj` and `fc1`.

## Open items

- GPTQModel cross-check (`ptq verify-backend`) needs a CUDA 13.0 toolkit for its JIT;
  bitsandbytes and torchao rows need neither and are the next backends to add.
- The full opt-6.7b grid (`streamed_opt67b.yaml`) beyond the gate cells.
- Official Llama weights once Hub access exists: a repeated fp16 cell checks the mirrors.
- HQQ's 3-bit collapse: whether its optimizer defaults or its grid are responsible.
- The embedding table: quantizing it (8-bit, or 4-bit with its own scales) is the only way
  past the ~2.2× size cap on the small models; nothing in the pipeline covers it yet.
- Activation quantization (W8A8 / W4A8), so the vault compiler can emit integer matmuls;
  the export format leaves room for it and no row measures it.
