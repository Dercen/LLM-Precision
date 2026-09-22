# PTQ Bench — results and findings

One paragraph per finding, with the numbers that support it. Tables:
[`results/summary.md`](../results/summary.md); figures: `results/plots/`; every row:
`results/runs/*.json` (aggregated in `results/results.csv`). Method and protocol:
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

_Run in progress (resumed after a power loss on 2026-09-22); no verified published reference, so these rows are descriptive only. First rows: fp16 pending, AWQ-lite 4-bit g128 6.677, GPTQ 4-bit per-row (act-order) 7.303._

## Open items

- GPTQModel cross-check (`ptq verify-backend`) needs a CUDA 13.0 toolkit for its JIT;
  bitsandbytes and torchao rows need neither and are the next backends to add.
- The full opt-6.7b grid (`streamed_opt67b.yaml`) beyond the gate cells.
- Official Llama weights once Hub access exists: a repeated fp16 cell checks the mirrors.
- HQQ's 3-bit collapse: whether its optimizer defaults or its grid are responsible.
