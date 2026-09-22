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

## Llama-2-7b and Llama-3.1-8B

_(filled in when the `streamed_llama` run completes)_ On the NousResearch mirror of
Llama-2-7b (the official repo is gated; see README), fp16 WikiText-2 = 5.4721 vs 5.47.
AWQ-lite 4-bit g128 = 5.6380, against the AWQ paper's 5.60 for full AWQ, RTN 5.73 and
GPTQ 5.69.

## Open items

- GPTQModel cross-check (`ptq verify-backend`) needs a CUDA 13.0 toolkit for its JIT;
  bitsandbytes and torchao rows need neither and are the next backends to add.
- The full opt-6.7b grid (`streamed_opt67b.yaml`) beyond the gate cells.
- Official Llama weights once Hub access exists: a repeated fp16 cell checks the mirrors.
- HQQ's 3-bit collapse: whether its optimizer defaults or its grid are responsible.
