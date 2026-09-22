# ptq-bench

Post-training-quantization perplexity benchmark for LLMs: how much does perplexity
degrade as weights are stored at fewer bits, and how much of that loss does each
quantization algorithm recover?

See [PLAN.md](PLAN.md) for the full design; [docs/verified-facts-2026-09-20.md](docs/verified-facts-2026-09-20.md)
for the web-verified version pins and reference numbers behind it.

## Status

| Milestone | State |
|---|---|
| M0 Environment | done |
| M1 First number | done |
| M2 Data + RTN + schema + CI | done |
| M3 GPTQ + block streaming | done |
| M4 Runner + plots + AWQ-lite + HQQ | done — 436-row resident matrix, 0 failures |
| M5 opt-6.7b streamed | done — nine cells within 0.8% of the paper |
| M6 Llama-2 / Llama-3.1 (mirrors) | running |

### Reproduced published numbers (opt-125m, fp16 on an RTX 4070)

| Dataset | Algo | Ours | Published | Δ |
|---|---|---|---|---|
| wikitext2 | fp16 | 27.6559 | 27.65 | +0.02% |
| wikitext2 | RTN4 per-row | 37.2831 | 37.28 | +0.008% |
| wikitext2 | RTN8 | 27.6595 | (≤0.05 of fp) | 0.0036 |
| ptb_new | fp16 | 38.9917 | 38.99 | +0.00% |
| ptb_new | RTN4 per-row | 53.8840 | 53.89 | −0.01% |
| c4_new | fp16 | 26.5637 | 26.56 | +0.01% |
| c4_new | RTN4 per-row | 33.8850 | 33.91 | −0.07% |
| wikitext2 | GPTQ4 per-row | 31.5475 | 31.12 | +1.37% |
| wikitext2 | GPTQ3 per-row | 53.0273 | 53.85 | −1.53% |
| ptb_new | GPTQ4 per-row | 45.7609 | 45.17 | +1.31% |
| c4_new | GPTQ4 per-row | 29.1953 | 29.22 | −0.08% |
| wikitext2 (opt-1.3b) | fp16 | 14.6239 | 14.63 | −0.04% |
| wikitext2 (opt-2.7b) | fp16 | 12.4711 | 12.47 | +0.01% |
| **wikitext2 (opt-6.7b, streamed)** | **fp16** | **10.8603** | **10.86** | **+0.003%** |
| wikitext2 (opt-6.7b, streamed) | RTN4 per-row | 12.0992 | 12.10 | −0.01% |
| wikitext2 (opt-6.7b, streamed) | GPTQ4 per-row | 11.4774 | 11.39 | +0.77% |
| **wikitext2 (Llama-2-7b mirror, streamed)** | **fp16** | **5.4721** | **5.47** | **+0.04%** |

The opt-6.7b row is the point of the streamed tier: 13.3 GB of fp16 weights evaluated on an
8 GB GPU at a peak of 1.95 GB VRAM, matching the published number to four decimals.

### The resident matrix (five models × five methods × four bit widths × three datasets)

436 rows, 61 of them against a published number: fp16 within 0.04% everywhere, GPTQ within
1.2% on average, opt-2.7b GPTQ4 per-row 12.917 vs 12.87. Full tables in `results/summary.md`,
figures in `results/plots/`. Two published RTN cells (OPT-1.3B on PTB and C4) could not be
reproduced despite the same weights matching on WikiText-2 — see PLAN.md §13.

4-bit g128 on WikiText-2 (lower is better):

| model | fp16 | RTN | GPTQ | AWQ-lite | HQQ |
|---|---|---|---|---|---|
| opt-125m | 27.66 | 30.48 | 29.45 | **29.30** | 30.54 |
| opt-350m | 22.00 | 24.51 | **23.21** | 23.60 | 24.24 |
| opt-1.3b | 14.62 | 15.29 | **14.87** | 15.00 | 15.14 |
| opt-2.7b | 12.47 | 13.02 | **12.62** | 12.86 | 13.28 |

opt-350m lands the same way: fp16 22.0017 / 22.00, RTN4 25.9412 / 25.94, GPTQ4 24.3851 / 24.24,
RTN3 64.5576 / 64.57, GPTQ3 32.7896 / 33.79. Three calibration seeds on opt-125m GPTQ4 give
31.55 / 31.87 / 31.24 (stdev 0.32), so the +1.37% is inside seed variance.

Streamed evaluation (weights in host RAM, one decoder block on the GPU at a time) matches
resident evaluation to 0.0 on three topologies, and offloaded GPTQ matches resident GPTQ on
every weight — the 7B path is the same computation, not an approximation of it.

**Llama family finding:** on SmolLM2-135M plain GPTQ4 is *worse* than RTN4 (27.91 vs 26.61)
while GPTQ4 with `act_order` is 24.18. Llama configs therefore default to `act_order: true`.
See PLAN.md §6.

AWQ-lite beats RTN at the same grid on both families: opt-125m 4-bit g128 30.48 → 29.30,
3-bit g128 51.20 → 36.97, SmolLM2 4-bit g64 19.95 → 17.47.

## Running the matrix

```bash
uv run ptq run configs/experiments/resident_full.yaml --dry-run   # estimate from results/timing.json
uv run ptq run configs/experiments/resident_full.yaml             # resumable; Ctrl-C finishes the row
uv run ptq run ... --filter "algo=gptq bits=4" --shard 0/2         # subsets and sharding
uv run ptq aggregate && uv run ptq plot                           # results.csv, summary.md, plots/
uv run ptq cache ls                                               # quantized-weight cache
```

M2 also settled a question the GPTQ README leaves open: its Tables 9 and 11 use the
`--new-eval` dataset variants, not `get_ptb`/`get_c4`. The plain keys miss by 7–17%
while the `_new` keys match to 0.1% on two independent columns each. See PLAN.md §7.

## Easiest way in: the wizard

```bash
source scripts/env.sh
uv run ptq            # or: uv run ptq wizard
```

Arrow-key menus for model → datasets → method → bits → group size, a one-line plan to
confirm, then a table with your perplexity next to the published number. "Quick preview"
runs 20 windows (marked partial) when you just want a look. A wizard row is the same row
`ptq run` would produce — same ids, same schema — so it lands in `results/runs/` and shows
up in `ptq aggregate` and `ptq plot`. Add `--device cpu` to try it while the GPU is busy.

## Quick start

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source scripts/env.sh
uv sync --extra cu130 --extra hqq --extra dev    # --extra cpu on a machine without CUDA
uv run ptq env-check
uv run ptq eval --model facebook/opt-125m --dataset wikitext2
uv run pytest -q
```

`scripts/env.sh` is safe to source repeatedly and works unchanged on a cluster
(it picks up `$SCRATCH` when set). `DISABLE_CUDA=1` in it is **hqq's build flag**,
not a torch flag — `ptq env-check` asserts CUDA is live so it cannot regress unnoticed.

## Gated models (needs you)

`meta-llama/Llama-2-7b-hf` and `meta-llama/Llama-3.1-8B` are gated. Until this machine
is logged in and the licences are accepted, the runner substitutes the pinned ungated
mirrors (`NousResearch/Llama-2-7b-hf`, `unsloth/Meta-Llama-3.1-8B`) and records
`loaded_from` on every row. To use the official repos:

```bash
source scripts/env.sh && uv run hf auth login     # token lands in $HF_HOME/token
# then accept the licences at huggingface.co/meta-llama/Llama-2-7b-hf and /Llama-3.1-8B
uv run ptq run configs/experiments/streamed_llama.yaml --filter algo=fp   # repeats fp16 on the official weights
```

The repeated fp16 cell is the check that the mirror weights are the same weights.

## Hardware

Developed on Pop!_OS 24.04, i9-14900HX, 31 GB RAM, RTX 4070 Laptop (8 GB, sm_89),
driver 595.84 / CUDA 13.2, torch 2.14.0+cu130. Models up to opt-1.3b run resident
in fp16; 6.7B-8B run through block-streamed evaluation. See PLAN.md §2.
