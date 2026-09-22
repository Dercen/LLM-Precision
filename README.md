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
| M4 Runner + plots + AWQ-lite + HQQ | next |

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

opt-350m lands the same way: fp16 22.0017 / 22.00, RTN4 25.9412 / 25.94, GPTQ4 24.3851 / 24.24,
RTN3 64.5576 / 64.57, GPTQ3 32.7896 / 33.79. Three calibration seeds on opt-125m GPTQ4 give
31.55 / 31.87 / 31.24 (stdev 0.32), so the +1.37% is inside seed variance.

Streamed evaluation (weights in host RAM, one decoder block on the GPU at a time) matches
resident evaluation to 0.0 on three topologies, and offloaded GPTQ matches resident GPTQ on
every weight — the 7B path is the same computation, not an approximation of it.

**Llama family finding:** on SmolLM2-135M plain GPTQ4 is *worse* than RTN4 (27.91 vs 26.61)
while GPTQ4 with `act_order` is 24.18. Llama configs therefore default to `act_order: true`.
See PLAN.md §6.

M2 also settled a question the GPTQ README leaves open: its Tables 9 and 11 use the
`--new-eval` dataset variants, not `get_ptb`/`get_c4`. The plain keys miss by 7–17%
while the `_new` keys match to 0.1% on two independent columns each. See PLAN.md §7.

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

## Hardware

Developed on Pop!_OS 24.04, i9-14900HX, 31 GB RAM, RTX 4070 Laptop (8 GB, sm_89),
driver 595.84 / CUDA 13.2, torch 2.14.0+cu130. Models up to opt-1.3b run resident
in fp16; 6.7B-8B run through block-streamed evaluation. See PLAN.md §2.
