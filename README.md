

# ptq-bench

Post-training-quantization perplexity benchmark for LLMs: how much does perplexity
degrade as weights are stored at fewer bits, and how much of that loss does each
quantization algorithm recover?

Method and protocol: [docs/DESIGN.md](docs/DESIGN.md). Findings with numbers: [docs/RESULTS.md](docs/RESULTS.md).
Running on a cluster: [docs/SERVER.md](docs/SERVER.md). The original plans are archived under `docs/`.

## Status

Milestones M0–M6 of the plan are complete (environment, first number, RTN, GPTQ, the
resident matrix, opt-6.7b streamed, Llama-2 on the mirror); the Llama-3.1-8B rows are the
last run in progress. Open items are listed at the end of [docs/RESULTS.md](docs/RESULTS.md).


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
reproduced despite the same weights matching on WikiText-2 — see docs/RESULTS.md.

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
See docs/RESULTS.md.

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
while the `_new` keys match to 0.1% on two independent columns each. See docs/DESIGN.md §7.

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

## Quick start (new machine)

Linux with an NVIDIA GPU; check the driver with `nvidia-smi` first.

```bash
# 1. tools
curl -LsSf https://astral.sh/uv/install.sh | sh          # uv: Python + environment manager
export PATH="$HOME/.local/bin:$PATH"

# 2. code
git clone https://github.com/Dercen/LLM-Precision.git
cd LLM-Precision

# 3. environment (one-time; ~3 GB of wheels, torch build chosen by extra)
source scripts/env.sh                                    # cache paths under ~/ml (or $SCRATCH), hqq's build flag
uv sync --extra cu130 --extra hqq --extra dev            # driver older than R580: --extra cu126; no GPU: --extra cpu

# 4. check
uv run ptq env-check                                     # torch+cu130, the GPU, TF32 off
uv run pytest -q -m "smoke and not gpu"                  # ~1 min; downloads opt-125m and wikitext2

# 5. use
uv run ptq                                               # the wizard
```

- `source scripts/env.sh` is needed in **every new shell** (or add it to `~/.bashrc`); it is
  safe to repeat. `DISABLE_CUDA=1` inside it is **hqq's build flag**, not a torch flag —
  `ptq env-check` asserts CUDA is live so that can never regress unnoticed.
- Nothing assumes this laptop: resident vs streamed, cache placement and time estimates
  are decided from the machine at run time, so a bigger or smaller GPU just moves where
  models stream.
- Models and datasets download on first use into `~/ml/hf` (`$SCRATCH/hf` on a cluster).
  `uv run ptq prefetch configs/experiments/<name>.yaml` fetches everything an experiment
  needs up front — do that before a queued or offline job.
- Result rows are one JSON file each under `results/runs/`, so rows from several machines
  merge by plain git or rsync and `uv run ptq aggregate` dedupes them by `run_id`.
- Cluster specifics (SLURM script, `--shard k/n` across GPUs, offline flags): `docs/SERVER.md`.

## Adding a model

Models are one small YAML file each in `configs/models/`. Copy an existing one and edit it:

```yaml
# configs/models/opt-6.7b.yaml
repo: facebook/opt-6.7b            # Hugging Face id
revision: a45aa65bbeb77c...        # optional but recommended: pins the exact weights
gated: false                       # true if the Hub asks you to accept a licence
mirror: SomeOrg/same-weights       # optional ungated copy, used when the repo is gated
tokenizer_class: GPT2Tokenizer     # what `type(tokenizer).__name__` prints; recorded on rows
family: opt                        # opt or llama (see below)
dtype: auto                        # auto = fp16, or bf16 for Llama-3-style models
extra_group_sizes: [64]            # optional: extra group sizes if 128 does not divide the hidden size
notes: anything worth remembering
```

The file name (without `.yaml`) is the model's key: it appears in the wizard's menu and
in experiment files automatically. Weights download on first use (or `uv run ptq prefetch
<experiment>`); the runner decides by itself whether the model fits on the GPU or has to
be streamed through it one block at a time.

`family` tells the code where the transformer blocks are and which layers to quantize.
Anything built like OPT or like Llama/Mistral/Qwen2 works as is. A genuinely different
architecture needs one entry in `src/ptqbench/models/families.py` (the block list path,
the linear layer names, the norms that come after the last block, and the AWQ scale
groups) — copy the `LLAMA` entry and adjust the names.

## Adding a dataset

Datasets are functions in `src/ptqbench/data/datasets.py`. Each one loads text, joins
it into a single stream, tokenizes it once and returns a `TokenStream`:

```python
@register("my_corpus")
def _my_corpus(tokenizer, seqlen: int) -> TokenStream:
    ds = _hf_load("some-org/some-dataset", split="test")      # any Hugging Face dataset
    text = "\n\n".join(ds["text"])                            # how the documents are joined
    ids = _truncate_to_windows(_encode(tokenizer, text), seqlen)
    return TokenStream("my_corpus", ids, seqlen, "test", '"\n\n".join', len(ds))
```

That is all: the key `my_corpus` is then accepted by `--dataset`, by experiment YAMLs
and by the wizard's checkbox list (add a one-line description to `DATASETS` in
`src/ptqbench/wizard.py` if you want it labelled there). If a paper reports perplexity
on it, add those rows to `references/literature.yaml` and `ptq aggregate` will show the
published number and the delta next to yours. Calibration sets for GPTQ/AWQ live in
`src/ptqbench/data/calibration.py` the same way.

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
in fp16; 6.7B-8B run through block-streamed evaluation. See docs/DESIGN.md §2.
