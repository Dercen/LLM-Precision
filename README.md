# ptq-bench

Post-training-quantization perplexity benchmark for LLMs: how much worse does a language
model get when its weights are stored at fewer bits, and how much of that loss does each
quantization method win back?

## Start here

| I want to… | Go to |
|---|---|
| **Read the results** (nothing to install) | The website: [dercen.github.io/LLM-Precision](https://dercen.github.io/LLM-Precision/), rebuilt from this repository after every push. The same pages are in [results/by-model/](results/by-model/README.md), and [results/README.md](results/README.md) explains the words. |
| **Measure one model myself** | [Quick start](#quick-start-new-machine) below: `./install.sh`, then `./ptq` opens a menu with a two-minute demo, or `./ptq ui` opens the same thing in your browser. No GPU? [Run it on Google Colab](https://colab.research.google.com/github/Dercen/LLM-Precision/blob/main/notebooks/ptq_colab.ipynb). |
| **Understand how it works** | [HOW-IT-WORKS.md](HOW-IT-WORKS.md): the whole pipeline in plain language, step by step. |
| **Run or extend the experiments** | [Running the matrix](#running-the-matrix), [Adding a model](#adding-a-model), and [docs/DESIGN.md](docs/DESIGN.md) for the protocol. |

Every document in the project, and who it is for:

| file | for | what it holds |
|---|---|---|
| [README.md](README.md) | everyone | setup, the menu, running experiments, adding models and datasets |
| [HOW-IT-WORKS.md](HOW-IT-WORKS.md) | newcomers | what happens between "run" and a number, in plain language |
| [results/README.md](results/README.md) | readers | the words in the tables and how to read a result |
| [results/by-model/](results/by-model/README.md) | readers | one page per model: best setting per bit width, every number, charts |
| [docs/RESULTS.md](docs/RESULTS.md) | researchers | findings with numbers, and the open items |
| [docs/DESIGN.md](docs/DESIGN.md) | researchers, developers | the measurement protocol, algorithms, datasets, reference numbers |
| [docs/EXPORT-MLIR.md](docs/EXPORT-MLIR.md) | compiler users | step by step: a quantized model as torch-mlir MLIR |
| [docs/SERVER.md](docs/SERVER.md) | cluster users | SLURM script, sharding, offline flags |
| [docs/USABILITY-TEST.md](docs/USABILITY-TEST.md) | maintainers | how to watch a newcomer use this and what to fix first |
| [notebooks/ptq_colab.ipynb](notebooks/ptq_colab.ipynb) | people without a GPU | the two-minute demo on a free Colab GPU |
| [docs/archive/](docs/archive/) | history | the original plans and fact checks; not maintained |

## Quick start (new machine)

Linux (or WSL on Windows) with an NVIDIA GPU is the normal setup; without a GPU the two
smallest models still run, slowly. One command does the whole setup:

```bash
git clone https://github.com/Dercen/LLM-Precision.git
cd LLM-Precision
./install.sh          # installs uv, picks the torch build for your NVIDIA driver, ~3 GB once
./ptq                 # the menu
```

`install.sh` ends with `./ptq doctor`, a checklist in plain words: GPU, disk, memory,
internet, what is already downloaded, with a fix for every red line. Run it again any time.
`./ptq` is the launcher: it sets up the environment and runs the program, so there is nothing
to `source` and no `uv run` to remember. `./ptq eval ...`, `./ptq run ...` and every other
command work the same way.

By hand, if you prefer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # uv: Python + environment manager
export PATH="$HOME/.local/bin:$PATH"
source scripts/env.sh                                    # cache paths under ~/ml (or $SCRATCH), hqq's build flag
uv sync --extra cu130 --extra hqq --extra dev            # driver older than R580: --extra cu126; no GPU: --extra cpu
uv run ptq env-check                                     # the exact, provenance-grade report (CI runs this)
uv run pytest -q -m "smoke and not gpu"                  # ~1 min; downloads opt-125m and wikitext2
```

- `source scripts/env.sh` is needed in **every new shell** that calls `uv run ptq` directly
  (`./ptq` does it for you). `DISABLE_CUDA=1` inside it is **hqq's build flag**, not a torch
  flag; `ptq env-check` asserts CUDA is live so that can never regress unnoticed.
- Nothing assumes this laptop: resident vs streamed, cache placement and time estimates
  are decided from the machine at run time, so a bigger or smaller GPU just moves where
  models stream.
- Models and datasets download on first use into `~/ml/hf` (`$SCRATCH/hf` on a cluster).
  `uv run ptq prefetch configs/experiments/<name>.yaml` fetches everything an experiment
  needs up front; do that before a queued or offline job.
- Result rows are one JSON file each under `results/raw/runs/`, so rows from several machines
  merge by plain git or rsync and `uv run ptq aggregate` dedupes them by `run_id`.
- Cluster specifics (SLURM script, `--shard k/n` across GPUs, offline flags): `docs/SERVER.md`.

## Easiest way in: the wizard

```bash
./ptq                 # or: uv run ptq wizard
```

The first question offers "Show me something in two minutes": opt-125m unmodified, then at
4 bits with plain rounding, both as quick previews, so the second number can be read against
the first. Before anything runs you see the plan and an estimated time, from the project's
own timing data, and what will be downloaded. Otherwise, arrow-key menus ask four things in plain words: which model, which text to test on, which
method, and how many bits. Grouping, GPTQ act-order and the calibration text sit behind one
"Advanced settings?" question and default to the recommended values (groups of 128, the
family's act-order default, C4). You confirm a one-line plan, watch it run, and get a table
with your perplexity next to the published number plus a sentence that says what it means,
for example "At 4 bits with GPTQ, groups of 128, opt-125m is 6.5% worse than the original on
WikiText-2: 29.45 vs 27.66. Between 5% and 20% is noticeable but usable." "Quick preview"
reads 20 passages when you just want a look (the row is marked partial). A wizard row is the
same row `ptq run` would produce, same ids and schema, so it lands in `results/raw/runs/` and
shows up in `ptq aggregate` and `ptq plot`. Add `--device cpu` to try it while the GPU is busy.
Anything that cannot run on this machine is explained in a sentence with the fix, not a
traceback.

## Without a terminal

Three ways in that never open a shell:

- **The website.** [dercen.github.io/LLM-Precision](https://dercen.github.io/LLM-Precision/)
  is `results/by-model/`, the charts, the results guide and HOW-IT-WORKS as web pages. CI
  rebuilds it after every push that changes `results/` (`.github/workflows/site.yml`; one-time
  setup: repository Settings → Pages → Source: GitHub Actions). `./ptq site` builds the same
  thing into `site/` locally.
- **The browser page.** `./ptq ui` starts a small local web page with the same choices as the
  menu: the two-minute demo, "Check this machine", model, text, method, bits, advanced settings
  behind a fold, a live log, and the verdict sentence. A run from the page is the same result
  row the menu produces, and the results pages refresh when it finishes.
  `scripts/desktop-shortcut.sh` puts "ptq-bench" in the Linux application menu so it starts
  with a click.
- **Google Colab.** [notebooks/ptq_colab.ipynb](https://colab.research.google.com/github/Dercen/LLM-Precision/blob/main/notebooks/ptq_colab.ipynb)
  runs the setup, the checks and the demo on a free GPU, nothing installed at home.

## Status and reproduced numbers (for researchers)

Milestones M0–M6 of the plan (now in `docs/archive/`) are complete (environment, first number, RTN, GPTQ, the
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
1.2% on average, opt-2.7b GPTQ4 per-row 12.917 vs 12.87. Start at `results/README.md`: one plain-language
page per model in `results/by-model/`, full tables in `results/tables/`, figures in `results/plots/`. Two published RTN cells (OPT-1.3B on PTB and C4) could not be
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
uv run ptq run configs/experiments/resident_full.yaml --dry-run   # estimate from results/raw/timing.json
uv run ptq run configs/experiments/resident_full.yaml             # resumable; Ctrl-C finishes the row
uv run ptq run ... --filter "algo=gptq bits=4" --shard 0/2         # subsets and sharding
uv run ptq aggregate && uv run ptq plot                           # tables/, by-model/, plots/
uv run ptq cache ls                                               # quantized-weight cache
```

M2 also settled a question the GPTQ README leaves open: its Tables 9 and 11 use the
`--new-eval` dataset variants, not `get_ptb`/`get_c4`. The plain keys miss by 7–17%
while the `_new` keys match to 0.1% on two independent columns each. See docs/DESIGN.md §7.

## Exporting a quantized model to torch-mlir

Step-by-step guide: [docs/EXPORT-MLIR.md](docs/EXPORT-MLIR.md).

`ptq export-mlir` takes a model and a bit-width config, runs the pipeline's own quantizer
once more while recording the grid it rounds on, and writes three things: a
`torch.nn.Module` whose target Linears hold **integer codes plus scale/zero-point** with the
dequantization performed in `forward` (not folded into a float weight), the same model as
**torch-dialect MLIR** via `torch_mlir.fx.export_and_import`, and the per-layer accuracy as
JSON keyed by module name. The dequantized weights are bitwise the ones the perplexity rows
were measured with; `layers.json` records that check per module.

```bash
uv sync --extra cu130 --extra hqq --extra dev --extra mlir     # + the torch-mlir dev wheel and ml_dtypes
uv run ptq export-mlir --model opt-125m --bits 4 --group-size 128 --out exports/opt125m-rtn4-g128
uv run ptq export-mlir --model opt-125m --bits 4 --algo gptq --act-order --calib c4 \
    --score-windows 4 --out exports/opt125m-gptq4-ao             # per-layer output error on 4 windows
uv run ptq export-mlir --model facebook/opt-125m --bits 3 --algo hqq --no-mlir --out exports/x   # Module + JSON only
```

`--model` is a `configs/models` key, a Hugging Face id or a local directory; `--algo` is
`rtn` (default), `gptq`, `awq_lite` or `hqq` with the same options as `ptq eval`. Output:

| file | contents |
|---|---|
| `model.mlir` | raw torch dialect, `func.func @main(%input_ids)`, weights embedded as dense resources (`--bytecode` adds `model.mlirbc`; `--dynamic-seqlen` makes the sequence length symbolic) |
| `layers.json` | per quantized module: bits, group size, scheme, dequant op, storage dtype, `quant_min/max`, relative and max weight error, SNR, scale/zero ranges, codes used, `roundtrip_max_abs_diff` / `bitwise_equal_to_evaluated`, and with `--score-windows N` the relative output error `‖Q(W)x − Wx‖/‖Wx‖` on N windows |
| `export.json` | the run's `quant_key`, quant and calibration fields, op counts, versions, provenance, and any `results/raw/runs` perplexity rows with the same `quant_key` |

From Python: `ptqbench.export.export_quantized(run, out_dir, device=..., options=ExportOptions(...))`
returns the module (`result.model`) and the records. `uv run torch-mlir-opt` is not shipped;
a consumer continues with `torch-match-quantized-custom-ops` and
`torchdynamo-export-to-torch-backend-pipeline`, which the test runs through to linalg.

**What maps onto torch-mlir's quantized-op support, and what does not** (torch-mlir
20260930.892, torch 2.14):

- Per-row grids (`group_size=-1`) are per-output-channel affine quantization with an
  integer zero-point: `quantized_decomposed.dequantize_per_channel(axis=0)`, which torch-mlir
  imports as a first-class torch-dialect op and lowers to linalg (`extui/subi/sitofp/mulf`).
- Grouped grids (g128, g64) have a `(rows, n_groups)` scale that ATen's quantized tensor
  types cannot hold; `quantized_decomposed.dequantize_per_channel_group` can, and torch-mlir
  imports and lowers it. This is the default for every RTN, AWQ-lite and non-act-order GPTQ row.
- 2–7-bit codes have no PyTorch dtype. They sit in a `uint8` container with
  `quant_min=0, quant_max=2^bits−1` on the op (the PT2E/ExecuTorch convention). That is exact
  for the values; only the type system sees 8 bits.
- **GPTQ `act_order` + group size** assigns scattered columns to each group (the Llama configs
  default to `act_order: true`). No `quantized_decomposed` op takes a per-column group index,
  so those layers are written as explicit ATen ops (`index_select` of scale/zero, `sub`, `mul`)
  with a `g_idx` buffer. torch-mlir lowers them, but as plain arithmetic, not as quantization.
  `--static-groups` keeps groups contiguous and the decomposed op. `--repr aten` writes every
  layer this way.
- **HQQ** zero-points are floats (its optimizer's output), so hqq layers also take the explicit
  ATen form; `zero_point_domain: "float"` in `layers.json`.
- **Activations are never quantized** by this benchmark, so every export is
  dequantize-then-float-matmul. torch-mlir's `FuseQuantizedOps` integer matmul needs both operands
  quantized and does not fire. The pre-rounding fp weight is the error reference for every layer
  (for AWQ-lite that is the scaled, clipped weight, noted as `error_reference`).

Practicalities: the `.mlir` text holds every weight as hex (opt-125m: 480 MB and 14 s on the
CPU, where the model is fp32; about 0.3 GB on the GPU at fp16), so `/exports/`, `*.mlir` and
`*.mlirbc` are git-ignored. Host RAM needs the model, the codes, and with `--score-windows`
another 1.5× the target weights (skipped with a note if that would not fit). An export on the
CPU has `dtype=torch.float32`, a different `quant_key` from the GPU fp16 rows, so
`perplexity_rows` only joins when the export runs on the same device class as the benchmark.
torch-mlir has no PyPI release: the `mlir` extra pins one dev wheel from its GitHub release
page (see `pyproject.toml`), and bf16 models need the `ml_dtypes` package it adds.

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
