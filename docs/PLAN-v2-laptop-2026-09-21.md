> **Archived 2026-09-22.** This plan was executed through M6. The living documents are [DESIGN.md](DESIGN.md) (protocol, algorithms, datasets, references), [RESULTS.md](RESULTS.md) (findings) and [SERVER.md](SERVER.md) (the M7 hand-off).

# PTQ Bench: bit-precision study for LLMs

Revision 2, 2026-09-21. Supersedes the 2026-09-20 plan (archived at `docs/PLAN-v1-windows-cpu-2026-09-20.md`), which targeted a Windows/OneDrive CPU-only laptop. The measurement protocol (§5), algorithms (§6), datasets (§7) and reference perplexities (§13) are unchanged and still rest on `docs/verified-facts-2026-09-20.md`. Everything about hardware, environment, model tiering and milestones is rewritten.

## 0. What changed and why

The laptop was replaced. The new machine is a Linux CUDA box, which removes every Windows and OneDrive constraint the v1 plan was shaped around and promotes the laptop from "small-model CPU staging" to the primary compute for the whole study.

| | v1 assumed | Measured 2026-09-21 |
|---|---|---|
| OS | Windows 11 Home + OneDrive sync | Pop!_OS 24.04 LTS, kernel 7.1.5, plain `~/Documents/research` |
| CPU | Ryzen 7 5825U, 8c/16t | Intel i9-14900HX, 24c/32t, AVX2 (no AVX-512) |
| RAM | 11.3 GB | 31 GB total, 23 GB available, + 20 GB swap (16 GB zram) |
| GPU | none usable (Vega 8 iGPU off every support list) | **RTX 4070 Laptop, 8188 MiB, sm_89 (Ada), PCIe 4.0 x8** |
| Driver / CUDA | n/a | 595.84 / CUDA 13.2 |
| dtype | fp32 only, forever | **fp16 (OPT, Llama-2), bf16 (Llama-3)** |
| Disk | not tracked | 516 GB free on `/` |
| glibc | n/a | 2.39 (manylinux_2_28 wheels OK) |
| Triton / torch.compile | impossible (no Windows build) | available (torch 2.14 Linux wheels bundle triton 3.8) |

Consequences, in rough order of how much they change the work:

1. **Laptop rows are now paper-comparable.** v1 §5.8 withheld `paper_comparable=true` from every laptop row because fp32-on-CPU is not the papers' fp16-on-GPU setting. Every laptop row is now fp16-on-GPU, so the laptop produces directly comparable numbers and the v1 "measure fp32-vs-fp16 drift at M5" cross-check disappears — M1 becomes an exact gate instead of an advisory one.
2. **The binding constraint moved from host RAM to VRAM.** 8188 MiB, not 11.3 GB of system RAM, now sets the model ceiling — but 31 GB of host RAM plus block streaming lifts the ceiling past 7B rather than lowering it.
3. **Everything through 7B runs locally.** opt-125m through opt-2.7b fit resident in fp16; opt-6.7b, Llama-2-7b and Llama-3.1-8B run through block-streamed eval and layer-offloaded GPTQ. `--stream-blocks` is promoted from a v1 §12 risk mitigation to a core, load-bearing feature built at M3.
4. **Runtimes collapse.** The v1 opt-125m full matrix was 8-15 h of CPU, run overnight or sharded across two evenings. On the 4070 the same matrix is minutes. The cost centre is now GPTQ on the 7B models, not anything small.
5. **The server is downstream, not upstream.** Access is conditional on demonstrating a working framework on this hardware, so the plan is laptop-complete: a full, paper-validated result set ships from this machine alone (M0-M6), and the server (M7) is scale-out that needs no code changes.
6. **Deleted as moot:** the OneDrive relocation decision (Option A/B, `C:\dev\ptq-bench`, `Move-Item`, the README-pointer folder), the PowerShell bootstrap, `scripts/env.ps1`, `scripts/bootstrap.ps1`, `scripts/ssh/pull_results.ps1`, `git config core.longpaths`, `HF_HUB_DISABLE_SYMLINKS_WARNING`, the torch-directml and ROCm-on-Windows analysis, the `PTQ_ENABLE_GPTQMODEL_WINDOWS` escape hatch, the Windows CI leg, the lm-eval `DISABLE_MULTIPROC` Windows workaround, `datasets(num_proc=1)`, and the spawn-safety framing of the `__main__` guards (the guards stay; they are good hygiene and cost nothing).
7. **Added, because a GPU makes them matter:** TF32 must be disabled explicitly (§5a) or the ±0.02 fp16 gates are unreachable; chunked lm_head + cross-entropy (§5b) to stop Llama-3's 128k vocab from spending 2.6 GB of an 8 GB card on logits; window-batched block streaming (§5c) to decouple host RAM from window count.

## 1. Goal and the results matrix

Unchanged in substance. Build a PyTorch package, `ptqbench`, measuring how perplexity degrades as LLM weights are stored at fewer bits, and how much of that loss each quantization algorithm recovers. The matrix, with the model axis re-tiered for the new hardware:

| Axis | Values |
|---|---|
| model | **resident tier** (fp16 on GPU): opt-125m, opt-350m, SmolLM2-135M, opt-1.3b, opt-2.7b — **streamed tier**: opt-6.7b, Llama-2-7b, Llama-3.1-8B |
| algorithm | fp16 baseline, rtn, gptq, awq_lite, hqq; v2 library backends |
| bits | 16, 8, 4, 3, 2 |
| group_size | -1 (per-row), 128, 64 (2-bit only) |
| dataset | wikitext2, c4, ptb; c4_new and ptb_new as opt-in keys |

opt-2.7b is new: it fits resident at 5.3 GB fp16, the GPTQ paper reports it, and it fills the gap between 1.3b and 6.7b that v1 had to leave empty. There is no longer a "laptop models" / "server models" split — there is a resident tier and a streamed tier, both on this machine, differing only in `eval_mode`.

Each cell is one JSON row holding token-level perplexity, full provenance and a `paper_comparable` flag. Accuracy comes from fake quantization: weights are quantized to integers and immediately dequantized inside the original `nn.Linear`, as the GPTQ, AWQ and OmniQuant papers measured perplexity. No custom kernels are needed.

## 2. Hardware reality

One machine for M0-M6:

| | This laptop |
|---|---|
| Machine | Pop!_OS 24.04 LTS, i9-14900HX (24c/32t), 31 GB RAM, RTX 4070 Laptop 8188 MiB (sm_89), PCIe 4.0 x8 |
| Driver / toolkit | 595.84, CUDA 13.2 runtime. No CUDA **toolkit** installed (`nvcc` absent) — only needed for GPTQModel's JIT, see §6 |
| torch build | `cu130`. Works on this driver (13.2 ≥ 13.0), matches the plan's server target so one lock serves both, and CUDA 13.0 is likelier than 13.2 in a cluster `module load` list — which is what GPTQModel's nvcc-must-match rule cares about. `cu126` stays declared for an older-driver server; `cu132` is available and unused |
| dtype | fp16 for OPT and Llama-2, bf16 for Llama-3.1 (sm_89 has native bf16). fp32 never, except CPU-only unit tests |
| VRAM budget | 8188 MiB total; ~400 MB CUDA context; **~7.6 GB usable** |
| Host RAM budget | 23 GB available; the streamed tier's ceiling |
| Disk | 516 GB free. Projected use ~155 GB: ~52 GB weights, ~2 GB datasets, up to ~100 GB quant cache |

Graphics mode: `system76-power graphics` reports `nvidia` and `cosmic-comp` holds `/dev/nvidia0`, but the COSMIC Wayland compositor renders through Intel (`Mesa Intel RPL-S`) and **2 MiB** of VRAM is in use, so the dGPU is effectively dedicated to compute already. Switching to `hybrid` would guarantee that permanently but costs a reboot and, on most laptops of this class, the HDMI/DP outputs wired to the dGPU. Not required; revisit only if an eval OOMs within a few hundred MB of the limit. For long runs, `system76-power profile performance` (currently `Balanced`) is worth setting.

### 2a. What fits, and how

fp16 weights, plus ~400 MB CUDA context, plus activations. The logits term assumes the chunked cross-entropy of §5b; without it, add the bracketed figure.

| Model | fp16 weights | Peak VRAM, resident | Verdict |
|---|---|---|---|
| SmolLM2-135M | 0.27 GB | ~0.9 GB | resident |
| opt-125m | 0.25 GB | ~0.9 GB | resident |
| opt-350m | 0.66 GB | ~1.3 GB | resident |
| opt-1.3b | 2.45 GB | **2.56 GB measured** | resident |
| opt-2.7b | 4.93 GB | **5.07 GB measured** | resident |
| opt-6.7b | 13.3 GB | 14+ GB [+1.0 GB logits] | **streamed** |
| Llama-2-7b | 13.5 GB | 14+ GB [+0.7 GB] | **streamed** |
| Llama-3.1-8B | 16.1 GB | 17+ GB [+2.6 GB] | **streamed** |

The v1 auto-select rule is kept — stream when the VRAM budget is below `1.3 × model_bytes` — with two corrections found while implementing it:

**The budget must not be instantaneous free VRAM.** `torch.cuda.mem_get_info()[0]` under-reports, because torch's caching allocator keeps freed blocks reserved. Measured directly: the same `should_stream(5.3 GB)` call returned `False` on a clean context and `True` after an earlier eval had run, purely from cached blocks. In a matrix run that would let model N's leftovers decide model N+1's eval mode. `device.py` therefore offers two bases — `available` (free + reclaimable cache, the live load decision) and `capacity` (total − a 0.8 GB reserve for context and activations, a property of the hardware, used for planning and `--dry-run`).

**opt-2.7b is resident — resolved by measurement at M4.** The 5.3 GB figure above was rounded; `config.json` gives 2.65B parameters = **4.93 GB** fp16, so the rule needs 1.3 × 4.93 = 6.41 GB against a 6.82 GB capacity and lands resident. Measured 2026-09-21: resident fp16 evaluation peaks at **5.07 GB**, ~2.5 GB to spare, 0.296 s/window; opt-1.3b peaks at 2.56 GB at 0.158 s/window. The first streamed model is opt-6.7b. `tests/test_env.py::test_opt_2_7b_is_resident_by_measurement` pins the resolution.

The unambiguous cases are unaffected: opt-1.3b at 2.63 GB is resident, opt-6.7b at 13.3 GB and Llama-2-7b at 13.5 GB are streamed.

Streamed tier, host-RAM cost with §5c window batching at 32 windows (opt-6.7b measured 2026-09-22: **13.1 GB host, 1.95 GB VRAM**, against the 14.5 GB / ~2.5 GB projected below):

| Model | weights (host) | hidden cache | total | of 23 GB |
|---|---|---|---|---|
| opt-6.7b | 13.3 GB | 1.1 GB | ~14.5 GB | comfortable |
| Llama-2-7b | 13.5 GB | 1.1 GB | ~14.7 GB | comfortable |
| Llama-3.1-8B | 16.1 GB | 1.1 GB | ~17.3 GB | fits |

Per-block VRAM while streaming is one decoder block (≤ 0.5 GB for 8B) plus the window batch's activations — roughly 1.5-2.5 GB, far inside the card. GPTQ is the tighter case because it must hold a Hessian and its Cholesky inverse: the largest is OPT-6.7b's `fc1/fc2` at `in_features=16384`, so `H` is 16384² fp32 = 1.07 GB and `H + H⁻¹` is 2.1 GB; Llama's `down_proj` at 11008 is 0.97 GB for the pair. Both fit. GPTQ calibration, unlike eval, needs all 128 calibration windows' block outputs resident to feed the next block, so it cannot use window batching and costs 2 × 128 × 2048 × hidden × 2 B ≈ 4.3 GB of host RAM: 17.6 GB total for opt-6.7b, 20.4 GB for Llama-3.1-8B. The 8B case is the tightest thing in this plan; if it thrashes, the hidden-state cache moves to a `numpy.memmap` under `PTQ_CACHE_DIR` and trades ~2 GB of RAM for disk I/O.

### 2b. Time

The v1 CPU estimates are void. Measured `seconds_per_window` per (model, eval_mode) goes into `results/timing.json` at M1 and M5, and `--dry-run` sizes every later run from it. Expectations to be replaced by measurement:

| Work | v1 (CPU) | Expected here |
|---|---|---|
| opt-125m, one wikitext2 eval (~140 windows) | ~10 min | seconds |
| opt-125m full matrix | 8-15 h | minutes |
| resident-tier full matrix (125m → 2.7b) | not attempted | ~1-2 h |
| GPTQ, one config, opt-125m / SmolLM2 / opt-350m | ~minutes on CPU | **measured: 19–21 s / 30–44 s / 59–96 s** resident; opt-125m 36 s offloaded |
| GPTQ, one config, 7B, streamed | server-only | ~1-2 h (PCIe 4.0 x8 transfer is not the bottleneck; the Cholesky/GEMM work is) |
| 7B model, 6 paper-comparable GPTQ configs | server-only | ~6-12 h, one overnight run |

The quant cache (§9) therefore stops being a convenience and becomes the thing that makes the streamed tier affordable: a crash between datasets must never repeat a 1-2 h GPTQ pass.

## 3. Environment setup

Python 3.12 (`requires-python = ">=3.12,<3.14"`, `.python-version = 3.12`); the system interpreter is already 3.12.3 but uv manages its own. uv 0.12.17 is installed at `~/.local/bin/uv`. One `uv.lock`; extras, not platform markers, select the torch index, so the same lock serves this CUDA laptop, a CPU CI runner and a future server.

Paths. There is no OneDrive and no `$SCRATCH`, so the v1 relocation machinery is gone; caches live under `~/.cache` and `~/ml` simply to keep multi-GB artifacts out of the repo and off any future backup sweep.

| Variable | Value | Scope |
|---|---|---|
| `UV_CACHE_DIR` | `~/.cache/uv` | `~/.bashrc` |
| `HF_HOME` | `~/ml/hf` | `~/.bashrc` |
| `PTQ_CACHE_DIR` | `~/ml/ptq-cache` | `~/.bashrc` |
| `TORCH_EXTENSIONS_DIR` | `~/.cache/torch-ext` | `~/.bashrc` |
| `HF_TOKEN` | read-scope token | written by `uv run hf auth login` into `$HF_HOME/token` after `HF_HOME` is set |
| `DISABLE_CUDA` | `1` | shell-scoped in `scripts/env.sh` and CI |
| `UV_NO_SYNC` | `1` | shell-scoped in `scripts/env.sh` |
| `OMP_NUM_THREADS` | `24` (physical cores) | `scripts/env.sh` |

`DISABLE_CUDA=1` is the one line that reads wrong on a CUDA machine and must not be removed: it is **hqq's `setup.py` flag**, not a torch flag. It stops the hqq sdist shelling out to a CUDA extension build (which would fail anyway — no `nvcc`) on every lock and sync. It has no effect on `torch.cuda.is_available()`, and `ptq env-check` asserts CUDA is live precisely so this can never silently regress.

`UV_PROJECT_ENVIRONMENT` is never set: an absolute path shared across projects lets every other uv project's exact `uv sync` uninstall this one's packages. `UV_NO_SYNC=1` means `uv run` never syncs, so the environment changes only through an explicit `uv sync --extra ...`; without it a `uv run` issued without the same extras re-resolves torch from PyPI, which on Linux is the CUDA 13.0 build.

Bootstrap:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # done: uv 0.12.17
mkdir -p ~/ml/hf ~/ml/ptq-cache ~/.cache/uv ~/.cache/torch-ext
#   append the ~/.bashrc rows above, then open a new shell
cd ~/Documents/research
uv python install 3.12
uv init --package --python 3.12 .   # then replace pyproject.toml with the one below,
                                    # rename src/ptq_bench -> src/ptqbench, add scripts/env.sh
git add -A && git commit -m "scaffold"
#   gh is not installed; either  sudo apt install gh  (or the GitHub apt repo), then:
gh auth login && gh repo create ptq-bench --private --source=. --remote=origin --push
#   or create the repo in the browser and:  git remote add origin <url> && git push -u origin main
source scripts/env.sh
uv sync --extra cu130 --extra hqq --extra dev
uv run hf auth login                # HF_HOME is set, so the token lands in ~/ml/hf/token
uv run ptq env-check && uv run pytest -m smoke
```

The repo stays at `~/Documents/research`; nothing syncs it and every script path is relative to the repo root, so the v1 relocation is pointless churn. The GitHub repo is still named `ptq-bench`.

`pyproject.toml`. Changes from v1: `cu130` is the default local extra rather than a server-only one, and the `server` extra (GPTQModel, torchao) is now a *local* M6 concern, still commented out until it is actually wanted.

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "ptq-bench"
version = "0.1.0"
requires-python = ">=3.12,<3.14"
dependencies = [
  "transformers==5.17.0", "datasets==5.0.1", "accelerate==1.15.0",
  "safetensors>=0.8", "huggingface-hub>=1.5,<2", "sentencepiece>=0.2.1", "protobuf",
  "pydantic>=2,<3", "pyyaml", "numpy", "pandas", "matplotlib", "tqdm", "psutil",
]
[project.optional-dependencies]
cpu   = ["torch==2.14.0"]      # CI and CPU-only unit tests
cu126 = ["torch==2.14.0"]      # older-driver server
cu130 = ["torch==2.14.0"]      # this laptop
hqq   = ["hqq==0.2.8.post1"]
bnb   = ["bitsandbytes==0.50.2"]
lmeval = ["lm_eval[hf]==0.4.13"]
# server = ["gptqmodel==7.5.0", "torchao==0.18.0"]   # M6; torchao needs no nvcc, gptqmodel does
dev = ["pytest", "pytest-timeout", "ruff", "setuptools>=77,<83", "wheel", "ninja"]
[project.scripts]
ptq = "ptqbench.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/ptqbench"]

[tool.pytest.ini_options]
markers = ["smoke: fast CPU smoke tests", "gpu: requires CUDA"]
testpaths = ["tests"]

[tool.uv]
conflicts = [[{ extra = "cpu" }, { extra = "cu126" }, { extra = "cu130" }]]
# no-build-isolation-package = ["gptqmodel"]   # uncommented at M6
[tool.uv.sources]
torch = [
  { index = "pytorch-cpu",   extra = "cpu" },
  { index = "pytorch-cu126", extra = "cu126" },
  { index = "pytorch-cu130", extra = "cu130" },
]
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
# pytorch-cu126 and pytorch-cu130 entries follow the same form
```

Pin rationale is unchanged from v1 §3: torch 2.14.0 is current stable; for torch 2.14.0 only the cpu, cu126, cu130 and cu132 indexes exist (cu128 does not); `pip install torch` from PyPI on Linux silently gives the CUDA 13.0 build, so the explicit index matters. llm-compressor or vLLM, if ever wanted, get a separate venv at torch 2.13.0 / transformers 5.14.x. `uv export --extra cu130 --format requirements-txt > requirements-cu130.txt` covers a cluster that forbids uv.

`torch.compile` is now *possible* (the Linux torch 2.14 wheels bundle triton 3.8) but stays **off by default**: it buys little on a block-streamed loop, and recompilation noise is not worth risking the ±0.02 gates. It is an opt-in flag, recorded in the row when used.

### 3a. v2 backends and the `nvcc` question

`nvcc` is absent and the CUDA toolkit is not installed. This gates exactly one library:

- **bitsandbytes 0.50.2** — prebuilt wheels, no nvcc. Available now.
- **torchao 0.18.0** — `py3-none-any`, rides torch's own ops and triton. No nvcc. Available now; its torch-2.14 pairing is still unconfirmed and is verified by an import check at M6.
- **GPTQModel 7.5.0** — sdist-only, JIT-builds CUDA extensions, needs an `nvcc` whose major.minor equals `torch.version.cuda` (13.0). Requires installing the CUDA 13.0 toolkit (`sudo apt install cuda-toolkit-13-0` from NVIDIA's repo, ~3-4 GB, or the `nvidia-cuda-nvcc-cu13` wheel as a lighter attempt).

So M6's v2 tier starts with bitsandbytes and torchao, which cost nothing. GPTQModel is only needed for `ptq verify-backend` — the cross-check that our fake-quant integers, repacked into a real kernel format, reproduce the same perplexity within 0.05. That check is valuable but not on the critical path, so the toolkit install is deferred until M6 and skipped entirely if it fights.

## 4. Repository layout

```
research/                      # git repo; GitHub remote is named ptq-bench
  pyproject.toml  uv.lock  .python-version  .gitattributes  .gitignore
  README.md  .env.example  PLAN.md
  docs/verified-facts-2026-09-20.md  docs/PLAN-v1-windows-cpu-2026-09-20.md
  src/ptqbench/
    cli.py              # env-check | prefetch | eval | run | aggregate | plot | verify-backend
    paths.py            # sole owner of filesystem locations, env-driven pathlib.Path
    device.py           # resolve("auto") -> cuda:0 | cpu; dtype policy; TF32 lockdown (§5a)
    config.py           # pydantic RunSpec and QuantSpec, YAML extends, matrix expansion, hashes
    provenance.py       # git sha, package versions, platform, GPU name, driver, HF commit, §5a flags
    registry.py         # Registry[T]: lazy import, requires/devices, skipped rows with reason
    models/loader.py    # from_pretrained(dtype=..., attn_implementation="sdpa"), use_cache=False
    models/families.py  # OPT and Llama: blocks, linear names, float-kept modules, AWQ scale groups
    data/               # wikitext2, c4, ptb, pile_val, local_text, calibration; tokenized caches
    quantizers/         # fakequant, fp, rtn, gptq, awq_lite, hqq_adapter, sequential (v1)
                        # bnb_adapter, torchao_adapter, gptqmodel_adapter (v2, optional)
    eval/perplexity.py  # §5 arithmetic; chunked CE (§5b); resident or streamed (§5c)
    eval/streaming.py   # the block-streaming primitive shared by gptq and --stream-blocks
    eval/lmeval.py      # optional zero-shot tasks through HFLM
    runner/             # matrix expansion, quantize-once loop, quant cache, atomic JSON, CSV export
    analysis/           # summarize, plots
  references/literature.yaml   # §13 rows with source, table id, calib, protocol_uncertain
  configs/base.yaml  configs/models/*.yaml
  configs/experiments/{smoke_opt125m,resident_full,streamed_opt67b,streamed_llama}.yaml
  scripts/{bootstrap.sh,env.sh}
  scripts/slurm/{prefetch,run_matrix}.sbatch            # M7, unused until the server exists
  tests/{test_fakequant,test_data,test_config,test_results,test_pipeline_smoke,
         test_parity,test_streaming,test_ce_chunking}.py
  tests/reference/gptq_quant.py   # vendored find_params from IST-DASLab/gptq quant.py
  results/runs/<run_id>.json      # committed, small
  results/timing.json             # seconds_per_window per (hostname, model, eval_mode)
  results/results.csv  results/plots/
  .github/workflows/smoke.yml     # ubuntu-latest only, --extra cpu, DISABLE_CUDA=1
```

Entry points keep their `if __name__ == "__main__"` guards and files are still written with `encoding="utf-8"` — cheap hygiene that also keeps a future Windows or macOS contributor working. The v1 `num_proc=1` rule for `datasets` is dropped; with 24 cores, parallel tokenization is worth having.

`eval/streaming.py` is new and deliberately factored out: v1 had the sequential-GPTQ loop and the `--stream-blocks` eval loop as separate ideas, but they are the same primitive — iterate decoder blocks, move one to the GPU, push a set of hidden states through it, write the outputs back to host RAM. Building it once at M3 means the streamed tier is exercised by every GPTQ run long before a 7B model is attempted.

## 5. Evaluation protocol

Unchanged from v1: the IST-DASLab/gptq protocol shared by GPTQ, AWQ, SmoothQuant and OmniQuant. `protocol_version = 1` is stored in every row and bumped on any change.

1. Load with an explicit `dtype` (transformers 5 defaults to `dtype="auto"`), `attn_implementation="sdpa"` with an `eager` fallback flag, `use_cache=False`. Tokenizer via `AutoTokenizer.from_pretrained(repo)` with no `use_fast` argument: transformers 5.17.0 ships `GPT2Tokenizer` and `LlamaTokenizer` as single `TokenizersBackend` classes, so there is no slow path to choose; the row records `tokenizer_class` and the `tokenizers` version. `tests/test_data.py` asserts `type(tok).__name__` matches `configs/models/*.yaml` (`GPT2Tokenizer` for OPT and SmolLM2, `LlamaTokenizer` for Llama-2).
2. One token stream per dataset key as in §7, tokenized once with default special-token handling.
3. `seqlen = 2048`, `nsamples = numel // seqlen`, window `i = ids[:, i*2048:(i+1)*2048]`, tail dropped, batch 1, `torch.no_grad()`.
4. Per window: `nll_i = (sum of per-token NLL over the 2047 predictions) / 2047 * 2048`, accumulated in fp32. This keeps the reference quirk — a mean over 2047 predictions multiplied by 2048 — while being computable chunk-wise (§5b).
5. `ppl = exp(sum(nll_i) / (nsamples * 2048))`; store `nll_sum`, `n_tokens`, `n_windows` so re-aggregation is exact.
6. Checksums in `tests/test_data.py`: about 140 wikitext2 windows with the OPT tokenizer, about 166 with Llama-2, exactly 256 for c4, plus a hash of the first 32 token ids per tokenizer and dataset.
7. `max_windows` or `seqlen < 2048` marks the row `partial=true`; aggregation excludes it.
8. `paper_comparable=true` for fp16-on-GPU rows (bf16 for Llama-3) whose QuantSpec equals the paper config of §6, and `false` for `algo=awq_lite` unless `calib.dataset == "pile_val"`, `calib.nsamples == 128` and `calib.seqlen == 512`. **This now admits every row this machine produces** — the v1 carve-out excluding fp32 laptop rows is deleted, along with the v1 plan to measure fp32-vs-fp16 drift, which no longer exists to measure.

Row schema, with three fields added for §5a and §5c: `run_id, quant_key, protocol_version, model, model_revision, tokenizer_class, algo, backend, bits, group_size, sym, act_order, true_sequential, percdamp, calib{dataset,nsamples,seqlen,seed}, dataset, split, join, dtype, device, device_name, driver_version, eval_mode, ce_chunk, stream_window_batch, tf32, deterministic, ppl, nll_sum, n_tokens, n_windows, partial, paper_comparable, quant_seconds, eval_seconds, peak_ram_gb, peak_vram_gb, git_sha, versions, hostname, slurm_job_id, started_at, finished_at, status, reason`.

lm-eval stays secondary and zero-shot-only; its `wikitext` task is per-document and word-level, not comparable. `bootstrap_iters=0` is kept (it is a speed choice, not only a Windows workaround); `DISABLE_MULTIPROC` is no longer needed.

### 5a. Numerical determinism on the GPU — mandatory

This did not exist in a CPU-only plan and is the single easiest way to silently miss the gates. TF32 matmuls on Ampere and later carry ~10 bits of mantissa and move perplexity in the third decimal — larger than the ±0.02 fp16 tolerance the §13 gates are stated to. Before any model runs, `device.py` sets:

- float32 matmul precision to **highest** (`torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False`; torch 2.14 also exposes the newer `fp32_precision` API — set whichever the installed version provides, and record which).
- `torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False`, so fp16 GEMM accumulation stays in fp32.
- cuDNN benchmark off.

`tf32` and `deterministic` go into every row, and `ptq env-check` prints them. `torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8` is available behind `--deterministic` but is not the default, because it errors on some ops and costs speed; the M1 gate is the real test of whether the default is tight enough. `tests/test_parity.py` re-runs one eval twice and asserts identical `nll_sum`.

### 5b. Chunked lm_head and cross-entropy

Materialising a full `(1, 2048, V)` logits tensor and then calling `.float()` on it is affordable on a 24 GB card and wasteful on 8 GB. Cost per window at fp16 plus the fp32 copy plus log-softmax workspace:

| Model | V | full-logits peak | chunked at 256 |
|---|---|---|---|
| Llama-2-7b | 32,000 | ~0.7 GB | ~0.09 GB |
| opt-* | 50,272 | ~1.0 GB | ~0.13 GB |
| Llama-3.1-8B | 128,256 | ~2.6 GB | ~0.33 GB |

So `eval/perplexity.py` runs `lm_head` over the final hidden states in chunks of `--ce-chunk` (default 256) positions, accumulating `sum of per-token NLL` in fp32 with `reduction="sum"`, and only then applies the `/2047 * 2048` of §5.4. This is the same quantity as the unchunked form up to floating-point associativity, and never materialises more than `chunk × V` logits. `tests/test_ce_chunking.py` asserts opt-125m wikitext2 ppl agrees between `--ce-chunk 2048` (i.e. unchunked) and 256 to within 1e-4. `ce_chunk` is recorded in the row.

### 5c. Window-batched block streaming

v1's `--stream-blocks` held a hidden-state cache of `n_windows × 2048 × hidden` in host RAM, which for Llama-3.1-8B on wikitext2 is ~2.8 GB per buffer and ~5.6 GB for the in/out pair — on top of 16.1 GB of weights, uncomfortably close to the 23 GB available.

Batching the windows removes the coupling entirely. Process `--stream-window-batch` windows (default 32) at a time: for each batch, walk the decoder blocks, moving one block to the GPU and pushing the batch's hidden states through it. Host RAM for the cache becomes `batch × 2048 × hidden × 2 B × 2 buffers` ≈ 1.1 GB regardless of dataset size. The price is re-streaming the block weights once per batch: 140 wikitext2 windows at batch 32 is 5 passes, and at PCIe 4.0 x8 (~12 GB/s effective with pinned host memory) a full 16 GB pass costs ~1.4 s, so ~7 s of transfer for an entire dataset eval. Negligible against the compute.

The block-major ordering matters: the naive window-major alternative (stream all blocks per window) would re-transfer the weights 140 times — ~1.9 TB — and is never used. Pinned host buffers and a second CUDA stream overlap the next block's H2D copy with the current block's compute.

GPTQ calibration is the exception and cannot batch windows: the Hessian `H = 2/n Σ xxᵀ` is additive and could be accumulated in a streaming fashion, but the *next* block needs every calibration window's output, so all 128 stay resident. That is the 4.3 GB figure in §2a, and the `numpy.memmap` fallback applies there if Llama-3.1-8B proves too tight.

## 6. Quantization algorithms

Unchanged from v1 except where the GPU changes the device and dtype arguments.

QuantSpec defaults: `sym=false` (asymmetric min-max with zero-point), `act_order=false`, `true_sequential=false`, `percdamp=0.01`, `blocksize=128`, `mse=false`, `static_groups=false`. The grid reproduces the reference `Quantizer.find_params`, which clamps the range to include zero and maps all-zero rows to [-1, 1]:

```python
z = torch.zeros(W.shape[0], device=W.device, dtype=W.dtype)
xmin = torch.minimum(W.min(dim=1).values, z)
xmax = torch.maximum(W.max(dim=1).values, z)
dead = (xmin == 0) & (xmax == 0); xmin[dead] = -1; xmax[dead] = 1
scale = ((xmax - xmin) / (2**bits - 1)).unsqueeze(1); zero = torch.round(-xmin.unsqueeze(1) / scale)
q = torch.clamp(torch.round(W / scale) + zero, 0, 2**bits - 1); W_hat = scale * (q - zero)
```

The grid runs in **fp32** even when the model is fp16: `W.float()` in, `W_hat.to(W.dtype)` out. On CPU that was automatic; on a fp16 GPU model it must be explicit, or `xmax - xmin` and the `round` both lose precision and the reference-parity test fails. `tests/test_fakequant.py::test_matches_reference_quantizer` compares scale, zero and `W_hat` against the vendored `tests/reference/gptq_quant.py` on CPU, and a `@pytest.mark.gpu` variant asserts the CUDA path agrees with the CPU path bit-for-bit after the fp32 round-trip.

Paper configs: GPTQ OPT tables use `group_size=-1`; AWQ and OmniQuant Llama tables use g128, g64 at 2-bit. Quantized modules: OPT `q_proj,k_proj,v_proj,out_proj,fc1,fc2`; Llama `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`. Kept in float: embeddings, `lm_head`, all norms, biases, opt-350m's `project_in/project_out`.

v1 algorithms, pure PyTorch in-repo:

- `fp`: no-op, bits=16.
- `rtn`: the grid above on every target Linear. Data-free and now GPU-resident — seconds even for 7B.
- `gptq`: port of the ~300-line reference, using the same grid, built on `eval/streaming.py`. A Catcher replaces block 0 and stores `*args` and `**kwargs` verbatim, since transformers 5 Llama layers receive `position_embeddings`, `cache_position` and `attention_mask` from the model and replaying hidden states alone silently breaks Llama. Per block: hooks accumulate `H = 2/n Σ xxᵀ` over the 128×2048 calibration tokens in fp32, damping `percdamp * mean(diag H)`, Cholesky inverse, blocked OBS column updates, optional `act_order`, `static_groups`, `true_sequential`; dequantized weights written back; block re-run to feed the next. The model stays in host RAM and one block at a time moves to the GPU — for the resident tier this is a no-op detour, so the runner skips the offload when the whole model already fits.
  **Llama family needs `act_order` (measured at M3).** On SmolLM2-135M, 4-bit per-row: RTN 26.61, plain GPTQ **27.91**, GPTQ+act_order **24.18**, +true_sequential 23.85; at g64: RTN 19.95, plain GPTQ 20.25, GPTQ+act_order 19.50. Plain GPTQ is *worse than RTN* on this family in both settings while `true_sequential` alone changes nothing. It is not a port defect: GPTQ's per-layer output error beat RTN's in 28/28 SmolLM2 layers (ratios 0.12–0.68, block 29's `down_proj` RTN error alone is 4.8e8), so the loss is in how fixed-order local errors compose through 30 outlier-heavy layers — exactly what act-order (largest-Hessian-diagonal columns first) exists to fix, and what the IST-DASLab README recommends for LLaMA. Consequences: every Llama-family experiment config sets `act_order: true`; the OPT configs keep the paper's `act_order: false`; and M6 must verify whether AWQ Table 4's "GPTQ" column is GPTQ or GPTQ-R before the literature join pairs it with either. `group_size` must divide `in_features`: SmolLM2's hidden size 576 rejects g128 (a partial trailing group would silently corrupt a paper-comparable row), so its grouped setting is g64; every other model in §8 divides by both 128 and 64.
- `awq_lite`, after GPTQ parity: per scale group from `families.py`, `s = mean|X|^a / mean|W|^(1-a)` on a 20-point grid in [0,1] minimizing block-output MSE under RTN, folded into the preceding norm or Linear, then a clip search.
- `hqq_adapter`: `HQQLinear(linear, quant_config=BaseQuantizeConfig(nbits=bits, group_size=None if gs == -1 else gs, axis=1), compute_dtype=model_dtype, device=str(device), del_orig=False)`; `W_hat = hqq_layer.dequantize().to(linear.weight.dtype)`, copied back. hqq asserts `group_size % 8 == 0` and `in_features % group_size == 0`, and `None` is its per-row mode, so `-1` is translated. `HQQLinear` defaults to `compute_dtype=float16, device="cuda"`, which is now correct for this machine but is still passed explicitly so the CPU unit test works. `tests/test_pipeline_smoke.py` monkeypatches `torch.cuda.is_available` to False and runs hqq 4-bit on one opt-125m Linear on CPU. When hqq is absent the registry writes `status=skipped, reason=import_error` rows.

v2 library backends through the registry; each declares `requires` and `devices`, and unavailable ones become `status=skipped` rows. Re-tiered by whether they need a CUDA toolkit (§3a):

| Library | Version | Needs nvcc | Status here | Notes |
|---|---|---|---|---|
| hqq | 0.2.8.post1 | no | available | pure PyTorch; `DISABLE_CUDA=1` is a build-time flag only |
| bitsandbytes | 0.50.2 | no (prebuilt wheels) | available | 8-bit and NF4 only; `BitsAndBytesConfig`; sm_89 supported |
| torchao | 0.18.0 | no (`py3-none-any`, torch/triton ops) | available, torch-2.14 pairing unconfirmed | Int8/Int4WeightOnlyConfig; sm_89 supported by the tinygemm int4 path; GPTQ and AWQ there are prototypes |
| GPTQModel | 7.5.0 | **yes**, matching torch's CUDA 13.0 | deferred to M6 | the only backend behind a toolkit install; used solely for `ptq verify-backend` |
| llm-compressor | 0.13.0 | no, but pins torch<=2.13.0 | separate venv, optional | AWQ / SmoothQuant / AutoRound cross-check |
| AutoGPTQ, AutoAWQ, optimum-quanto | archived 2025-04-11, archived 2025-05-11, maintenance | — | do not use | AutoAWQ pins transformers 4.47.1 |

`ptq verify-backend` packs our GPTQ integers into gptqmodel's format and checks perplexity agrees within 0.05. It is now a local check rather than a server one, contingent on the toolkit.

## 7. Datasets

Unchanged from v1. No loading scripts (datasets>=4.0 removed them and `trust_remote_code`) and only namespaced ids (bare `wikitext` raises `HfUriError` on datasets 5.0.x).

| key | load call | split | join | notes |
|---|---|---|---|---|
| `wikitext2` | `load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")` | test, 4,358 rows | `"\n\n".join(rows["text"])` | primary column in all papers; the only hard gate |
| `c4` | `load_dataset("allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"}, split="validation")` | shard 0, 45,576 docs | none | `random.seed(0)`, 256 docs redrawn until >=2048 tokens, one random 2048 window each. No config name. GPTQ `get_c4`. **Not the paper column** — see below |
| **`c4_new`** | same shard | validation | `" ".join(docs[:1100])`, first 256×2048 tokens | GPTQ `--new-eval` variant. **This is the paper column** |
| `ptb` | `load_dataset("ptb-text-only/ptb_text_only", revision="refs/convert/parquet")` | validation, 3,370 rows, column `sentence` | `"\n\n".join` | GPTQ `get_ptb`. **Not the paper column** |
| **`ptb_new`** | same | test, 3,761 rows | `" ".join` | GPTQ `get_ptb_new`. **This is the paper column** |

**Resolved at M2 (2026-09-21).** The GPTQ README never states whether Tables 9 and 11 used `get_ptb`/`get_c4` or the `--new-eval` variants, so v1 carried both as `protocol_uncertain`. Running all four keys on opt-125m settles it: the `_new` variants reproduce the published numbers and the plain ones miss by 7-17%.

| key | fp16 | vs paper | RTN4 | vs paper |
|---|---|---|---|---|
| `ptb` | 32.5525 | −16.51% | 45.0970 | −16.32% |
| **`ptb_new`** | **38.9917** | **+0.00%** | **53.8840** | **−0.01%** |
| `c4` | 24.6055 | −7.36% | 31.6199 | −6.75% |
| **`c4_new`** | **26.5637** | **+0.01%** | **33.8850** | **−0.07%** |

Two independent columns agree for each dataset, so this is not a coincidence of one number. `ptb_new` and `c4_new` therefore become the default paper columns and carry `protocol_uncertain: false`; `ptb` and `c4` stay available as opt-in keys for anyone comparing against a study that used them. This reverses v1's default, which had the plain keys primary.

Measured window counts (OPT tokenizer, seqlen 2048), all confirmed against §5.6: wikitext2 140, `c4`/`c4_new` exactly 256, `ptb` 45, `ptb_new` 47. **`c4_new` is only "exactly 256" for tokenizers as verbose as OPT's**: it takes the first 1100 documents and keeps the first 256 × 2048 tokens, so a more efficient tokenizer yields fewer — Llama-3.1's 128k vocabulary gives 252. The random-draw `c4` key is always 256 by construction. Llama-3.1-8B (unsloth mirror, `TokenizersBackend`, vocab 128256, untied lm_head, GQA 8/32): wikitext2 141, c4_new 252, ptb_new 46. SmolLM2-135M gives 148 on wikitext2. Llama-2 tokenizer (NousResearch mirror, `LlamaTokenizer`, vocab 32000, one BOS at the stream start), measured 2026-09-22: wikitext2 **166** (§5.6 said about 166), c4_new 256, ptb_new 49; first-32-token hashes are pinned in `tests/test_data.py`.

PTB fallback: `load_dataset("parquet", data_files="hf://datasets/ptb-text-only/ptb_text_only@refs%2Fconvert%2Fparquet/penn_treebank/{split}/0000.parquet")`; the three parquet files (3.46 MB total) are copied into `PTQ_CACHE_DIR/ptb/` on first load. PTB is research-only licensed. C4: `verification_mode="no_checks"` is an opt-in fallback; alternatively `hf_hub_download` the shards and parse gzip JSONL directly, skipping the ~0.9 GB Arrow cache.

Calibration for every data-dependent method: 128 × 2048 windows from `data_files={"train": "en/c4-train.00000-of-01024.json.gz"}` (356,317 docs, 319 MB) using the reference loop under `random.seed(seed)`: `i = randint(0, len(train)-1)`, tokenize doc `i`, retry while `n_tok <= 2048`, then `s = random.randint(0, n_tok - 2048 - 1)`, slice `[s:s+2048]`. The `randint` bound is the reference's, so seed-0 windows are identical to IST-DASLab/gptq `get_c4`; the only deviation is the gate (the reference accepts a document of exactly 2048 tokens and then raises from `randint(0, -1)`, while this loop redraws it), and the protocol notes say so. Seed 0 by default; seeds 0, 1, 2 for the variance run. Windows cache to `PTQ_CACHE_DIR/calib/<dataset>-<tokenizer_hash>-n<N>-s<L>-seed<S>.pt`. `--calib wikitext2` is what the smoke config uses (6 MB train split, no C4 download). `--calib pile_val` is the AWQ paper's set. **Checked at M4: it does not load through `datasets` 5.0.1** — the repo is a single `val.jsonl.zst` and the JSON builder raises "Compression type zstd not supported" even with `zstandard` installed. The loader therefore fetches the file with `hf_hub_download`, decompresses it once into `PTQ_CACHE_DIR/datasets/pile_val.jsonl` (`zstandard` is now a dependency) and runs AWQ's own `get_calib_dataset` loop verbatim: shuffle with seed 42 + `spec.seed`, keep documents of at most `seqlen` tokens, concatenate, cut into `seqlen` blocks, take the first `nsamples`. Seed 0 is the paper's draw; 128 × 512 builds in 13 s and caches.

## 8. Models

Tiering is now by `eval_mode`, not by machine. Every model in this table runs on the laptop.

| # | Model | Weights | fp16 VRAM | Tier | Gate |
|---|---|---|---|---|---|
| 1 | facebook/opt-125m | 0.25 GB .bin, ungated | 0.25 GB | resident | §13 parity targets; the M1 and M2 hard gates |
| 2 | facebook/opt-350m | 0.66 GB, ungated | 0.66 GB | resident | topology test: post-LN and `project_in/out` |
| 3 | HuggingFaceTB/SmolLM2-135M | 0.27 GB safetensors, ungated | 0.27 GB | resident | Llama family map and Catcher must pass before any Llama run |
| 4 | facebook/opt-1.3b | 2.63 GB, ungated | 2.63 GB | resident | fp16 wikitext2 14.63 |
| 5 | facebook/opt-2.7b | 5.30 GB, ungated | 5.30 GB | resident, ~1.5 GB headroom | **new in v2**; fills the 1.3b→6.7b gap |
| 6 | facebook/opt-6.7b | 13.32 GB fp16 .bin, ungated | streamed, ~2.5 GB peak | streamed | fp16 wikitext2 10.86 ± 0.03 — the M5 scale gate |
| 7 | meta-llama/Llama-2-7b-hf | 13.5 GB, gated manual | streamed | streamed | fp16 5.47 ± 0.02; mirror NousResearch/Llama-2-7b-hf (ungated, SHA-equivalence unverified, fallback only) |
| 8 | meta-llama/Llama-3.1-8B | 16.06 GB bf16, gated manual | streamed, tightest host-RAM case | streamed | bf16; mirror unsloth/Meta-Llama-3.1-8B; no verified reference |

`configs/models/*.yaml` holds `repo`, `mirror`, `gated`, `tokenizer_class` and the dtype policy; the runner records which id was loaded. `models/loader.py` computes model bytes from `config.json` and picks resident vs streamed by the §2a rule, overridable with `--eval-mode`; the v1 "refuse above 70% of free RAM unless `--force`" guard is kept but now applies to *host* RAM for the streamed tier and to VRAM for the resident tier. Each Llama repo is gated separately and approval latency is real, so the license applications stay in M0 even though Llama is not touched until M6. OPT ships only `.bin` shards; transformers 5.17.0 loads them via `torch.load(weights_only=True)`, with a one-off `save_pretrained` safetensors conversion as fallback. Every row records repo id plus commit hash.

Projected weight storage: ~52 GB, against 516 GB free.

## 9. Matrix runner and results

`ptq run configs/experiments/X.yaml [--filter algo=gptq,bits=4] [--shard k/n] [--index N] [--list] [--force] [--rerun-incomplete] [--no-quant-cache] [--dry-run] [--eval-mode resident|streamed] [--ce-chunk N] [--stream-window-batch N]`

1. Expand the YAML product into `RunSpec`s. `quant_key = sha256(model, model_revision, QuantSpec, calib spec, dtype)[:12]`, `run_id = sha256(quant_key, dataset key, eval spec)[:12]`; hostname, device_name, git_sha and timestamps are excluded so ids are stable across reruns, while `dtype` is included. `eval_mode` is **not** in `run_id`: resident and streamed must produce the same number, and §10's M5 gate asserts exactly that on opt-1.3b, where both modes are possible.
2. Group by `quant_key`, sort deterministically, take shard `k` of `n` by index modulo `n`.
3. Per group: load once, build or load calibration, quantize once, evaluate every dataset. Each row is written to `results/runs/<run_id>.json.tmp` then moved with `os.replace`. Quantized int weights, scales and zeros are saved to `PTQ_CACHE_DIR/quant/<quant_key>.safetensors` (~3.5 GB for a 7B model at 4 bits, 1.5 GB at 2 bits); `sequential` loads it instead of re-quantizing, so a crash between datasets does not repeat a 1-2 h GPTQ pass. `--no-quant-cache` disables this for the variance study. A row is done when its file parses with the same `protocol_version`; unparsable, `failed` and `skipped` files are not-done under `--rerun-incomplete`.
4. Exceptions become `status=failed` rows with the traceback in `results/runs/<run_id>.log`; the runner continues. Ctrl-C finishes the current row. CUDA OOM is caught specifically and retried once at the next smaller `--stream-window-batch`, then once in streamed mode if it was resident, before being recorded as failed with the attempted budgets in `reason`.
5. `ptq aggregate` writes `results/results.csv` joined with `references/literature.yaml` on (model, algo, bits, group_size, sym, act_order, dataset, calib), adding `paper_ppl`, `paper_source`, `protocol_uncertain`, `delta_vs_paper`, `delta_vs_fp16`, excluding partial rows. `ptq plot`: per model three panels (wikitext2, c4, ptb), bits categorical on x, log-y perplexity, one line per algorithm, solid g128 and dashed per-row, fp16 horizontal line, paper numbers as hollow markers; plus a `delta_vs_fp16` heatmap and `results/summary.md`.

Quant-cache growth is the one new housekeeping job: ~10 configs × 3 streamed models at 1.5-3.5 GB each is ~100 GB. `ptq cache gc --keep-newest N` and a `--max-cache-gb` guard that evicts least-recently-used `quant_key` entries keep it bounded; at 516 GB free this is a convenience, not a constraint.

Experiment configs, renamed from v1's machine-based names to mode-based ones:

- `smoke_opt125m.yaml`: seqlen 512, max_windows 8, fp/rtn/gptq/hqq on wikitext2 with `calib: {dataset: wikitext2, nsamples: 16, seqlen: 512, seed: 0}` — uses the 6 MB wikitext2 train split, never downloads the 319 MB C4 shard, well under a minute on the GPU.
- `resident_full.yaml` (was `laptop_full.yaml`): models 1-5; fp; rtn, gptq and hqq at bits {8,4,3,2} × group_size {-1,128} plus 2-bit g64; awq_lite at 4 and 3 bits g128; datasets wikitext2, c4, ptb; C4 128×2048 seed-0 calibration. Expected ~1-2 h total, sized properly by `--dry-run` after M1.
- `streamed_opt67b.yaml` (was `server_opt67b.yaml`): model 6, same grid, `--stream-blocks`.
- `streamed_llama.yaml` (was `server_llama.yaml`): models 7-8 at the six §13-verified configurations first, the rest of the grid second.

CI is ubuntu-latest only, `--extra cpu`, `DISABLE_CUDA=1`, `HF_HOME` cached on the model and dataset list: `uv run pytest -m "smoke and not gpu"` plus `uv run ptq eval --model facebook/opt-125m --dataset wikitext2 --max-windows 4` from M2, and `ptq run configs/experiments/smoke_opt125m.yaml` from M4. GPU-marked tests run only locally, via `uv run pytest -m gpu`, and are part of each milestone's definition of done rather than CI.

## 10. Milestones

Restructured around one fact: **server access is earned by demonstrating a working framework on this hardware.** So M0-M6 all run on the laptop and together produce a complete, paper-validated result set; the server (M7) adds scale, not credibility, and needs no code changes to unlock. v1's M5 ("server + opt-6.7b") and M6 ("Llama + v2 backends") become local milestones; the server work moves to M7.

| M | Scope | Definition of done | Effort |
|---|---|---|---|
| **M0** Environment | uv (done), Python 3.12, `uv init`, pyproject with the cu130 extra, `~/.bashrc` vars and `scripts/env.sh`, `env-check`, git remote, minimal `ptq eval` (loader, wikitext2, perplexity, `--max-windows`), Llama license applications | `uv run ptq env-check` prints torch 2.14.0+cu130, `torch.cuda.is_available()` True, device `NVIDIA GeForce RTX 4070 Laptop GPU`, driver 595.84, sm_89, 8188 MiB, **TF32 off and fp16-reduction off** (§5a), `sys.prefix` inside the repo; `uv run python -c "from hqq.core.quantize import HQQLinear; print('hqq ok')"` prints ok (or the failure is recorded and `--extra hqq` dropped, making hqq rows `skipped`); `uv run ptq eval --model facebook/opt-125m --dataset wikitext2 --max-windows 4` gives a finite ppl and `partial=true`; origin remote exists and `git push` works; Llama-2-7b-hf and Llama-3.1-8B access requested at huggingface.co and `uv run hf auth whoami` works | half day |
| **M1** First number | full-length run of the M0 path, `fp` only, fp16 on GPU; `results/timing.json` | **opt-125m fp16 wikitext2 = 27.65 ± 0.05**, ~140 windows, and the row is `paper_comparable=true`. This is now an *exact* gate, not the v1 fp32 approximation — if it misses, §5a is the first suspect. Runtime recorded as `seconds_per_window` | half day |
| **M2** Data + RTN + schema + CI | c4, ptb, c4_new, ptb_new, `fakequant` with the reference grid and the explicit fp32 round-trip, `rtn`, JSON schema, token-hash and tokenizer-class tests, `test_matches_reference_quantizer` (CPU and GPU), chunked-CE equivalence test, smoke suite, CI | **done 2026-09-21.** RTN4 per-row **37.2831** vs 37.28 (+0.008%, gate ±1%); RTN8 **27.6595** vs fp **27.6559** (Δ 0.0036, gate ≤0.05); RTN3 1277 vs ~1.3e3. Chunked CE agrees to 5e-7. Dataset variant question resolved — see §7. *Still open, moved to M3:* the C4 calibration loop, which nothing needs until GPTQ | 1.5 days |
| **M3** GPTQ + the streaming primitive | `eval/streaming.py`, C4 calibration loop, `gptq` on the streamer with the kwargs-replaying Catcher, `--eval-mode streamed`, opt-350m and SmolLM2-135M runs, `test_streaming`, `test_gptq` | **done 2026-09-21.** opt-125m **GPTQ4 31.5475** vs 31.12 (+1.37%, gate ±3%), **GPTQ3 53.0273** vs 53.85 (−1.53%, gate ±5%); ptb_new GPTQ4 45.7609 vs 45.17 (+1.31%), c4_new 29.1953 vs 29.22 (−0.08%). opt-350m fp16 22.0017, RTN4 25.9412, GPTQ4 24.3851 (+0.60%), RTN3 64.5576, GPTQ3 32.7896 (−2.96%). Seeds 0/1/2: 31.55 / 31.87 / 31.24, stdev 0.32 — the seed-0 deviation from the paper is inside seed variance. **Resident and streamed eval agree to 0.0 exactly** on opt-125m, opt-350m and SmolLM2; two GPTQ runs, and resident vs offloaded GPTQ, agree on all 84.9M weights exactly. **SmolLM2 gate passed only with `act_order`** — see §6; the finding is pinned in `test_gptq.py` | 3 days |
| **M4** Runner + plots + AWQ-lite + HQQ + the resident matrix | registry, runner, shard/index/skip, atomic writes, quant cache, `--rerun-incomplete`, `--dry-run`, OOM-retry ladder, aggregate, plot, `awq_lite`, `hqq_adapter` with the CPU test, `pile_val` load check, `resident_full.yaml`, smoke config wired into CI, `ptq prefetch`, `ptq cache gc`, `status=skipped` rows for indivisible `group_size`, `ptq wizard` | **done 2026-09-22.** `resident_full.yaml`: **436 complete rows, 42 by-design skips (SmolLM2 g128), 0 failures**, 164 min + an 89 min `--rerun-incomplete` pass for opt-2.7b's eleven GPTQ/AWQ groups, which had OOMed twice over: the eval-only 1.3× rule ignored quantizer memory, and the retry ladder's caught exception kept the failed resident model alive. 61 cells joined to a published number: fp16 mean |Δ| 0.01% (max 0.04%), GPTQ mean 1.2% (max 5.6%, a 3-bit PTB cell), opt-2.7b GPTQ4 per-row 12.917 vs 12.87 and GPTQ3 16.865 vs 16.88. AWQ4 g128 below RTN4 g128 on every model. Resume verified in anger. Findings: GPTQ wins 4-bit g128 on opt-350m/1.3b/2.7b, AWQ-lite on opt-125m; **HQQ collapses at 3 bits** (96–299 on OPT, worse than RTN g128 — its defaults, not investigated); **two published RTN cells could not be reproduced** — see §13. Derived timings: GPTQ 17 s (125m) / 55 s (350m) / 172 s (1.3b) / 342 s (2.7b) per config | 2.5 days |
| **M5** Scale up: opt-6.7b streamed | prefetch and run `streamed_opt67b.yaml`; `peak_vram_gb` / `peak_ram_gb` instrumentation proven at scale | **fp16 gate passed 2026-09-22: opt-6.7b streamed wikitext2 = 10.8603 vs 10.86 (Δ +0.0003), peak VRAM 1.95 GB, peak host RAM 13.1 GB, 174 s for 140 windows, `paper_comparable=true`.** First attempt crashed on an uninitialised CUDA context — the pure-offload path had never run in a process where no tensor had touched the GPU — fixed in `lock_numerics` with a cold-process regression test. **Quantized gate passed 2026-09-22 (streamed, 33 min for all nine cells): RTN4 per-row 12.0992 vs 12.10, GPTQ4 per-row 11.4774 vs 11.39 (+0.087); PTB 15.7699 / 18.8578 / 16.4841 vs 15.77 / 18.84 / 16.56; C4 12.7121 / 14.3730 / 13.1715 vs 12.71 / 14.36 / 13.18. GPTQ on 6.7B took 857 s, not the 1–2 h projected in §2b; peak VRAM 1.95 GB, host RAM 13.6 GB.** Original targets: ptb 15.77 and c4 12.71 advisory; `peak_vram_gb` under 8.0 and `peak_ram_gb` under 23; **opt-1.3b evaluated both resident and streamed agrees to 1e-4**, proving the two paths are one protocol. This is the milestone that constitutes the evidence for server access: a 7B-class model fully quantized and matched to published numbers on an 8 GB laptop GPU | 2 days + one overnight |
| **M6** Llama + v2 backends | **fp16 gate passed 2026-09-22 on the NousResearch mirror: Llama-2-7b wikitext2 = 5.4721 vs 5.47, streamed, 172 s.** `streamed_llama.yaml` with `act_order: true`, verify whether the AWQ Table 4 / OmniQuant Table 1 GPTQ baselines are GPTQ or GPTQ-R and set `act_order` on those `literature.yaml` rows accordingly, `bnb_adapter` and `torchao_adapter` (neither needs nvcc), optional CUDA 13.0 toolkit → `gptqmodel_adapter` and `verify-backend`, optional lm-eval zero-shot rows, write-up | **Llama-2-7b fp16 5.47 ± 0.02, RTN4 g128 5.72-5.73, GPTQ4 g128 5.61-5.69**, and the other four §13-verified configurations within tolerance; Llama-3.1-8B bf16 completes at least fp/rtn/gptq at 4 bits g128 without OOM (no verified reference, so it is descriptive only); no empty pivot cell in the resident tier; if the toolkit is installed, fake-quant vs gptqmodel within 0.05 | 3-5 days, several overnight |
| **M7** Server scale-out *(unlocked by M5/M6)* | SSH/SLURM access, `scripts/env.sh` reused verbatim with the `cu126` or `cu130` extra as the driver dictates, `--shard k/n` across GPUs, larger models (opt-13b, opt-30b, Llama-2-13b), re-run the 7B matrix resident and faster | Server rows merge into the same `results/runs/` and dedupe by `run_id`; a 7B cell run resident on the server matches the laptop's streamed cell within 1e-4, which is the strongest possible statement that the protocol is machine-independent | 1-2 days setup, then GPU time |

The critical-path item with external latency is unchanged: Llama gating approval is requested at M0 and not needed until M6, which is ample.

## 11. Server hand-off (M7)

Nothing here is needed until M5 is demonstrated, and nothing here requires a code change — it is why extras rather than platform markers select the torch index.

Setup, when access arrives:

```bash
nvidia-smi; ldd --version | head -1        # driver picks cu126 vs cu130; glibc must be >= 2.28
git clone <origin url> ~/ptq-bench && cd ~/ptq-bench
#   scripts/env.sh reads $SCRATCH when set and falls back to ~/ml locally, so it is used unedited
source scripts/env.sh
uv python install 3.12
uv sync --extra cu130 --extra hqq --extra dev    # or --extra cu126 on an older driver
uv run ptq env-check && uv run pytest -m smoke
```

Code moves by git; results return as JSON files: `rsync -av --exclude '*.log' user@server:~/ptq-bench/results/runs/ results/runs/`, and the aggregator dedupes by `run_id`. (v1's `pull_results.ps1` / `scp` workaround existed only because Git Bash ships no rsync; both machines are Linux now.)

Plain SSH box: `tmux new -s ptq`, `source scripts/env.sh`, `uv run ptq prefetch configs/experiments/streamed_llama.yaml`, then one loop per GPU: `CUDA_VISIBLE_DEVICES=$i nohup uv run --frozen --no-sync ptq run $CFG --shard $i/$N > results/logs/worker$i.log 2>&1 &`.

SLURM sketch (`source scripts/env.sh` in the login shell before `sbatch` so `results/slurm` exists; `--gres=gpu:1` vs `--gpus=1` is cluster-specific):

```bash
#!/bin/bash
#SBATCH --job-name=ptq --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=12:00:00
#SBATCH --array=0-7 --output=results/slurm/%x_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env.sh
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
uv run --frozen --no-sync ptq run "$1" --shard "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
```

`--frozen --no-sync` matters: a plain `uv run` locks and syncs first, which can touch the network on an offline compute node. `ptq run --list` prints the group count to size the array; `prefetch.sbatch` runs first on a node with internet. Resume is re-submitting the same command. On the server the streamed tier becomes resident again (24 GB+ cards), which is exactly the cross-machine check in M7's definition of done. `module load cuda/X.Y` is needed in `env.sh` only if GPTQModel's JIT is used there, and the module's nvcc major.minor must equal `torch.version.cuda`.

## 12. Risks and mitigations

Rewritten: every Windows and OneDrive risk is deleted, and the new ones are all VRAM, numerics or scale.

| Risk | Mitigation |
|---|---|
| **TF32 silently shifts perplexity past the ±0.02 gates** | §5a: TF32 off for matmul and cuDNN, fp16 reduced-precision reduction off, flags recorded in every row and printed by `env-check`; M1 is the canary |
| **8 GB VRAM OOM mid-matrix** | §2a budget table; auto-select streamed at `free < 1.3 × model_bytes`; chunked CE (§5b) keeps a 128k-vocab logits tensor off the card; runner OOM-retry ladder (smaller window batch → streamed → failed with budgets in `reason`) |
| **Llama-3.1-8B GPTQ exceeds 23 GB host RAM** | window batching does not apply to calibration, so this is the tightest case at ~20.4 GB; `numpy.memmap` fallback for the hidden-state cache; 20 GB of swap as a backstop; the model is descriptive-only (no verified reference) so it never gates a milestone |
| Streamed tier crashes on a cold CUDA context | found by the M5 gate: memory-stats calls raise "Invalid device argument" before any tensor has touched the device, and `--eval-mode auto` masked it by querying VRAM first. `lock_numerics` now initialises the context on every entry path; `tests/test_streaming.py::test_streamed_eval_from_a_cold_process` runs the offload path in a fresh subprocess |
| **GPTQ without act_order is worse than RTN on the Llama family** | measured on SmolLM2 at M3 (§6); Llama configs default `act_order: true`, OPT configs keep the paper's `false`; `test_gptq.py` pins both the failure and the fix; M6 verifies which variant the AWQ/OmniQuant GPTQ baselines are |
| `group_size` does not divide `in_features` | `fakequant` refuses rather than tiling a partial group; the M4 runner writes `status=skipped, reason=group_size_indivisible`; only SmolLM2 (576) is affected, and it uses g64 |
| Streamed and resident eval disagree | they are one primitive (`eval/streaming.py`); M3 asserts agreement on opt-350m and M5 on opt-1.3b, both to 1e-4; `eval_mode` is excluded from `run_id` so the two cannot be silently treated as different cells |
| GPTQ on 7B costs 1-2 h per config, a crash wastes it | quant cache keyed on `quant_key`, written before any dataset is evaluated; `--rerun-incomplete`; atomic `os.replace` |
| Quant cache fills the disk | ~100 GB projected against 516 GB free; `ptq cache gc --keep-newest N` and `--max-cache-gb` LRU eviction |
| dGPU loses VRAM to the desktop | currently 2 MiB in use (compositor renders on Intel); `system76-power graphics hybrid` reclaims the rest if ever needed, at the cost of a reboot and the dGPU-wired display outputs |
| hqq sdist probes CUDA on every lock and sync | `DISABLE_CUDA=1` in `env.sh` and CI — a build-time hqq flag, not a torch one; `env-check` asserts CUDA is live so this cannot regress unnoticed |
| GPTQModel needs an nvcc matching torch's CUDA 13.0, and none is installed | deferred to M6; it serves only `verify-backend`; bitsandbytes and torchao cover the v2 tier with no toolkit; `cuda-toolkit-13-0` install documented as optional |
| torchao 0.18 with torch 2.14 unconfirmed | import-checked at install; Linux-only adapter; never gates a result |
| transformers 5 `dtype="auto"` picks the wrong dtype | loader always passes an explicit dtype; fp16 for OPT and Llama-2, bf16 for Llama-3.1 |
| Grid loses precision on an fp16 model | the reference grid runs in fp32 with an explicit `W.float()` / `.to(dtype)` round-trip; GPU-vs-CPU parity test |
| GPTQ Catcher misses Llama kwargs | verbatim `*args/**kwargs` replay; SmolLM2-135M gate at M3 before any 7B Llama |
| Llama gating latency | **Status 2026-09-22: no Hugging Face login on this machine** (`hf auth whoami` → not logged in; the M0 license step is still open and needs the user: `uv run hf auth login`, then accept the Llama-2 and Llama-3.1 licences on the Hub). Checked with a real download: `meta-llama/Llama-2-7b-hf` and `meta-llama/Llama-3.1-8B` are gated; the mirrors `NousResearch/Llama-2-7b-hf` and `unsloth/Meta-Llama-3.1-8B` download today. M6 therefore runs on the mirrors, with the mirror id and commit on every row, and re-runs on the official repos become a cheap upgrade once access exists — a repeated fp16 cell is the SHA-equivalence check the plan asked for |
| Crash mid-write; concurrent writers | tmp file plus `os.replace`; one JSON per `run_id` |
| PTB / C4 paper protocol unstated | `protocol_uncertain: true`; both variants run at M2; wikitext2 is the only hard gate |
| awq_lite rows mislabeled comparable | `paper_comparable=false` unless pile_val 128×512 calibration; `calib` participates in the literature join |
| C4 accidental full download | only two named shards; never `load_dataset("allenai/c4", "en")` |
| `datasets` cannot read pile_val's `.jsonl.zst` | measured at M4; manual `hf_hub_download` + `zstandard` decode, AWQ's loop reproduced verbatim (§7) |
| PTB parquet branch disappears | three files (3.46 MB) cached locally on first load |
| GPTQ run-to-run variance | 3 seeds on the resident tier; tolerances widened only after measuring spread |
| Server access never materialises | M0-M6 are laptop-complete and produce the full paper-comparable result set on their own; M7 is additive |

## 13. Reference perplexities

Unchanged from v1. seqlen 2048, fp16. GPTQ-paper tables use per-row asymmetric RTN and GPTQ with C4 128×2048 seed-0 calibration; AWQ Table 4 uses pile_val 128×512; OmniQuant Table 1 calibrates OmniQuant itself on 128×2048 WikiText2 segments and does not state how its RTN, GPTQ and AWQ baselines were obtained. Sources: GPTQ Tables 3, 9, 11; AWQ Table 4; OmniQuant Table 1.

| Model | Dataset | fp16 | RTN4 | GPTQ4 | RTN3 | GPTQ3 |
|---|---|---|---|---|---|---|
| OPT-125M | wikitext2 | 27.65 | 37.28 | 31.12 | 1.3e3 | 53.85 |
| OPT-125M | ptb | 38.99 | 53.89 | 45.17 | 1.4e3 | 73.19 |
| OPT-125M | c4 | 26.56 | 33.91 | 29.22 | 834 | 42.41 |
| OPT-350M | wikitext2 | 22.00 | 25.94 | 24.24 | 64.57 | 33.79 |
| OPT-1.3B | wikitext2 | 14.63 | 48.17 | 15.47 | 1.3e4 | 20.97 |
| OPT-6.7B | wikitext2 | 10.86 | 12.10 | 11.39 | 5.8e3 | 14.86 |
| OPT-6.7B | ptb | 15.77 | 18.84 | 16.56 | 5.7e3 | 21.88 |
| OPT-6.7B | c4 | 12.71 | 14.36 | 13.18 | 5.3e3 | 17.14 |

Llama-2-7B wikitext2, verified: fp16 5.47. W4 g128: RTN 5.73 (AWQ) or 5.72 (OmniQuant), GPTQ 5.69 or 5.61, AWQ 5.60 or 5.62. W4 per-row: RTN 6.11, GPTQ 5.83, AWQ 6.15. W3 g128: RTN 6.66, GPTQ 6.29, AWQ 6.24. W3 per-row: RTN 539.48, GPTQ 8.37. W2 g128: RTN 4.2e3, GPTQ 36.77. W2 g64: RTN 431.97, GPTQ 20.85.

`literature.yaml` rows carry `source`, `table`, `calib` and `protocol_uncertain`. The OPT ptb and c4 rows were `protocol_uncertain: true` in v1; M2 resolved them empirically to the `--new-eval` variants (see §7), so they now carry `protocol_uncertain: false` and `dataset_key: ptb_new` / `c4_new`. AWQ rows carry `calib: {dataset: pile_val, nsamples: 128, seqlen: 512}`; GPTQ-paper rows `calib: {dataset: c4, nsamples: 128, seqlen: 2048}`; OmniQuant Table 1 rows carry `protocol_uncertain: true`. The join therefore never pairs an awq_lite row with the wrong calibration.

**Verified from the GPTQ paper PDF on 2026-09-21** (arXiv 2210.17323, Tables 3, 9, 11 read directly): every row above matches, and the following are now in `literature.yaml` too — OPT-2.7B wikitext2 12.47 / RTN4 16.92 / GPTQ4 12.87 / RTN3 1.6e4 / GPTQ3 16.88, PTB 17.97 / 31.05 / 19.14 / 1.4e4 / 24.81, C4 14.34 / 18.43 / 15.00 / 1.1e4 / 18.17; OPT-1.3B PTB 20.29 / 57.30 / 21.85 / 1.3e4 / 32.10 and C4 16.07 / 24.51 / 16.97 / 5.2e3 / 21.63; OPT-350M PTB 31.08 / 36.79 / 34.52 / 88.04 / 47.08 and C4 22.59 / 26.21 / 24.63 / 55.49 / 31.33. Measured at M4 before the matrix: opt-1.3b fp16 wikitext2 **14.6239** (paper 14.63), opt-2.7b **12.4711** (paper 12.47). One wrinkle: the paper's Appendix A.2.1 describes concatenation "using two linebreaks as separators", yet the released code's `--new-eval` variants join with a single space and those are what reproduce Tables 9 and 11 (§7); the paper text and code disagree and the code wins empirically.

**Two published cells do not reproduce and are flagged `unreproduced` in `literature.yaml`:** OPT-1.3B RTN4 per-row on PTB (paper 57.30, measured **75.33**) and C4 (paper 24.51, measured **27.49**). The same quantized weights reproduce Table 3's wikitext2 cell to 0.03% (48.18 vs 48.17), the model's fp16 and GPTQ cells match on all three datasets to under 1%, and the measured values are stable to 0.05% across SDPA vs eager attention and fp16 vs fp32 — so this is not a pipeline, kernel or precision effect. OPT-2.7B shows the same pattern mildly (PTB +4.4%, C4 +2.2%, wikitext2 exact). The likeliest explanation is that those tables' RTN rows came from a separate run under slightly different conditions; the cells stay in the join, marked, and never gate anything.

Still unverified, never used as gates: Llama-3-8B fp16 about 6.1 on wikitext2 and 9.2 on c4. An fp16 match within 0.02 to 0.05 means the pipeline is correct; quantized rows within about 0.1 for 7B models, and on opt-125m RTN within 1% and GPTQ within 3 to 5%, are consistent with paper variance from the calibration seed and act-order choices. **The v1 caveat that fp32-vs-fp16 drift widens these tolerances no longer applies** — every row this machine produces is fp16 or bf16 on GPU, the papers' own setting.
