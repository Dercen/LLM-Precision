# PTQ Bench: cross-platform bit-precision study for LLMs

Final plan, 2026-09-20. Every version, dataset id and reference number below is taken from the facts verified today by web search against PyPI, GitHub, the Hugging Face Hub and the GPTQ, AWQ and OmniQuant papers; a few items are marked uncertain where verification was not possible.

Already done in `C:\Users\zache\OneDrive\Documents\research`: `git init` (branch `main`) and a `.gitignore` that excludes venvs, weights, caches and large outputs. Nothing is installed yet; the PyTorch install is the first step of M0.

## 1. Goal and the results matrix

Build a PyTorch package, `ptqbench`, that measures how perplexity degrades as LLM weights are stored at fewer bits, and how much of that loss each quantization algorithm recovers. "Compare datasets against algorithms" means filling and plotting this matrix:

| Axis | Values |
|---|---|
| model | opt-125m, opt-350m, SmolLM2-135M, opt-1.3b (laptop); opt-6.7b, Llama-2-7b, Llama-3.1-8B (server) |
| algorithm | fp16 baseline, rtn, gptq, awq_lite, hqq; v2 library backends |
| bits | 16, 8, 4, 3, 2 |
| group_size | -1 (per-row), 128, 64 (2-bit only) |
| dataset | wikitext2, c4, ptb; c4_new and ptb_new as opt-in keys |

Each cell is one JSON row holding token-level perplexity, full provenance and a `paper_comparable` flag. The aggregator joins rows onto `references/literature.yaml` (section 13) so plots show our number beside the published one. Accuracy comes from fake quantization: weights are quantized to integers and immediately dequantized inside the original `nn.Linear`, which is how the GPTQ, AWQ and OmniQuant papers measured perplexity. No kernels are needed, so identical code runs on the Windows CPU laptop and the Linux GPU server.

## 2. Hardware reality

| | Laptop | Server |
|---|---|---|
| Machine | Windows 11 Home, Ryzen 7 5825U, 11.3 GB RAM, no CUDA | Linux, NVIDIA GPUs, specs unknown |
| Compute | CPU only. The Vega 8 iGPU is on no AMD Windows or Linux PyTorch support list; torch-directml is frozen at torch 2.4.1 and DirectML is in maintenance mode | CUDA; glibc >= 2.28 required by the manylinux_2_28 torch 2.14 wheels |
| dtype | fp32 always. Zen 3 has no AVX512-BF16; bf16 on this CPU class measured ~80x slower than fp32 | fp16 for OPT and Llama-2, bf16 for Llama-3 |
| Model ceiling | opt-125m, opt-350m, SmolLM2-135M; opt-1.3b at 5.3 GB fp32 as an overnight option. Never opt-6.7b: 13.3 GB fp16 weights exceed RAM before the 9.96 GB .bin shard's transient load | opt-6.7b resident eval estimated at 15 to 16 GB VRAM (13.3 GB weights plus CUDA context and activations; arithmetic, not measured), marginal on a 16 GB card, 24 GB comfortable; `--stream-blocks` estimated at 4 to 5 GB; host RAM for layer-offloaded GPTQ on 7B is an unverified estimate (13.5 GB fp16 model plus about 4.3 GB of cached hidden states plus overhead, covered by the sbatch `--mem=64G`); all three confirmed from `peak_vram_gb` and `peak_ram_gb` at M5 |
| Time | opt-125m full matrix: measured at M1 from `seconds_per_window`; expected 8 to 15 h, run overnight or shard with `--shard` over two evenings | 8 to 12 GPU-hours per 7B model |

OneDrive caveat: `C:\Users\zache\OneDrive\Documents\research` syncs to the cloud, so a venv, HF cache or weights placed there would upload tens of GB and risk locked or dehydrated files. Decision needed before M0. Option A (recommended): the repo lives at `C:\dev\ptq-bench` and its venv is `C:\dev\ptq-bench\.venv` on both OSes; the OneDrive folder keeps only a README pointing there, and the git repo already initialised here (with `.gitignore`) is moved with `Move-Item`. Option B: keep the source here (it is small text) but relocate the venv by setting `UV_PROJECT_ENVIRONMENT=C:\ml\venvs\ptq-bench` inside `scripts/env.ps1` only, and change the `env-check` rule so only `sys.prefix`, `HF_HOME`, `UV_CACHE_DIR` and `PTQ_CACHE_DIR` are forbidden from containing `OneDrive`. Option B still risks OneDrive locking `.git` objects mid-commit, which is why A is preferred. The rest of this document assumes A. These variables (User-level on Windows, `~/.bashrc` on Linux, unless marked shell-scoped) keep everything else out:

| Variable | Windows | Linux |
|---|---|---|
| `UV_CACHE_DIR` | `C:\ml\uv-cache` | `$SCRATCH/uv-cache` |
| `HF_HOME` | `C:\ml\hf` | `$SCRATCH/hf` |
| `PTQ_CACHE_DIR` | `C:\ml\ptq-cache` | `$SCRATCH/ptq-cache` |
| `TORCH_EXTENSIONS_DIR` | unset | `$SCRATCH/torch-ext` |
| `XDG_CACHE_HOME` | unset | `$SCRATCH/cache` (gptqmodel JIT cache) |
| `HF_HUB_DISABLE_SYMLINKS_WARNING` | `1` | unset |
| `HF_TOKEN` | not set as a variable; token file `C:\ml\hf\token` is written by `uv run hf auth login` run in a shell where `HF_HOME` is already set | `export HF_TOKEN=hf_...` (read scope) in `~/.bashrc` before any prefetch |
| `DISABLE_CUDA` | `1`, shell-scoped in `scripts/env.ps1` | `1`, shell-scoped in `scripts/env.sh` and the CI workflow env |
| `UV_NO_SYNC` | `1`, shell-scoped in `scripts/env.ps1` | `1`, shell-scoped in `scripts/env.sh` |
| `OMP_NUM_THREADS` | `8`, physical cores | `$SLURM_CPUS_PER_TASK` |

`UV_PROJECT_ENVIRONMENT` is never set User-level: an absolute path shared across projects makes every other uv project's exact `uv sync` uninstall this one's packages. If a relocated venv is ever wanted it is set only inside `scripts/env.ps1` (shell-scoped). `UV_NO_SYNC=1` means `uv run` never syncs; the environment changes only through an explicit `uv sync --extra ...`. Without it, a `uv run` issued without the same `--extra` set re-resolves torch from PyPI, which on Linux is the CUDA 13 build. `ptq env-check` resolves every path plus `sys.prefix`, prints them, and exits non-zero if `sys.prefix` is not under the repo root or any path contains `OneDrive`.

## 3. Environment setup

Python 3.12: `requires-python = ">=3.12,<3.14"`, `.python-version = 3.12`. 3.10 reaches EOL in October 2026; 3.14: flash-attn has no cp314 wheels and fewer prebuilt kernels exist; 3.12 is the common HPC `module load` default and the only version AMD's ROCm-on-Windows build accepts. The universal lock also covers the laptop's installed 3.13. uv 0.12.17 manages interpreters and one `uv.lock` for both OSes.

Windows laptop (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
git config --global core.longpaths true
New-Item -ItemType Directory -Force C:\dev,C:\ml\hf,C:\ml\ptq-cache,C:\ml\uv-cache
[Environment]::SetEnvironmentVariable("HF_HOME","C:\ml\hf","User")   # and the other User-level rows of section 2; open a new shell
uv python install 3.12
Move-Item C:\Users\zache\OneDrive\Documents\research C:\dev\ptq-bench   # carries the existing .git (branch main) and .gitignore
New-Item -ItemType Directory C:\Users\zache\OneDrive\Documents\research
Set-Content C:\Users\zache\OneDrive\Documents\research\README.md "Repo lives at C:\dev\ptq-bench"
cd C:\dev\ptq-bench; uv init --package --python 3.12 .
#   replace the generated pyproject.toml with the one below, rename src/ptq_bench to src/ptqbench, add scripts/env.ps1, then:
git add -A; git commit -m "scaffold"
gh auth login; gh repo create ptq-bench --private --source=. --remote=origin --push   # gh: winget install GitHub.cli; --push refuses a repo with no commits
#   or create the repo on GitHub, then: git remote add origin <url>; git push -u origin main
. .\scripts\env.ps1                                   # DISABLE_CUDA=1 for this shell
uv sync --extra cpu --extra hqq --extra dev           # venv is C:\dev\ptq-bench\.venv
uv run hf auth login                                  # HF_HOME is set, so the token lands in C:\ml\hf\token
uv run ptq env-check; uv run pytest -m smoke
```

Linux server (bash):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
nvidia-smi; ldd --version | head -1                   # driver decides cu126 vs cu130; glibc must be >= 2.28
mkdir -p $SCRATCH/hf $SCRATCH/ptq-cache $SCRATCH/uv-cache $SCRATCH/torch-ext $SCRATCH/cache
git clone <origin url> ~/ptq-bench && cd ~/ptq-bench
source scripts/env.sh                                 # section 2 exports, DISABLE_CUDA=1, UV_NO_SYNC=1, PATH, module load cuda/12.6 or 13.0, mkdir results/slurm results/logs
uv python install 3.12
uv sync --extra cu126 --extra hqq --extra dev         # cu130 only with an R580 or newer driver
uv run ptq env-check; uv run pytest -m smoke
```

Server kernels, at M6, after the base sync has passed `env-check` and the smoke tests:

```bash
module load cuda/12.6      # or cuda/13.0 with the cu130 extra; nvcc major.minor must equal `python -c 'import torch;print(torch.version.cuda)'`
nvcc --version
# edit pyproject.toml: uncomment  server = ["gptqmodel==7.5.0", "torchao==0.18.0"]  and  no-build-isolation-package = ["gptqmodel"]
uv sync --extra cu126 --extra hqq --extra dev                  # step 1: torch, setuptools, wheel and ninja are now in .venv
uv sync --extra cu126 --extra hqq --extra dev --extra server   # step 2: gptqmodel builds inside .venv (uv's documented two-step sync)
uv run python -c "import gptqmodel, torchao, numpy; print(gptqmodel.__version__, torchao.__version__, numpy.__version__)"
git add pyproject.toml uv.lock; git commit -m "server extra"; git push
```

The `server` extra and its `[tool.uv]` stanza are commented out in the M0 pyproject so the laptop never resolves gptqmodel; they are enabled and locked on the server at M6, after which the laptop runs `uv sync --frozen --extra cpu --extra hqq --extra dev` and never selects `server`. `no-build-isolation-package` builds gptqmodel inside `.venv`, which is why `setuptools`, `wheel` and `ninja` sit in the `dev` extra; `[tool.uv.extra-build-dependencies]` only affects isolated builds and is not used. Fallback if the lock fails on the server: `uv pip install "setuptools>=77,<83" wheel ninja` then `uv pip install --no-build-isolation gptqmodel==7.5.0`, followed by `uv sync --inexact ...` for every later sync, since a plain `uv sync` removes it.

Plain pip equivalents: `pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu` on the laptop, `.../whl/cu126` on the server. For torch 2.14.0 only cpu, cu126, cu130 and cu132 indexes exist; cu128 does not. `pip install torch` from PyPI on Linux silently gives the CUDA 13 build, so the explicit index matters. Extras rather than platform markers select the index, so a Linux CPU box or CI can also `uv sync --extra cpu`.

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
cpu = ["torch==2.14.0"]
cu126 = ["torch==2.14.0"]
cu130 = ["torch==2.14.0"]
hqq = ["hqq==0.2.8.post1"]
# server = ["gptqmodel==7.5.0", "torchao==0.18.0"]   # Linux CUDA only; uncommented and locked on the server at M6
bnb = ["bitsandbytes==0.50.2"]
lmeval = ["lm_eval[hf]==0.4.13"]
dev = ["pytest", "pytest-timeout", "ruff", "setuptools>=77,<83", "wheel", "ninja"]   # build deps for gptqmodel inside .venv at M6
[project.scripts]
ptq = "ptqbench.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/ptqbench"]

[tool.pytest.ini_options]
markers = ["smoke: fast CPU smoke tests"]
testpaths = ["tests"]

[tool.uv]
conflicts = [[{ extra = "cpu" }, { extra = "cu126" }, { extra = "cu130" }]]
# no-build-isolation-package = ["gptqmodel"]   # uncommented on the server at M6
[tool.uv.sources]
torch = [
  { index = "pytorch-cpu", extra = "cpu" },
  { index = "pytorch-cu126", extra = "cu126" },
  { index = "pytorch-cu130", extra = "cu130" },
]
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
# pytorch-cu126 and pytorch-cu130 entries follow the same form
```

Pin rationale: torch 2.14.0 is current stable and gptqmodel 7.5.0 needs only torch>=2.8. If llm-compressor or vLLM are ever wanted they get a separate venv at torch 2.13.0 and transformers 5.14.x, because llm-compressor 0.13.0 caps torch<=2.13.0 and transformers<=5.14.1. hqq is an optional extra because its sdist runs setup.py on every lock and sync and shells out to a CUDA build unless `DISABLE_CUDA=1`, which the env scripts and CI set permanently; if the hqq sdist fails to build on Windows, run without `--extra hqq` and hqq rows are skipped. gptqmodel is sdist-only, imports torch and setuptools at build time and pins numpy==2.2.6 plus torchao>=0.16, so it enters the lock only at M6 on the server and the laptop never installs it; its `server` extra pulls torchao, whose 0.18.0 pairing with torch 2.14 is unconfirmed and is verified by the import line above. `uv export --extra cu126 --format requirements-txt > requirements-cu126.txt` covers clusters that forbid uv. `torch.compile` stays off everywhere since Triton has no official Windows build.

## 4. Repository layout

```
ptq-bench/
  pyproject.toml  uv.lock  .python-version  .gitattributes (* text=auto eol=lf)
  .gitignore            # .venv/, *.log, caches, *.safetensors, *.bin
  README.md  .env.example
  src/ptqbench/
    cli.py              # env-check | prefetch | eval | run | aggregate | plot | verify-backend
    paths.py            # sole owner of filesystem locations, env-driven pathlib.Path
    device.py           # resolve("auto") -> cpu or cuda:N; dtype policy per model family
    config.py           # pydantic RunSpec and QuantSpec, YAML extends, matrix expansion, hashes
    provenance.py       # git sha, package versions, platform, GPU name, HF commit hash
    registry.py         # Registry[T]: lazy import, requires/devices/platforms, skipped rows with reason
    models/loader.py    # from_pretrained(dtype=..., attn_implementation="sdpa"), use_cache=False
    models/families.py  # OPT and Llama: blocks, linear names, float-kept modules, AWQ scale groups
    data/               # wikitext2, c4, ptb, pile_val, local_text, calibration; tokenized caches
    quantizers/         # fakequant, fp, rtn, gptq, awq_lite, hqq_adapter, sequential (v1)
                        # bnb_adapter, gptqmodel_adapter, torchao_adapter (v2, optional)
    eval/perplexity.py  # section 5 arithmetic, resident or --stream-blocks
    eval/lmeval.py      # optional zero-shot tasks through HFLM
    runner/             # matrix expansion, quantize-once loop, quant cache, atomic JSON per run_id, CSV export
    analysis/           # summarize, plots
  references/literature.yaml   # section 13 rows with source, table id, calib, protocol_uncertain
  configs/base.yaml  configs/models/*.yaml   # repo, mirror, gated, tokenizer_class, dtype
  configs/experiments/{smoke_opt125m,laptop_full,server_opt67b,server_llama}.yaml
  scripts/{bootstrap.ps1,bootstrap.sh,env.ps1,env.sh}
  scripts/slurm/{prefetch,run_matrix}.sbatch  scripts/ssh/{worker_loop.sh,pull_results.ps1}
  tests/{test_fakequant,test_data,test_config,test_results,test_pipeline_smoke,test_parity}.py
  tests/reference/gptq_quant.py   # vendored find_params from IST-DASLab/gptq quant.py
  results/runs/<run_id>.json   # committed, small
  results/timing.json          # seconds_per_window per hostname, measured at M1
  results/results.csv  results/plots/   # regenerated
  .github/workflows/smoke.yml  # windows-latest + ubuntu-latest, DISABLE_CUDA=1, HF_HOME cached
```

Every entry point sits under `if __name__ == "__main__"` because Windows multiprocessing uses spawn; `datasets` calls use `num_proc=1` on Windows; files are written with `encoding="utf-8"`.

## 5. Evaluation protocol

This is the IST-DASLab/gptq protocol shared by GPTQ, AWQ, SmoothQuant and OmniQuant. `protocol_version = 1` is stored in every row and bumped on any change.

1. Load with an explicit `dtype` (transformers 5 defaults to `dtype="auto"`, which would load OPT in fp16 on CPU), `attn_implementation="sdpa"` with an `eager` fallback flag, `use_cache=False`. Tokenizer via `AutoTokenizer.from_pretrained(repo)` with no `use_fast` argument: transformers 5.17.0 pops and ignores `use_fast` and ships `GPT2Tokenizer` and `LlamaTokenizer` as single `TokenizersBackend` classes, so there is no slow path to choose; the row records `tokenizer_class` and the `tokenizers` package version inside `versions`. `tests/test_data.py` asserts `type(tok).__name__` equals the `tokenizer_class` in `configs/models/*.yaml` (`GPT2Tokenizer` for OPT and SmolLM2, whose tokenizer_config.json declares that class even though its architecture is Llama; `LlamaTokenizer` for Llama-2). The window counts and first-32-token-id hashes of step 6 are the only guard against drift from the papers' transformers-4.x slow tokenizers.
2. Build one token stream per dataset key as in section 7, tokenized once with default special-token handling, so a Llama tokenizer adds one BOS at the stream start only.
3. `seqlen = 2048`, `nsamples = numel // seqlen`, window `i = ids[:, i*2048:(i+1)*2048]`, tail dropped, batch 1, `torch.no_grad()`.
4. Per window: `loss = CrossEntropyLoss()(logits[:, :-1].float().reshape(-1, V), ids[:, 1:].reshape(-1))`, `nll_i = loss * 2048`. Keep the reference quirk of a mean over 2047 predictions multiplied by 2048.
5. `ppl = exp(sum(nll_i) / (nsamples * 2048))`; store `nll_sum`, `n_tokens`, `n_windows` so re-aggregation is exact.
6. Checksums asserted in `tests/test_data.py`: about 140 wikitext2 windows with the OPT tokenizer, about 166 with Llama-2, exactly 256 for c4, plus a hash of the first 32 token ids per tokenizer and dataset.
7. `max_windows` or `seqlen < 2048` marks the row `partial=true`; aggregation excludes it.
8. `paper_comparable=true` only for fp16-on-GPU rows whose QuantSpec equals the paper config of section 6, and `false` for `algo=awq_lite` unless `calib.dataset == "pile_val"`, `calib.nsamples == 128` and `calib.seqlen == 512`, the AWQ paper's calibration. fp32 laptop rows are not comparable even if fp32-vs-fp16 drift on opt-125m turns out to be under 0.02, as expected; the drift is measured by comparing the M1 laptop run with the M5 GPU parity run.

Row schema: `run_id, quant_key, protocol_version, model, model_revision, tokenizer_class, algo, backend, bits, group_size, sym, act_order, true_sequential, percdamp, calib{dataset,nsamples,seqlen,seed}, dataset, split, join, dtype, device, device_name, eval_mode, ppl, nll_sum, n_tokens, n_windows, partial, paper_comparable, quant_seconds, eval_seconds, peak_ram_gb, peak_vram_gb, git_sha, versions, hostname, slurm_job_id, started_at, finished_at, status, reason`.

lm-eval is secondary. Its `wikitext` task is per-document and word-level, not comparable to paper numbers, so it serves only zero-shot tasks: move the model to its device, `lm = HFLM(pretrained=model, tokenizer=tok, max_length=2048)`, then `simple_evaluate(model=lm, tasks=["piqa","arc_easy","arc_challenge","hellaswag","winogrande"], num_fewshot=0, bootstrap_iters=0)`, and on Windows set `DISABLE_MULTIPROC=1` because open issue #2581 reloads the model per spawned worker.

## 6. Quantization algorithms

QuantSpec defaults: `sym=false` (asymmetric min-max with zero-point), `act_order=false`, `true_sequential=false`, `percdamp=0.01`, `blocksize=128`, `mse=false`, `static_groups=false`. Grid, reproducing the reference `Quantizer.find_params`, which clamps the range to include zero and maps all-zero rows to [-1, 1]:

```python
z = torch.zeros(W.shape[0], device=W.device, dtype=W.dtype)
xmin = torch.minimum(W.min(dim=1).values, z)
xmax = torch.maximum(W.max(dim=1).values, z)
dead = (xmin == 0) & (xmax == 0); xmin[dead] = -1; xmax[dead] = 1
scale = ((xmax - xmin) / (2**bits - 1)).unsqueeze(1); zero = torch.round(-xmin.unsqueeze(1) / scale)
q = torch.clamp(torch.round(W / scale) + zero, 0, 2**bits - 1); W_hat = scale * (q - zero)
```

The same code runs per output row or per `group_size` slice of input columns. `tests/test_fakequant.py::test_matches_reference_quantizer` compares scale, zero and `W_hat` against the vendored `tests/reference/gptq_quant.py`. Paper configs: GPTQ OPT tables use `group_size=-1`; AWQ and OmniQuant Llama tables use g128, g64 at 2-bit. Quantized modules: OPT `q_proj,k_proj,v_proj,out_proj,fc1,fc2`; Llama `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`. Kept in float: embeddings, `lm_head`, all norms, biases, opt-350m's `project_in/project_out`.

v1, pure PyTorch in-repo, both machines:

- `fp`: no-op, bits=16.
- `rtn`: the grid above on every target Linear; seconds on CPU.
- `gptq`: port of the ~300-line reference, using the same grid. A Catcher replaces block 0 and stores `*args` and `**kwargs` verbatim, since transformers 5 Llama layers receive `position_embeddings`, `cache_position` and `attention_mask` from the model and replaying hidden states alone silently breaks Llama. Per block: hooks accumulate `H = 2/n sum x x^T` over the 128x2048 calibration tokens, damping `percdamp * mean(diag H)`, Cholesky inverse, blocked OBS column updates, optional `act_order`, `static_groups`, `true_sequential`; dequantized weights written back; block re-run to feed the next. On CUDA the model stays in host RAM and one block at a time moves to the GPU.
- `awq_lite` after GPTQ parity: per scale group from `families.py`, `s = mean|X|^a / mean|W|^(1-a)` on a 20-point grid in [0,1] minimizing block-output MSE under RTN, folded into the preceding norm or Linear, then a clip search.
- `hqq_adapter`: `HQQLinear(linear, quant_config=BaseQuantizeConfig(nbits=bits, group_size=None if gs == -1 else gs, axis=1), compute_dtype=torch.float32 if device.type == "cpu" else model_dtype, device=str(device), del_orig=False)`; `W_hat = hqq_layer.dequantize().to(linear.weight.dtype)`, copied back into `linear.weight`; data-free, 2 to 8 bits. hqq asserts `group_size % 8 == 0` and `in_features % group_size == 0`, and `None` is its per-row mode, so `-1` is translated. `HQQLinear` defaults to `compute_dtype=float16, device="cuda"`, which raises on the laptop or lands in the fp16 CPU slow path, so both are always passed. `tests/test_pipeline_smoke.py` monkeypatches `torch.cuda.is_available` to False and runs hqq 4-bit on one opt-125m Linear. When hqq is not installed the registry writes `status=skipped, reason=import_error` rows.

v2, library backends through the registry; each declares `requires`, `devices`, `platforms`, and unavailable ones become `status=skipped` rows.

| Library | Version | Windows CPU | Linux CUDA | Notes |
|---|---|---|---|---|
| bitsandbytes | 0.50.2 | yes, official win_amd64 wheel, AVX2 CPU backend | yes | 8-bit and NF4 only; `BitsAndBytesConfig`, `device_map="cpu"` |
| hqq | 0.2.8.post1 | likely, pure PyTorch, verified by the M0 import check | yes | `hqq` extra; `DISABLE_CUDA=1` from env scripts and CI |
| GPTQModel | 7.5.0 | claimed, unverified, Torch kernel only; behind `PTQ_ENABLE_GPTQMODEL_WINDOWS=1` | yes, primary real-kernel GPTQ and AWQ | `server` extra enabled at M6 with the two-step sync, or manual `--no-build-isolation` install; JIT needs the CUDA toolkit matching the torch build (12.6 or 13.0); transformers `GPTQConfig` backend (AutoGPTQ is archived) |
| torchao | 0.18.0 | no Windows binary wheel (the py3-none-any wheel installs but its quantization APIs are unsupported there), not used | Linux only; torch 2.14 pairing unconfirmed | Int8/Int4WeightOnlyConfig; GPTQ and AWQ there are prototypes |
| llm-compressor | 0.13.0 | no | yes, own venv at torch 2.13.0 and transformers 5.14.x | AWQ, SmoothQuant, AutoRound cross-check |
| AutoGPTQ, AutoAWQ, optimum-quanto | archived 2025-04-11, archived 2025-05-11, maintenance mode | do not use | do not use | AutoAWQ hard-requires triton and pins transformers 4.47.1 |

`ptq verify-backend` packs our GPTQ integers into gptqmodel's format on the server and checks perplexity agrees within 0.05.

## 7. Datasets

No loading scripts (datasets>=4.0 removed them and `trust_remote_code`) and only namespaced ids (bare `wikitext` raises `HfUriError` on datasets 5.0.x).

| key | load call | split | join | notes |
|---|---|---|---|---|
| `wikitext2` | `load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")` | test, 4,358 rows | `"\n\n".join(rows["text"])` | primary column in all papers; the only hard gate |
| `c4` | `load_dataset("allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"}, split="validation")` | shard 0, 45,576 docs | none | `random.seed(0)`, 256 docs redrawn until >=2048 tokens, one random 2048 window each. No config name: `"allenai--c4"` raises on current datasets. GPTQ `get_c4`; `protocol_uncertain: true` |
| `c4_new` | same shard | validation | `" ".join(docs[:1100])`, first 256x2048 tokens | GPTQ `--new-eval` variant, opt-in |
| `ptb` | `load_dataset("ptb-text-only/ptb_text_only", revision="refs/convert/parquet")` | validation, 3,370 rows, column `sentence` | `"\n\n".join` | GPTQ `get_ptb`; the README does not state whether Table 9 used this or `get_ptb_new`, so `protocol_uncertain: true` |
| `ptb_new` | same | test, 3,761 rows | `" ".join` | GPTQ `get_ptb_new`, opt-in |

PTB fallback: `load_dataset("parquet", data_files="hf://datasets/ptb-text-only/ptb_text_only@refs%2Fconvert%2Fparquet/penn_treebank/{split}/0000.parquet")`; the three parquet files (3.46 MB in total; test/0000.parquet is 262 kB) are copied into `PTQ_CACHE_DIR/ptb/` on first load since HF no longer regenerates that branch. PTB is research-only licensed. C4: `verification_mode="no_checks"` is an opt-in fallback for split-verification errors; alternatively `hf_hub_download` the two shards and parse gzip JSONL directly, skipping the ~0.9 GB Arrow cache.

Calibration for every data-dependent method: 128 x 2048 windows from `data_files={"train": "en/c4-train.00000-of-01024.json.gz"}` (356,317 docs, 319 MB) using the reference loop under `random.seed(seed)`: `i = randint(0, len(train)-1)`, tokenize doc `i`, retry while `n_tok <= 2048`, then `s = random.randint(0, n_tok - 2048 - 1)`, slice `[s:s+2048]`. The `randint` bound is the reference's, so seed-0 windows are identical to IST-DASLab/gptq `get_c4` whenever the reference completes; the only deviation is the gate: the reference accepts a document of exactly 2048 tokens (`>= seqlen`) and then raises from `randint(0, -1)`, while this loop redraws it, and the protocol notes say so. The 256 C4 eval windows of the `c4` row use the same gate and bound under `random.seed(0)`. Seed 0 by default; seeds 0, 1, 2 for the variance run on opt-125m and opt-1.3b. Windows cache to `PTQ_CACHE_DIR/calib/<dataset>-<tokenizer_hash>-n<N>-s<L>-seed<S>.pt`. `--calib wikitext2` exists for studies that calibrated on WikiText2 and is what the smoke config uses (6 MB train split, no C4 download). `--calib pile_val` loads `load_dataset("mit-han-lab/pile-val-backup", split="validation")` and draws 128 x 512 windows, the AWQ paper's set; confirm it loads script-free on datasets 5.0.1 at M4 before relying on it.

## 8. Models

| # | Model | Weights | Need | Where | Gate |
|---|---|---|---|---|---|
| 1 | facebook/opt-125m | 0.25 GB .bin, ungated | 0.5 GB fp32 | laptop | section 13 parity targets |
| 2 | facebook/opt-350m | 0.66 GB, ungated | 1.3 GB fp32 | laptop | topology test, post-LN and project_in/out; advisory wikitext2 parity fp16 22.00, RTN4 25.94, GPTQ4 24.24 |
| 3 | HuggingFaceTB/SmolLM2-135M | 0.27 GB safetensors, ungated | 0.5 GB fp32 | laptop | Llama family map and Catcher must pass before any server Llama run |
| 4 | facebook/opt-1.3b | 2.63 GB, ungated | 5.3 GB fp32 | laptop overnight, optional | fp16 14.63 on server |
| 5 | facebook/opt-6.7b | 13.32 GB fp16 .bin, ungated | estimated 15 to 16 GB VRAM resident, 24 GB comfortable, 4 to 5 GB with `--stream-blocks`; host RAM for GPTQ per the section 2 estimate | server only | fp16 wikitext2 10.86 +/- 0.03 before any matrix job |
| 6 | meta-llama/Llama-2-7b-hf | 13.5 GB, gated manual | 24 GB VRAM | server | fp16 5.47 +/- 0.02; mirror NousResearch/Llama-2-7b-hf, ungated, reported to be the same weights but SHA-equivalence to Meta's repo is unverified; fallback only, repo id and commit recorded on every row |
| 7 | meta-llama/Llama-3.1-8B | 16.06 GB bf16, gated manual | 24 GB VRAM | server | bf16; mirror unsloth/Meta-Llama-3.1-8B; no verified reference |

`configs/models/*.yaml` holds `repo`, `mirror`, `gated: true|false`, `tokenizer_class` and the dtype policy; the runner records which id was actually loaded. `models/loader.py` computes model bytes from `config.json` and refuses loads above 70% of free RAM unless `ptq eval|run --force` is passed. Each Llama repo is gated separately and approval latency is on the critical path, so the license steps are part of M0. OPT ships only `.bin` shards; transformers 5.17.0 still loads them via `torch.load(weights_only=True)`, with a one-off `save_pretrained` safetensors conversion as fallback. Every row records repo id plus commit hash.

## 9. Matrix runner and results

`ptq run configs/experiments/X.yaml [--filter algo=gptq,bits=4] [--shard k/n] [--index N] [--list] [--force] [--rerun-incomplete] [--no-quant-cache] [--dry-run]`

1. Expand the YAML product into `RunSpec`s. `quant_key = sha256(model, model_revision, QuantSpec, calib spec, dtype)[:12]`, `run_id = sha256(quant_key, dataset key, eval spec)[:12]`; hostname, device_name, git_sha and timestamps are excluded so ids are stable across reruns, while `dtype` is included, so a laptop fp32 row and a server fp16 row of the same cell have different ids and both survive the `run_id` dedupe of section 11. `--force` skips the 70% free-RAM guard of section 8. `--dry-run` prints `n_groups`, `n_rows` and `estimated_hours` from the `seconds_per_window` for this hostname in `results/timing.json`.
2. Group by `quant_key`, sort deterministically, take shard `k` of `n` by index modulo `n`.
3. Per group: load once, build or load calibration, quantize once, evaluate every dataset. Each row is written to `results/runs/<run_id>.json.tmp` then moved with `os.replace`, so a kill mid-write never leaves a truncated file that counts as done. Quantized int weights, scales and zeros are saved to `PTQ_CACHE_DIR/quant/<quant_key>.safetensors` (about 3.5 GB for a 7B model at 4 bits, 1.5 GB at 2 bits); `sequential` loads it instead of re-quantizing when present, so a crash between datasets does not repeat a 20 to 60 min GPTQ pass; `--no-quant-cache` disables this for the variance study. A row is done when its file parses with the same `protocol_version`; unparsable, `status=failed` and `status=skipped` files are treated as not-done under `--rerun-incomplete`, which is how skipped hqq or gptqmodel rows get filled once the backend is installed. One file per run, no shared writer, so SLURM arrays never collide.
4. Exceptions become `status=failed` rows with the traceback in `results/runs/<run_id>.log`; the runner continues. Ctrl-C finishes the current row.
5. `ptq aggregate` writes `results/results.csv` joined with `references/literature.yaml` on (model, algo, bits, group_size, sym, act_order, dataset, calib), adding `paper_ppl`, `paper_source`, `protocol_uncertain`, `delta_vs_paper`, `delta_vs_fp16`, excluding partial rows. `ptq plot`: per model three panels (wikitext2, c4, ptb), bits as categorical x, log-y perplexity, one line per algorithm, solid g128 and dashed per-row, fp16 horizontal line, paper numbers as hollow markers; plus a `delta_vs_fp16` heatmap and `results/summary.md`.

`smoke_opt125m.yaml`: seqlen 512, max_windows 8, fp/rtn/gptq/hqq on wikitext2 with `calib: {dataset: wikitext2, nsamples: 16, seqlen: 512, seed: 0}`, so it uses the 6 MB wikitext2 train split already fetched and never downloads the 319 MB C4 shard; under 2 minutes. At M2 CI runs `uv run pytest -m smoke` and `uv run ptq eval --model facebook/opt-125m --dataset wikitext2 --max-windows 4` on windows-latest and ubuntu-latest, with `DISABLE_CUDA=1` in the workflow env and `HF_HOME` cached by actions/cache keyed on the model and dataset list; `ptq run configs/experiments/smoke_opt125m.yaml` joins smoke.yml at M4, once `ptq run`, `gptq` and `hqq_adapter` exist.

`laptop_full.yaml`: fp; rtn, gptq and hqq at bits {8, 4, 3, 2} x group_size {-1, 128} plus 2-bit g64; awq_lite at 4 and 3 bits g128 only; datasets wikitext2, c4, ptb; calibration C4 128x2048 seed 0. Its runtime is whatever `--dry-run` prints from the M1 measurement, expected 8 to 15 h.

## 10. Milestones

| M | Scope | Definition of done | Where |
|---|---|---|---|
| M0 Environment | uv, Python 3.12, git init plus origin remote, pyproject with build system, env vars and env scripts, `env-check`, minimal `ptq eval` (loader, wikitext2, perplexity, `--max-windows`), Llama license steps | `env-check` shows torch 2.14.0+cpu, `sys.prefix` under `C:\dev\ptq-bench`, no OneDrive path; origin remote exists, `git push` from the laptop succeeds, the OneDrive research folder contains only README.md pointing to `C:\dev\ptq-bench`; `uv run python -c "from hqq.core.quantize import HQQLinear, HQQBackend; print('hqq ok')"` prints hqq ok, or the Windows hqq build failure is recorded in README and `--extra hqq` is dropped so hqq rows become `status=skipped`; `uv run ptq eval --model facebook/opt-125m --dataset wikitext2 --max-windows 4` (251 MB + 7 MB total) finishes with a `partial=true` row and a finite ppl; logged in at huggingface.co, opened meta-llama/Llama-2-7b-hf and meta-llama/Llama-3.1-8B, clicked Agree and access, completed Meta's form (status under Settings > Gated Repositories), `uv run hf auth whoami` works and `uv run python -c "from huggingface_hub import model_info; print(model_info('meta-llama/Llama-2-7b-hf').sha)"` prints a sha once approved | laptop, half day |
| M1 First number | full-length run of the M0 eval path, `fp` only; `results/timing.json` with `seconds_per_window`; no registry or runner yet | opt-125m fp32 wikitext2 27.65 +/- 0.1 (paper value is fp16 on GPU; fp32 drift is expected to be under 0.02 but is unmeasured until this step), ~140 windows, under 10 min | laptop, half day |
| M2 Data + RTN + schema | c4, ptb, c4_new, ptb_new, calibration loop, `fakequant` with the reference grid, `rtn`, JSON schema, token-hash and tokenizer-class tests, `test_matches_reference_quantizer`, smoke suite, CI | wikitext2 is the only hard gate: RTN4 per-row 37.28 +/- 1% (fp32 grid vs paper fp16), RTN8 within 0.05 of fp. Advisory: run both ptb and ptb_new (c4 and c4_new), record which one lands within 1% of 38.99 (26.56) and set that key as the paper column. CI green on both OSes | laptop, 2 days |
| M3 GPTQ | `sequential` with kwargs-replaying Catcher, `gptq`, SmolLM2-135M and opt-350m runs, `test_parity` | GPTQ4 per-row 31.12 +/- 3%, GPTQ3 per-row 53.85 +/- 5%; ptb 45.17 and c4 29.22 advisory on the key chosen at M2; opt-350m advisory fp16 22.00, RTN4 25.94, GPTQ4 24.24, RTN3 64.57, GPTQ3 33.79; SmolLM2 GPTQ4 finite and below RTN4; 3-seed spread recorded | laptop, 3 days |
| M4 Runner + plots + AWQ-lite + HQQ | registry, runner, shard/index/skip, atomic writes, quant cache, `--rerun-incomplete`, `--dry-run` estimate, aggregate, plot, `awq_lite`, `hqq_adapter` with the CPU test, `pile_val` load check, `laptop_full.yaml`, `smoke_opt125m.yaml` wired into CI, `ptq prefetch` plus `scripts/slurm/prefetch.sbatch`, `--stream-blocks` eval mode recording `eval_mode` and `peak_vram_gb` | `--dry-run` prints the estimate; the opt-125m matrix completes unattended in the estimated time (expected 8 to 15 h, overnight or two evenings); a run killed mid-way resumes without re-quantizing; CSV and plots regenerate in one command; AWQ4 g128 below RTN4 g128 | laptop, 3 days |
| M5 Server + OPT-6.7b | in order: `uv sync --extra cu126 --extra hqq --extra dev`, `uv run ptq env-check`, `uv run pytest -m smoke`, opt-125m fp16 GPU parity, `ptq prefetch`, opt-6.7b fp16 gate, then `server_opt67b.yaml` via sbatch or worker loop | opt-125m fp16 on GPU 27.65 +/- 0.05 and rows agree with laptop within 0.1 before any large download; opt-6.7b fp16 10.86 +/- 0.03, GPTQ4 per-row 11.39 +/- 0.15, RTN4 12.10 +/- 0.05; `eval_mode` and `peak_vram_gb` recorded; results merged back | server, 1 to 2 days |
| M6 Llama + v2 backends | `server` extra enabled and locked on the server (section 3), `server_llama.yaml` at g128, `verify-backend` vs gptqmodel, optional bitsandbytes and lm-eval rows, write-up | Llama-2-7b fp16 5.47 +/- 0.02, RTN4 g128 5.72 to 5.73, GPTQ4 g128 5.61 to 5.69; no empty pivot cell; fake-quant vs gptqmodel within 0.05 | server, 2 to 3 days GPU |

## 11. Server hand-off

Code moves by git, results return as JSON files. From Windows use `scp` since Git Bash ships no rsync: `scripts/ssh/pull_results.ps1` copies `results/runs/*.json` and the aggregator dedupes by `run_id`. From Linux: `rsync -av --exclude '*.log' user@server:~/ptq-bench/results/runs/ results/runs/`.

`scripts/env.sh` does `export PATH="$HOME/.local/bin:$PATH"` (uv is not on PATH in a batch shell), the section 2 exports, `export DISABLE_CUDA=1 UV_NO_SYNC=1`, `module load cuda/12.6` (or `cuda/13.0` when the `cu130` extra was chosen) and `mkdir -p results/slurm results/logs`. torch needs only the driver; gptqmodel/torchao kernel JIT needs the CUDA toolkit, so the module load is required, uncommented, in env.sh, and the module's nvcc major.minor must equal `torch.version.cuda`.

Plain SSH box: `tmux new -s ptq`, `source scripts/env.sh`, `uv run ptq prefetch configs/experiments/server_opt67b.yaml`, then `scripts/ssh/worker_loop.sh` starts one loop per GPU: `CUDA_VISIBLE_DEVICES=$i nohup uv run --frozen --no-sync ptq run $CFG --shard $i/$N > results/logs/worker$i.log 2>&1 &`.

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

`--frozen --no-sync` matters: a plain `uv run` locks and syncs the project first, which can touch the network on an offline compute node. `ptq run --list` prints the group count to size the array; `prefetch.sbatch` runs first on a node with internet. Resume is re-submitting the same command: finished `run_id` files are skipped, failed and skipped ones retried with `--rerun-incomplete`, and cached quantized weights are reused.

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| OneDrive syncs venv, caches or weights | repo and `.venv` at `C:\dev\ptq-bench`, section 2 env vars, `env-check` refuses OneDrive paths and a `sys.prefix` outside the repo |
| A User-level `UV_PROJECT_ENVIRONMENT` lets other uv projects uninstall this env | never set User-level; shell-scoped in `scripts/env.ps1` only if ever needed |
| transformers 5 `dtype="auto"` loads OPT fp16 on CPU | loader always passes explicit dtype; CPU forces fp32 |
| hqq sdist probes CUDA on every lock and sync | `DISABLE_CUDA=1` in env scripts and CI; `hqq` is an optional extra; if the sdist fails to build on Windows, run without `--extra hqq`, hqq rows are skipped |
| hqq adapter assumes CUDA and fp16 | explicit `device` and `compute_dtype`; CPU test with `torch.cuda.is_available` monkeypatched |
| gptqmodel or torchao break the lock, or an exact `uv sync` removes them | `server` extra enabled and locked on the server at M6 with the two-step sync; fallback manual install with setuptools preinstalled and `uv sync --inexact` afterwards; `UV_NO_SYNC=1` stops `uv run` from re-syncing; skipped rows if absent |
| gptqmodel JIT needs nvcc matching torch's CUDA | `module load cuda/12.6` (or `cuda/13.0` with the `cu130` extra) in env.sh; `nvcc --version` checked before install |
| torchao 0.18 with torch 2.14 unconfirmed, no Windows binary wheel | Linux-only adapter, verified by the import line at install, never gates a result |
| Server driver too old for cu126 or cu130, or glibc < 2.28 | `nvidia-smi` and `ldd --version` first; extras make the index a one-flag switch |
| GPU has < 24 GB | `ptq eval --stream-blocks` reuses the sequential block-streaming loop (one decoder block on GPU, hidden-state cache of nsamples x 2048 x hidden in host RAM, an estimated 4 to 5 GB VRAM for opt-6.7b) and is auto-selected when `torch.cuda.mem_get_info()[0] < 1.3 x model_bytes`; row records `eval_mode=resident|streamed` and `peak_vram_gb` |
| Llama gating latency, token missing on compute nodes | M0 license steps; `HF_TOKEN` in `~/.bashrc`; `hf auth login` only after `HF_HOME` is set; mirrors with `gated: false` in `configs/models`, repo plus commit recorded |
| GPTQ Catcher misses Llama kwargs | verbatim `*args/**kwargs` replay; SmolLM2 gate in M3 |
| Grid differs from the reference quantizer | zero-inclusive range and dead-row rule; `test_matches_reference_quantizer` |
| Crash mid-write or mid-matrix | tmp file plus `os.replace`; `--rerun-incomplete`; quant cache in `PTQ_CACHE_DIR/quant` |
| Concurrent writes from SLURM arrays | one JSON per `run_id` |
| Big downloads before anything works | smoke config calibrates on wikitext2; M0 `--max-windows 4` eval; M5 parity before `prefetch` |
| PTB parquet branch disappears | three files cached locally on first load |
| PTB and C4 paper protocol unstated | `protocol_uncertain: true`; both variants run at M2; wikitext2 is the only hard gate |
| awq_lite rows mislabeled comparable | `paper_comparable=false` unless pile_val 128x512 calibration; `calib` in the literature join |
| C4 accidental full download | only two named shards; never `load_dataset("allenai/c4", "en")` |
| Laptop RAM | 70% free-RAM guard; opt-6.7b server-only |
| lm-eval Windows multiprocessing bug | `DISABLE_MULTIPROC=1`, `bootstrap_iters=0`, `__main__` guard |
| GPTQ run-to-run variance | 3 seeds on small models; widen tolerances only after measuring spread |
| Tokenizer drift vs the papers' transformers-4.x slow tokenizers | transformers 5 has only `TokenizersBackend` tokenizers; `sentencepiece` and `protobuf` pinned for Llama conversions; class asserted in tests; window counts and first-32-token hashes checked against the reference at M2 |

## 13. Reference perplexities

seqlen 2048, fp16. GPTQ-paper tables use per-row asymmetric RTN and GPTQ with C4 128x2048 seed-0 calibration; AWQ Table 4 uses pile_val 128x512; OmniQuant Table 1 calibrates OmniQuant itself on 128x2048 WikiText2 segments and does not state how its RTN, GPTQ and AWQ baselines were obtained. Sources: GPTQ Tables 3, 9, 11; AWQ Table 4; OmniQuant Table 1.

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

`literature.yaml` rows carry `source`, `table`, `calib` and `protocol_uncertain`. The OPT ptb and c4 rows are `protocol_uncertain: true` with the note "GPTQ README does not state whether Tables 9/11 used get_ptb/get_c4 or the --new-eval variants". AWQ rows carry `calib: {dataset: pile_val, nsamples: 128, seqlen: 512}`; GPTQ-paper rows `calib: {dataset: c4, nsamples: 128, seqlen: 2048}`; OmniQuant Table 1 rows carry `protocol_uncertain: true` with the note "OmniQuant calibrates its own method on 128 x 2048 WikiText2 segments and does not state how its baselines were obtained" (RTN rows are data-free and unaffected). The join therefore never pairs an awq_lite row with the wrong calibration. The OPT-350M row is an M3 advisory gate only.

Uncertain, not in the verified facts and never used as gates: OPT-1.3B ptb 20.29 and c4 16.07; Llama-3-8B fp16 about 6.1 on wikitext2 and 9.2 on c4. An fp16 match within 0.02 to 0.05 means the pipeline is correct; quantized rows within about 0.1 for 7B models, and on opt-125m RTN within 1% and GPTQ within 3 to 5%, are consistent with paper variance from the fp32 grid, calibration seed and act-order choices.