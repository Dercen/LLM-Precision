# PTQ Bench — design reference

What the code does and why, for anyone reading results or extending the benchmark.
The measurement protocol, algorithm definitions, dataset recipes, hardware budget and
reference numbers live here; findings live in [RESULTS.md](RESULTS.md), the server
hand-off in [SERVER.md](SERVER.md). The original plans are archived alongside
(`PLAN-v1-windows-cpu-2026-09-20.md`, `PLAN-v2-laptop-2026-09-21.md`).

Section numbers are kept from the plan so commit messages and test docstrings that
cite "PLAN.md 5a" still resolve: 2 hardware, 5 protocol, 6 algorithms, 7 datasets,
8 models, 13 references.

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

