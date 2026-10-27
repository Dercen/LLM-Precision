# Exporting a quantized model to torch-mlir, step by step

`ptq export-mlir` takes one model, squeezes its weights to a chosen number of bits with one of
the project's quantization methods, and writes the result in a form other tools can read:

- `model.mlir`: the whole model as **MLIR** (a compiler format), with the rounded weights
  stored as integers and the "integer to float" conversion kept visible in the graph.
- `layers.json`: for every quantized layer, how much the rounding changed it.
- `export.json`: what was exported, with which settings, and a few checks.

The weights in the export are exactly the ones the perplexity results were measured with.

## Step 1: Open a terminal in the project folder

```bash
cd ~/LLM-Precision
source scripts/env.sh
```

The second line sets up cache folders. Run it once in every new terminal window.

## Step 2: Install the export add-on (first time only)

```bash
uv sync --extra cu130 --extra hqq --extra dev --extra mlir
```

This adds `torch-mlir` to the environment. Use `--extra cpu` instead of `--extra cu130` on
a machine without an NVIDIA GPU. Check it worked:

```bash
uv run python -c "import torch_mlir; print('torch-mlir ok')"
```

## Step 3: Run the export

The simplest useful command. It rounds opt-125m to 4 bits with plain rounding (RTN), in
groups of 128 weights, and writes everything into `exports/opt125m-rtn4-g128/`:

```bash
uv run ptq export-mlir --model opt-125m --bits 4 --group-size 128 --out exports/opt125m-rtn4-g128
```

The first run downloads the model (250 MB) if it is not already on disk. On a laptop
this takes about 15 seconds after the download. When it finishes you see a short summary:

```
exported facebook/opt-125m rtn4g128 -> exports/opt125m-rtn4-g128
  quant_key                    eb8aeb32c3b2
  quantized modules            72
  dequant ops                  quantized_decomposed.dequantize_per_channel_group
  mean rel. weight error       0.109
  bitwise = evaluated          True
  mlir                         exports/opt125m-rtn4-g128/model.mlir (480.0 MB)
```

`bitwise = evaluated: True` is the line that matters. It means the exported weights are
identical to the ones the benchmark measured.

## Step 4: Look at the output

```bash
ls -lh exports/opt125m-rtn4-g128
```

- **`export.json`** (small): open it in any text editor. `quant` shows the settings, `summary`
  the overall error, `mlir.ops` how many quantized layers the MLIR contains.
- **`layers.json`** (small): one entry per layer, named like
  `model.decoder.layers.0.self_attn.q_proj`. The useful numbers:
  - `rel_error`: how far the rounded weights are from the originals. 0.11 means 11%.
  - `codes_used`: how many of the 16 possible 4-bit values the layer actually uses.
  - `bitwise_equal_to_evaluated`: should always be `true`.
- **`model.mlir`** (large, hundreds of MB): the model itself. Do not open it in a normal
  editor. To peek at the first quantized layer:

  ```bash
  grep -m1 dequantize_per_channel_group exports/opt125m-rtn4-g128/model.mlir | cut -c1-200
  ```

## Step 5: Variations

Change the method with `--algo`. GPTQ and AWQ-lite need sample text, which downloads on
first use and takes a few minutes:

```bash
uv run ptq export-mlir --model opt-125m --bits 4 --group-size 128 --algo gptq --out exports/opt125m-gptq4
uv run ptq export-mlir --model opt-125m --bits 3 --algo hqq --out exports/opt125m-hqq3
```

Add a per-layer *output* error, measured on real text, to `layers.json`:

```bash
uv run ptq export-mlir --model opt-125m --bits 4 --group-size 128 --score-windows 4 --out exports/scored
```

Other switches you may want:

| switch | effect |
|---|---|
| `--model facebook/opt-350m` | any Hugging Face model id, or a local folder, not only the configured keys |
| `--no-mlir` | skip the big `.mlir` file; just the JSON and the check |
| `--output-type linalg` or `both` | also (or only) write `model.linalg.mlir`, the lowered form; see below |
| `--explicit-positions` | pass the token positions to the model instead of letting it compute them; needed for a lowered file that a plain `mlir-opt` can read; see below |
| `--bytecode` | also write `model.mlirbc`, a compact binary version |
| `--dynamic-seqlen` | let the exported model accept any input length |
| `--device cpu` | run on the CPU even if a GPU is present (put it before `export-mlir`) |

## Golden test vectors: `--golden`

Anyone who runs the exported function somewhere else (a simulator, a C harness, another
compiler) needs a known input and the answer the real model gives for it. `--golden`
writes exactly that under `golden/`:

```
golden/index.json                   what every tensor is, where it came from, and the tolerance
golden/main/inputs/input_ids.npy    the input, numpy format (shape and dtype inside)
golden/main/inputs/input_ids.bin    the same bytes raw: little-endian, row-major, no header
golden/main/outputs/logits.npy      the output PyTorch produced for that input
golden/main/outputs/logits.bin
```

- **The input is real text, never random numbers.** It is the first `--seqlen` tokens of
  one window of the benchmark's own WikiText-2 stream (`--golden-dataset` chooses the
  text, `--golden-seed` the window; the index records which window). The same tensor is
  the example input the MLIR was traced with.
- **The index lists, per tensor:** name, role (input or output), shape, dtype, byte
  length, both file names and a SHA-256 of the bytes.
- **The tolerance is measured, not guessed.** The function is run a second time in
  float64 on the same weights (the dequantized values, which float64 holds exactly), so
  the distance between the two outputs is the reference's own fp32 rounding. The index
  records that distance (`max_abs_err_vs_fp64`, `rel_fro_err_vs_fp64`) and the
  recommended tolerance: `atol` = 8 × the measured distance (with a floor of 8 fp32
  last-place units of the largest output value), `rel_fro_tol` = 8 × the measured
  relative distance. The factor 8 is two for the consumer's own independent rounding and
  four for a different accumulation order, fused multiply-adds and a different `exp` or
  `rsqrt` in a bare-metal math library; it is a named constant in
  `src/ptqbench/export/golden.py` and is written into the index as `tolerance_policy`.
  A consumer passes when `max|got − golden| ≤ atol` and `‖got − golden‖ / ‖golden‖ ≤
  rel_fro_tol`. For logits the index also says how often the top token agreed between
  fp32 and float64 (`argmax_agreement`), a coarser check that needs no tolerance.

`export.json` repeats the headline numbers under `golden`.

## One piece of the model: `--slice`

A whole model is a lot to compile at once. `--slice` exports one decoder block, or just
its attention or its MLP, chosen by `--block-index`, as a function of the hidden states
that piece really receives:

```bash
uv run ptq export-mlir --model smollm2-135m --bits 4 --group-size 64 \
    --slice block --block-index 3 --golden --output-type both --out exports/smollm2-block3
```

How it works: the model reads a real text sample (the same seeded WikiText-2 window
`--golden` uses), and forward hooks record what block 3 was handed, what it returned,
and every other argument it received: Llama's rotary cos/sin tables, the position ids,
the (absent) attention mask and cache. The exported function `block_3(hidden_states)`
replays those other arguments as constants, so it is a pure function of one tensor. It
is checked to reproduce the recorded output bit for bit before anything is written.

| `--slice` | function | input | output |
|---|---|---|---|
| `block` | `block_<i>` | `[1, S, hidden]` hidden states entering the block | the block's output, same shape |
| `attention` | `attention_<i>` | the post-norm hidden states entering `self_attn` | `attn_output` (the attention weights are dropped) |
| `mlp` | `mlp_<i>` | the hidden states entering the MLP: Llama `[1, S, hidden]`, OPT `[S, hidden]` because the OPT block flattens before `fc1` | the MLP's output; the residual add stays outside |

Files are named after the function (`block_3.mlir`, `block_3.linalg.mlir`), `model.mlir`
is not written, and `export.json` describes the piece under `slice`: shapes, dtype, the
quantized layers inside it, and every constant that was frozen in. With `--golden` the
recorded input and output are the golden vectors. Works for the Llama and OPT families
through `src/ptqbench/models/families.py`, which names each family's attention module
and MLP (OPT's `fc1 -> activation -> fc2` chain is wrapped as one module there).
SmolLM2-135M, the primary target, has hidden size 576, so use `--group-size 64`.

## Weights outside the IR: `--weights external`

By default every weight sits inside the `.mlir` as a hex blob, which is why the file is
hundreds of megabytes. `--weights external` writes a graph-only `.mlir` whose function
takes the weights as arguments, plus two files:

```bash
uv run ptq export-mlir --model smollm2-135m --bits 4 --group-size 64 \
    --weights external --golden --output-type both --explicit-positions --out exports/smollm2-ext
```

- **`weights.bin`**: every tensor, one after another, each starting at a multiple of
  **64 bytes** (`WEIGHT_ALIGNMENT_BYTES` in `src/ptqbench/export/weights.py`, written
  into the index as `alignment`). Scales, zero-points, biases and unquantized tensors are
  stored in exactly the dtype the function argument has; codes narrower than a byte are
  packed.
- **`weights.json`**: per tensor its `name` (the model's own module path, e.g.
  `model.layers.3.mlp.gate_proj.weight_q`), `role` (`codes`, `scale`, `zero`, `g_idx`,
  `bias`, `float` for unquantized float tensors such as embeddings, norms and rotary
  tables, `int` for integer constants), `shape`, `dtype`, `bits`, `group_size`,
  `scheme`, byte `offset` and `length`, whether it is `packed` and, if so, its
  `row_stride_bytes`, a SHA-256, and any `aliases` (a tied `lm_head.weight` is stored
  once under the embedding's name). `functions` lists, per exported function, which
  tensor each argument is, in argument order.
- **The `.mlir`** has no `dense_resource` at all. `main(%input_ids, %arg1, %arg2, ...)`
  takes the weights in the order the graph first uses them (execution order); the
  `functions` entry maps every `%argN` to a tensor.

**Packing, defined once.** Codes are packed row by row as a little-endian bit stream.
Row `r` of a `(rows, cols)` code matrix takes `ceil(cols × bits / 8)` bytes; within a
row, code `j` occupies bits `[j·bits, (j+1)·bits)` of the row's bit stream, least
significant bit first, and bit `k` of the stream is bit `k mod 8` of byte `k ÷ 8`. So at
4 bits code 0 is the **low nibble** of byte 0 and code 1 the high nibble; at 3 bits codes
straddle byte boundaries (codes `[1, 2, 3]` become bytes `[0xD1, 0x00]`); at 2 bits four
codes share a byte. Unused trailing bits of a row's last byte are zero, so a group of 64
or 128 codes always starts on a byte boundary. 8-bit codes are one byte each and marked
`packed: false`. In C: `code_j = (row >> (j * bits)) & ((1 << bits) - 1)` with the row
read as a little-endian integer. `src/ptqbench/export/weights.py` has `pack` and
`unpack`, tested to round-trip at every width from 2 to 8 bits.

**Why the importer hook, and not a wrapper.** torch-mlir asks a hook how to turn each
parameter into IR; the embedded export answers "a dense resource", the external one
"a new function argument". The graph is otherwise identical, tied tensors become one
argument, and every argument keeps its module name. Wrapping the model in
`torch.func.functional_call` would also work but adds a layer, duplicates tied weights
and loses the name mapping.

**Checked:** `tests/test_export_weights.py` rebuilds a model from `weights.bin` and
`weights.json` alone and gets the golden logits back bit for bit.

## Prefill, prefill-KV and decode: `--stages`

Text generation runs a model in three ways: the whole prompt at once (**prefill**), more
prompt on top of a saved history (**prefill-KV**), and one token at a time (**decode**).
`--stages` exports each as its own function, with the key/value cache passed in and out
as plain tensors, following the three-module split of arXiv 2607.15865:

```bash
uv run ptq export-mlir --model smollm2-135m --bits 4 --group-size 64 \
    --stages --max-cache-len 256 --golden --weights external --output-type both --out exports/smollm2-stages
```

| function | inputs | outputs |
|---|---|---|
| `prefill` | `input_ids [1, S]` int64 | `logits [1, S, V]`, `k_cache`, `v_cache` |
| `prefill_kv` | `input_ids [1, S2]`, `k_cache`, `v_cache`, `position` | `logits [1, S2, V]`, `k_cache`, `v_cache` |
| `decode` | `input_ids [1, 1]`, `k_cache`, `v_cache`, `position` | `logits [1, 1, V]`, `k_cache`, `v_cache` |

- `k_cache` and `v_cache` are `[layers, 1, kv_heads, max_cache_len, head_dim]` in the
  model's dtype (SmolLM2: `[30, 1, 3, 256, 64]` float32 on the CPU). `position` is a 0-d
  int64 tensor: how many tokens are already in the cache, which is also the position of
  the first new token. A consumer keeps `position + new tokens <= max_cache_len`.
- Every shape is static: `S` is `--seqlen` (default 128), `S2` is `--prefill-kv-seqlen`
  (default 16), `max_cache_len` is `--max-cache-len` (default 256). `prefill_kv` and
  `decode` are the same `nn.Module` traced at two input lengths.
- Files: `prefill.mlir`, `prefill_kv.mlir`, `decode.mlir` (plus `.linalg.mlir` forms);
  with `--weights external` one `weights.bin` serves all three, and `weights.json`
  records each function's argument order separately.
- **Checked against the real model.** The export runs prefill, then prefill-KV, then
  four decode steps on a real text window (each decode step fed the true next token),
  through the exact modules that were exported, and compares every logit with the
  eager model run once over the whole window without a cache. The tolerance is the
  golden one for the prefill logits (8 × the measured fp32-vs-float64 distance); the
  export fails if any step exceeds it, and `export.json` records every step under
  `stages.check`.
- With `--golden`, each stage gets its own golden case; the caches `prefill_kv` and
  `decode` are given are the real outputs of the previous stage.

How the cache became explicit: transformers 5.17 has `StaticCache`, whose layers update
preallocated tensors in place with `index_copy_`, and an ExecuTorch recipe that exports
those tensors as mutated buffers. Neither gives a function with the cache as inputs and
outputs, and the in-place copy lowers to `tm_tensor.scatter`, which a plain `mlir-opt`
cannot read. The export subclasses `StaticLayer` with an out-of-place update (gather the
new keys into place, merge with `where`), inherits the static-cache mask logic
unchanged, and turns `use_cache` on only inside the exported function; the evaluation
loader's `use_cache=False` is untouched. Llama-family models work as is; OPT needs its
position ids and an all-ones mask passed explicitly (its own derivation is
data-dependent and torch.export refuses it), which the export does. Models with
sliding-window layers are refused.

## The lowered form: `--output-type linalg`

`model.mlir` is the model in torch-mlir's *torch dialect*: one op per PyTorch operation,
with the integer-to-float conversion of every quantized layer kept as a single
`quantized_decomposed.dequantize_per_channel_group` op. A compiler usually wants the next
level down, **linalg-on-tensors**: matrix multiplies as `linalg.matmul`, everything else as
loops over tensors (`linalg.generic`), with plain `arith` and `math` ops inside. That is
what `--output-type linalg` writes to `model.linalg.mlir`; `--output-type both` writes
both files from the same export:

```bash
uv run ptq export-mlir --model smollm2-135m --bits 4 --group-size 64 \
    --output-type both --explicit-positions --out exports/smollm2-rtn4-g64
```

Three things to know about the lowered file:

- **It is checked before it is written.** The MLIR verifier must accept it, it must contain
  linalg ops, and no torch-dialect op may be left. `export.json` lists the linalg op counts
  under `mlir.linalg_ops`.
- **Dequantization is no longer visible as such.** After lowering, "take the 4-bit code,
  subtract the zero-point, multiply by the scale" is just integer-to-float arithmetic
  (`arith.extui`, `arith.sitofp`, `arith.subf`, `arith.mulf`) inside a `linalg.generic`.
  The codes are an `i8` tensor like any other. Only `model.mlir` and `layers.json` still
  say which tensors are 4-bit and how they are grouped.
- **Use `--explicit-positions` for a file other tools can read.** Without it, the model
  computes its token positions with a running sum (`cumsum`), which lowers to
  `tm_tensor.scan`, an op from torch-mlir's own private dialect: `torch-mlir-opt` reads it,
  a plain `mlir-opt` does not, and `export.json` adds a note when it is present. With
  `--explicit-positions` the positions `0..S-1` and an all-ones attention mask are passed
  in, the model is unchanged, and the lowered file contains only standard dialects.

## Step 6: Clean up

The `exports/` folder is ignored by git and can be deleted at any time:

```bash
rm -rf exports/opt125m-rtn4-g128
```

## If something goes wrong

| message | what it means | what to do |
|---|---|---|
| `torch-mlir is not installed` | Step 2 was skipped | run the `uv sync` line in Step 2, or add `--no-mlir` |
| `is not a valid model identifier` or another download error | the model name is wrong or the machine is offline | check the name at huggingface.co, or use a key from `configs/models/` such as `opt-125m` |
| `in_features ... is not divisible by group_size` | this model's layers do not split into groups of that size | try `--group-size 64` or `--group-size -1` |
| `CUDA out of memory` | the GPU is too small for this model plus the quantizer | add `--device cpu` before `export-mlir`, or `--eval-mode streamed` |
| the run is slow or the file is huge | the `.mlir` text stores every weight; big models make big files | use a smaller model, or `--no-mlir` if you only need the JSON |

## What the export can and cannot represent

- Per-row and grouped grids (the normal settings) use torch-mlir's native quantized
  operations and lower all the way to its linalg backend.
- GPTQ with `--act-order` and a group size, and every HQQ export, use plain arithmetic
  operations instead, because torch-mlir has no quantized operation for scattered groups or
  float zero-points. They are still exact; `export.json` lists this under `notes`.
- Only weights are quantized. The model's activations stay in floating point, so the export
  is "convert the weights, then multiply in float", not an integer-only model.

More detail: the "Exporting a quantized model to torch-mlir" section of the README, and the
docstring at the top of `src/ptqbench/export/quantized.py`.
