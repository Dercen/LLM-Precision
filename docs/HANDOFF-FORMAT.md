# Hand-off format: what `ptq export-mlir` gives the vault compiler

This is the contract between this repository and the compiler that turns a quantized
model into one bare-metal program per vault (torch-mlir ingestion, spatial slicing,
tiling, LLVM/RVV, one ELF per vault). It lists every file an export can produce, every
function signature, the weight-file layout, and the golden vectors a simulation run is
checked against. The numbers are for the primary target, `smollm2-135m` (Llama
architecture: 30 blocks, hidden size 576, 9 query heads, 3 key/value heads, head size
64, vocabulary 49,152, group size 64), quantized RTN 4-bit and exported on the CPU, so
every float is `f32`. The commands that produce each artifact are at the end.

## 1. Files

One export directory holds one of three things: the whole model, one slice, or the
three stages. The common files come first.

| file | written | contents |
|---|---|---|
| `export.json` | always | what was exported and with which settings; `files` lists every other file; `mlir` the op counts and forms; `golden`, `slice`, `stages`, `weights` the sections described below; `notes` anything a consumer must know (for example a `tm_tensor` op) |
| `layers.json` | always | one entry per quantized Linear, keyed by module name (`model.layers.3.mlp.gate_proj`): bits, group size, scheme, dequantization op, weight error, and `bitwise_equal_to_evaluated` (always `true`: the exported integers reproduce the weights the perplexity numbers were measured with) |
| `model.mlir` | default | the whole model as one function `main` in torch-mlir's **raw torch dialect**, weights embedded as `dense_resource` blobs. The only form in which dequantization is still a named op (`quantized_decomposed.dequantize_per_channel_group`) |
| `model.linalg.mlir` | `--output-type linalg` or `both` | the same function lowered to **linalg-on-tensors** (linalg, arith, tensor, math, cf on static tensors). Verified: at least one linalg op, no torch-dialect op left. Dequantization is plain arithmetic here |
| `<kind>_<i>.mlir`, `<kind>_<i>.linalg.mlir` | `--slice` | one piece of block `i` instead of the whole model (section 2.2); `model.mlir` is not written |
| `prefill.mlir`, `prefill_kv.mlir`, `decode.mlir` (+ `.linalg.mlir`) | `--stages` | the three generation stages (section 2.3); `model.mlir` is not written |
| `weights.bin`, `weights.json` | `--weights external` | every weight outside the IR, packed and aligned (section 3); the `.mlir` files are then graph-only and take the weights as arguments |
| `golden/index.json`, `golden/<function>/inputs/*`, `golden/<function>/outputs/*` | `--golden` | a real input and the reference output for every exported function, with tolerances (section 4) |
| `*.mlirbc` | `--bytecode` | MLIR bytecode twins of the `.mlir` files |

Sizes for SmolLM2-135M RTN4 g64 on the CPU: `model.mlir` 445 MB embedded, 1.1 MB
graph-only; `weights.bin` 171 MB; `block_3.mlir` 7.8 MB; the golden logits for a
128-token prompt 25 MB.

Two things about the lowered form that a consumer should know:

- A whole-model graph exported as plain `model(input_ids)` keeps one `cumsum` (OPT
  derives its position ids from the mask; Llama's mask builder checks the position ids
  for packed sequences), which lowers to `tm_tensor.scan`. `tm_tensor` is torch-mlir's
  private dialect: `torch-mlir-opt` reads it, a stock `mlir-opt` does not.
  `--explicit-positions` removes the `cumsum`; slices and stages never have one.
  `export.json` counts any `tm_tensor` ops under `mlir.tm_tensor_ops` (per function for
  stages) and adds a note. Every artifact listed in section 6 was exported without any.
- Unsigned 8-bit codes are `!torch.vtensor<[...],ui8>` in the raw form and signless
  `tensor<...xi8>` after lowering; the arithmetic that consumes them is `arith.extui`
  (unsigned widening), so the bytes are to be read as unsigned.

## 2. Functions

Every function is a `func.func` in its own file, with static shapes. Batch is always 1.
Token ids are `i64`. Floats are the model's dtype (`f32` on the CPU, `f16` or `bf16`
if exported on the GPU; `export.json["dtype"]` says which).

### 2.1 Whole model: `main`

```
main(input_ids: tensor<1x128xi64>) -> tensor<1x128x49152xf32>      // logits
```

`128` is `--seqlen`. Positions are `0..127`. With `--weights external` the signature
continues `, %arg1: tensor<49152x576xf32>, %arg2: tensor<32xf32>, ...` with one argument
per tensor in `weights.json["functions"]["main"]["arguments"]`, in the order the graph
first uses them (execution order: embeddings, rotary table, then block 0's norm, its
codes/scale/zero, and so on).

### 2.2 Slices: `block_i`, `attention_i`, `mlp_i`

`--slice block|attention|mlp --block-index i` exports one piece of block `i` as a
function of the hidden states that piece actually receives. The model is run on a real
text sample; forward hooks capture the piece's input, its output, and everything else
it was handed (the rotary `cos`/`sin` tables, position ids, the absent mask), and the
exported function replays those other arguments as constants.

| function | input | output | notes |
|---|---|---|---|
| `block_i` | `hidden_states tensor<1x128x576xf32>`, the block's input | `tensor<1x128x576xf32>`, the block's output | the whole block: norms, attention, MLP, both residual adds |
| `attention_i` | the post-norm hidden states entering `self_attn`, `tensor<1x128x576xf32>` | `attn_output tensor<1x128x576xf32>` | q/k/v/o projections, rotary embedding and softmax attention; no residual, no norm |
| `mlp_i` | the hidden states entering the MLP: Llama `tensor<1x128x576xf32>`, OPT `tensor<128xhidden>` (the OPT block flattens before `fc1`) | `mlp_output`, same shape | gate/up/down (Llama) or `fc1`/act/`fc2` (OPT); no residual, no norm |

`export.json["slice"]` gives the shapes, the quantized layers inside the piece, and
every constant that was frozen in (`captured_kwargs`), e.g. `position_embeddings_0`
and `_1`, the rotary cos and sin of shape `[1, 128, 64]`. With `--weights external`
those constants are entries of `weights.json` with role `float` (or `int`), named as in
`captured_kwargs`. The replay is checked to reproduce the captured output bit for bit
(`replay_bitwise: true`).

### 2.3 Stages: `prefill`, `prefill_kv`, `decode`

`--stages` exports text generation as three functions with an explicit static
key/value cache, after the three-module split of arXiv 2607.15865.

```
prefill   (input_ids: tensor<1x128xi64>)
          -> (logits: tensor<1x128x49152xf32>, k_cache: tensor<30x1x3x256x64xf32>, v_cache: tensor<30x1x3x256x64xf32>)

prefill_kv(input_ids: tensor<1x16xi64>, k_cache: tensor<30x1x3x256x64xf32>, v_cache: tensor<30x1x3x256x64xf32>, position: tensor<i64>)
          -> (logits: tensor<1x16x49152xf32>, k_cache: tensor<30x1x3x256x64xf32>, v_cache: tensor<30x1x3x256x64xf32>)

decode    (input_ids: tensor<1x1xi64>, k_cache: tensor<30x1x3x256x64xf32>, v_cache: tensor<30x1x3x256x64xf32>, position: tensor<i64>)
          -> (logits: tensor<1x1x49152xf32>, k_cache: tensor<30x1x3x256x64xf32>, v_cache: tensor<30x1x3x256x64xf32>)
```

- **What each stage means.** `prefill` reads the whole prompt at once and starts from
  an empty cache (zeros, position 0, created inside the function). `prefill_kv` reads
  more prompt on top of an existing cache. `decode` reads one token on top of an
  existing cache. `prefill_kv` and `decode` are the same computation at two input
  lengths; `decode` is the one that runs once per generated token.
- **Cache layout** `[layers, batch=1, kv_heads, max_cache_len, head_dim]`, one tensor
  for keys and one for values, in the model's dtype. Slot `t` holds the key/value of
  the token at position `t`; slots at or beyond `position + new tokens` are untouched
  (zeros after `prefill`). The attention mask inside the function hides every slot the
  current tokens must not see, so the garbage in unused slots never matters.
- **`position`** is a 0-d `i64` tensor: the number of tokens already in the cache,
  which is also the position of the first new token. `prefill` sets it to 0
  internally and `prefill_kv`/`decode` take it as an argument. The consumer's contract
  is `position + new tokens <= max_cache_len`; writing past the cache is undefined.
- **Static sizes** come from three named options: the prompt length `--seqlen`
  (128), the second chunk `--prefill-kv-seqlen` (16) and `--max-cache-len` (256).
- **Round trip.** The export runs `prefill`, then `prefill_kv`, then four `decode`
  steps on a real text window (each decode step fed the true next token) through the
  exact modules that were exported, and compares every logit with the eager model run
  once over the whole window with no cache. `export.json["stages"]["check"]` records
  every comparison; the export fails if a step exceeds the tolerance of section 4.
  SmolLM2 RTN4 g64: see section 6 for the measured numbers.
- **How the cache is updated**, since it shows in the IR: the new keys are gathered
  into position with `index_select` and merged into the old cache with `select`
  (`torch.where`), all plain elementwise ops, so the lowered form has no scatter and no
  `tm_tensor`. It is `O(max_cache_len)` work per step, which a compiler may turn back
  into an in-place write once it has proven the slots.
- With `--weights external` one `weights.bin` serves all three; each function's
  argument order is recorded separately under `weights.json["functions"]`.

## 3. Weights

### 3.1 `weights.json`

```json
{
  "format_version": 1, "file": "weights.bin", "byte_order": "little", "alignment": 64,
  "packing": { "bit_order": "lsb-first", "rule": "...", "example_4bit": "...", "example_3bit": "...", "c_read": "..." },
  "total_bytes": 179742080, "n_tensors": 693,
  "roles": { "codes": 210, "scale": 210, "zero": 210, "g_idx": 0, "bias": 0, "float": 63, "int": 0 },
  "tensors": [ { "name": "model.layers.0.self_attn.q_proj.weight_q", "role": "codes", "shape": [576, 576],
                 "dtype": "uint8", "storage_dtype": "uint8", "bits": 4, "group_size": 64, "scheme": "per_group",
                 "offset": 113248640, "length": 165888, "packed": true, "row_stride_bytes": 288,
                 "sha256": "...", "module": "model.layers.0.self_attn.q_proj", "aliases": [],
                 "quant": { "sym": false, "zero_point_domain": "int", "quant_min": 0, "quant_max": 15, "n_groups": 9, "algo": "rtn" } },
               ... ],
  "functions": { "main": { "n_user_inputs": 1, "arguments": [ { "index": 1, "name": "model.embed_tokens.weight" }, ... ] } }
}
```

Per tensor: `name` is the model's own module path (the same keys as `layers.json`
for the quantized leaves); `role` is one of

| role | what | dtype |
|---|---|---|
| `codes` | the integer codes `q` of a quantized Linear, `(out_features, in_features)` | `uint8` in the IR; packed in the file when `bits < 8` |
| `scale` | `(out_features, n_groups)` | `float32` |
| `zero` | `(out_features, n_groups)` zero-points | `int32` (integer grids: RTN, GPTQ, AWQ-lite), `float32` for HQQ (`zero_point_domain: "float"`) |
| `g_idx` | `(in_features,)` group index per input column, only for GPTQ with `--act-order` and a group size (`scheme: per_group_permuted`) | `int64` |
| `bias` | a quantized Linear's bias (OPT has them, Llama does not) | model dtype |
| `float` | every unquantized float tensor: token embeddings (also the tied `lm_head`, listed under `aliases`), norm weights, the rotary `inv_freq` table, a slice's captured `cos`/`sin` | model dtype |
| `int` | integer constants a slice captured, if the piece uses them | as captured |

`shape` and `dtype` are those of the function argument; `storage_dtype` is the numpy
dtype of the bytes (`uint16` for bfloat16). `offset`/`length` locate the bytes;
`packed` says whether `unpack` is needed; `row_stride_bytes` is the packed row size.

### 3.2 Dequantization, so the arithmetic in the IR can be recognised

For a quantized Linear with codes `q[r, c]`, `scheme`:

- `per_row` (`group_size = -1`, `n_groups = 1`): `w[r, c] = scale[r, 0] * (q[r, c] - zero[r, 0])`
- `per_group` (`group_size = G`): `g = c // G`, `w[r, c] = scale[r, g] * (q[r, c] - zero[r, g])`
- `per_group_permuted` (GPTQ act-order): `g = g_idx[c]`, same formula

computed in `f32` (`compute_dtype`), then cast to the model dtype; the Linear is
`y = x @ w^T + bias`. In the raw form this is one `dequantize_per_channel_group` op
(or explicit `index_select`/`sub`/`mul` for `per_group_permuted` and HQQ); in the
lowered form it is `arith.extui` (ui8 → i32), `arith.sitofp`, `arith.subf`, `arith.mulf`
inside a `linalg.generic` feeding a `linalg.matmul`. The IR consumes codes **unpacked**,
one byte per code, shape `(rows, cols)`.

### 3.3 Packing: `pack` / `unpack`

Codes narrower than a byte are packed **row by row** as a **little-endian bit stream**:

1. Row `r` of a `(rows, cols)` code matrix becomes `row_stride_bytes = ceil(cols × bits / 8)`
   bytes; rows follow each other with no gap.
2. Within a row, code `j` occupies stream bits `[j × bits, (j + 1) × bits)`, least
   significant bit first.
3. Stream bit `k` is bit `k mod 8` of byte `k ÷ 8`.
4. Trailing bits of a row's last byte are zero.

So at 4 bits, code 0 is the **low nibble** of byte 0 and code 1 the high nibble
(`codes [1, 2, 3, 4] → bytes [0x21, 0x43]`). At 3 bits, codes straddle byte boundaries
(`codes [1, 2, 3] → bytes [0xD1, 0x00]`; code 2 is bits 6..8). At 2 bits, four codes per
byte (`[1, 2, 3, 0] → [0x39]`). 8-bit codes are one byte each (`packed: false`). Any
group size that is a multiple of 8 keeps every group on a byte boundary
(`64 × bits / 8 = 8 × bits` bytes per group). In C:

```c
/* row: pointer to the row's bytes; j: code index; bits: 2..8 */
unsigned code_at(const uint8_t *row, unsigned j, unsigned bits) {
    unsigned bit = j * bits, byte = bit >> 3, shift = bit & 7;
    unsigned v = row[byte] | ((unsigned)row[byte + 1] << 8);   /* a code spans at most two bytes for bits <= 8 */
    return (v >> shift) & ((1u << bits) - 1u);
}
```

(Read `row[byte + 1]` only when `shift + bits > 8`; the Python `unpack` in
`src/ptqbench/export/weights.py` is the reference and is tested at every width.)

### 3.4 Alignment

Every tensor's `offset` in `weights.bin` is a multiple of **64 bytes**
(`WEIGHT_ALIGNMENT_BYTES` in `src/ptqbench/export/weights.py`, repeated as
`weights.json["alignment"]`); the gaps are zero bytes. 64 is a cache line and a
multiple of every RISC-V vector element width, so the first vector load of a tensor
never straddles a line.

## 4. Golden vectors

`--golden` writes, for every exported function, a fixed input and the output PyTorch
produces for it. Layout:

```
golden/index.json
golden/<function>/inputs/<name>.npy      numpy format: shape and dtype inside
golden/<function>/inputs/<name>.bin      the same bytes: little-endian, row-major, no header
golden/<function>/outputs/<name>.npy
golden/<function>/outputs/<name>.bin
```

`index.json["functions"][<function>]` lists `inputs` and `outputs` in argument/result
order, each with `name`, `role`, `shape`, `dtype`, `storage_dtype`, `byte_length`,
`npy`, `bin` (relative to the function directory) and `sha256`; `source` says where
the input came from; `tolerance` is the per-output report below.

**Inputs are real.** The whole model and `prefill` take the first `--seqlen` tokens of
one seeded window of the WikiText-2 test stream the benchmark itself evaluates on
(`source`: dataset, split, window index, length, the stream's first-token hash).
`prefill_kv` and `decode` take the next tokens of the same window, and their `k_cache`,
`v_cache` and `position` inputs are the previous stage's actual outputs (`cache_from`).
A slice takes the hidden states its block received on that window
(`kind: captured_activations`). Nothing is random noise.

**Tolerance, derived not guessed.** Each function is rerun in float64 on the same
weights (the dequantized `f32` values, exact in float64), so the distance between the
two outputs is the reference's own fp32 rounding. The index records, per output,
`max_abs_err_vs_fp64`, `rel_fro_err_vs_fp64` (‖diff‖/‖exact‖), `max_abs_value`, and for
logits `argmax_agreement`, then

```
atol        = max(8 × max_abs_err_vs_fp64, 8 × 2^-23 × max_abs_value)
rel_fro_tol = 8 × rel_fro_err_vs_fp64
```

The factor 8 (`TOLERANCE_SAFETY_FACTOR`) is 2 for the consumer's own independent fp32
rounding and 4 for a different accumulation order, fused multiply-adds and a different
`exp`/`rsqrt` in a bare-metal libm; the floor is 8 fp32 last-place units of the largest
output value. The rule and constants are written into every index as
`tolerance_policy`. A consumer passes when, for every output,

```
max |got − golden| ≤ atol   and   ‖got − golden‖ / ‖golden‖ ≤ rel_fro_tol
```

Measured for SmolLM2-135M RTN4 g64 on the CPU, 128-token prompt: logits of magnitude
up to 33 differ from float64 by at most 5.0e-5 → `atol` 4.0e-4, `rel_fro_tol` 5.8e-6,
top token identical at every position. For `block_3`: 4.9e-5 → `atol` 6.4e-4. These
are what an SST run of the same function should be compared with; `argmax_agreement`
is the coarse check that needs no tolerance.

## 5. Things to confirm

- **Element types.** `lower.py`'s signature parser accepts `f32` tensor arguments only.
  The external-weights functions take `i8` (codes), `i32` (zero-points), `i64` (token
  ids, position, any `g_idx`) and 0-d `tensor<i64>` (position) as well. Either the
  parser grows, or the weights stay embedded (`--weights embedded`, the default) and
  only `input_ids` is an argument.
- **Which form.** The linalg form has dequantization folded into `linalg.generic`
  bodies; a vault program that wants to keep 4-bit codes in memory and dequantize on the
  fly must recognise that pattern or start from the raw form, where it is one named op.
- **The cache update** is select-plus-gather over the whole cache per step (no scatter,
  no `tm_tensor`). Fine for correctness; a compiler should turn it into a slot write.
- **dtype.** These artifacts are `f32` (CPU export). A GPU export is `f16`/`bf16`; say
  which the vault toolchain wants.
- **`tm_tensor`.** Only the plain whole-model `main` without `--explicit-positions`
  carries one; every artifact in section 6 was produced without any.
- **Biases.** Llama models have none, so `roles.bias` is 0 for SmolLM2; OPT has them.
- **Activation quantization** is out of scope and the number formats are undecided.
  Nothing here precludes it: activations are plain `f32` tensors at every boundary, and a
  future W8A8 export would add activation scales as more `float` entries and change
  the `linalg.generic` bodies, not the file layout.

## 6. Commands that produced the artifacts referred to above

```bash
source scripts/env.sh
# whole model, both forms, golden vectors, no tm_tensor
uv run ptq --device cpu export-mlir --model smollm2-135m --bits 4 --group-size 64 \
    --output-type both --explicit-positions --golden --out exports/smollm2-rtn4-g64
# one block
uv run ptq --device cpu export-mlir --model smollm2-135m --bits 4 --group-size 64 \
    --slice block --block-index 3 --golden --output-type both --out exports/smollm2-block3
# whole model, weights outside the IR
uv run ptq --device cpu export-mlir --model smollm2-135m --bits 4 --group-size 64 \
    --weights external --golden --output-type both --explicit-positions --out exports/smollm2-ext
# the three stages
uv run ptq --device cpu export-mlir --model smollm2-135m --bits 4 --group-size 64 \
    --stages --max-cache-len 256 --golden --weights external --output-type both --out exports/smollm2-stages
```

`--algo gptq --act-order`, `--algo awq_lite` and `--algo hqq` work the same way; see
`docs/EXPORT-MLIR.md` for every switch and `README.md` for what each method maps onto.
