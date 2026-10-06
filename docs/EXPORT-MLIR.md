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
| `--bytecode` | also write `model.mlirbc`, a compact binary version |
| `--dynamic-seqlen` | let the exported model accept any input length |
| `--device cpu` | run on the CPU even if a GPU is present (put it before `export-mlir`) |

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
