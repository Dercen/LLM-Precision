"""Weight footprint of a quantization configuration, from config.json alone.

Two numbers per row, the ones a memory-bound accelerator cares about:

* `bits_per_weight`: what one weight of the quantized Linears costs once its share of the
  scales, zero-points and (act-order GPTQ) group index is counted. 4-bit with groups of
  128 and a 16-bit scale plus a 4-bit zero-point is 4 + 20/128 = 4.16 bits.
* `weight_bytes`: the whole model's weights as the hand-off format stores them
  (`ptqbench.export.weights`): the quantized Linears packed as above, everything else
  (embeddings, norms, biases, OPT's positional table) untouched in the model's dtype.

The Linear shapes come from the Hugging Face config the same way `loader.estimate_model_bytes`
derives them, per family (`ptqbench.models.families`). The remaining parameters are
counted from the config as well; when a row carries the loaded model's real byte count
(`model_bytes`) that is used instead, which makes the total exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

#: One 16-bit scale (fp16 or bf16) per group of weights.
SCALE_BITS = 16
#: hqq's zero-points are the output of its optimizer, not integers: stored as 16-bit floats.
FLOAT_ZERO_BITS = 16
#: Symmetric grids pin the zero-point at mid-range, so nothing is stored for it.
SYM_ZERO_BITS = 0
#: Act-order GPTQ with groups assigns scattered columns to each group, so every input
#: column carries a group index. The export stores it as int32; 16 bits hold any group
#: count met here and is what a packed format would spend.
GROUP_INDEX_BITS = 16
#: A full-precision (fp) row keeps every weight in the model's 16-bit dtype.
FP_BITS = 16
#: Methods whose zero-point is a float rather than an integer code.
FLOAT_ZERO_ALGOS = frozenset({"hqq"})


@dataclass(frozen=True)
class LinearShape:
    """One quantized Linear of a decoder block: `weight` is `(out_features, in_features)`."""

    name: str
    out_features: int
    in_features: int
    bias: bool

    @property
    def weights(self) -> int:
        return self.out_features * self.in_features


@dataclass(frozen=True)
class Footprint:
    quantized_weights: int  #: weights in the quantized Linears
    quantized_bits: int  #: bits they take: codes + scales + zero-points (+ group index)
    other_bits: int  #: every other parameter, in the model dtype
    exact: bool  #: `other_bits` came from the loaded model's byte count, not the config

    @property
    def bits_per_weight(self) -> float:
        return self.quantized_bits / self.quantized_weights

    @property
    def weight_bytes(self) -> int:
        return math.ceil((self.quantized_bits + self.other_bits) / 8)


def text_config(cfg: Any) -> Any:
    """The text sub-config of a multimodal config, else the config itself."""
    return getattr(cfg, "text_config", None) or cfg


def family_name(cfg: Any) -> str:
    """`opt`, `llama` or `gemma`, by the same rule as `families.for_model`."""
    model_type = str(getattr(text_config(cfg), "model_type", ""))
    if model_type == "opt":
        return "opt"
    if model_type.startswith("gemma"):
        return "gemma"
    return "llama"


def block_linears(cfg: Any) -> list[LinearShape]:
    """The quantized Linears of one decoder block, in the family's target order."""
    cfg = text_config(cfg)
    hidden = cfg.hidden_size
    heads = cfg.num_attention_heads
    kv_heads = getattr(cfg, "num_key_value_heads", None) or heads
    head_dim = getattr(cfg, "head_dim", None) or hidden // heads
    if family_name(cfg) == "opt":
        bias = bool(getattr(cfg, "enable_bias", True))
        ffn = cfg.ffn_dim
        return [
            LinearShape("self_attn.q_proj", hidden, hidden, bias),
            LinearShape("self_attn.k_proj", hidden, hidden, bias),
            LinearShape("self_attn.v_proj", hidden, hidden, bias),
            LinearShape("self_attn.out_proj", hidden, hidden, bias),
            LinearShape("fc1", ffn, hidden, bias),
            LinearShape("fc2", hidden, ffn, bias),
        ]
    attn_bias = bool(getattr(cfg, "attention_bias", False))
    mlp_bias = bool(getattr(cfg, "mlp_bias", False))
    inter = cfg.intermediate_size
    return [
        LinearShape("self_attn.q_proj", heads * head_dim, hidden, attn_bias),
        LinearShape("self_attn.k_proj", kv_heads * head_dim, hidden, attn_bias),
        LinearShape("self_attn.v_proj", kv_heads * head_dim, hidden, attn_bias),
        LinearShape("self_attn.o_proj", hidden, heads * head_dim, attn_bias),
        LinearShape("mlp.gate_proj", inter, hidden, mlp_bias),
        LinearShape("mlp.up_proj", inter, hidden, mlp_bias),
        LinearShape("mlp.down_proj", hidden, inter, mlp_bias),
    ]


def other_params(cfg: Any) -> int:
    """Parameters outside the quantized Linears: embeddings, lm_head when untied, norms,
    biases, and OPT's positional table and projections. Exact for OPT and Llama (checked
    against real modules in the tests); Gemma-2/3 carry extra norms and are close."""
    cfg = text_config(cfg)
    hidden = cfg.hidden_size
    layers = cfg.num_hidden_layers
    vocab = cfg.vocab_size
    fam = family_name(cfg)
    linears = block_linears(cfg)
    biases = sum(s.out_features for s in linears if s.bias)
    if fam == "opt":
        embed_dim = getattr(cfg, "word_embed_proj_dim", None) or hidden
        #: OPT offsets positions by 2 (padding idx + 1) and keeps a learned table.
        total = vocab * embed_dim + (cfg.max_position_embeddings + 2) * hidden
        if embed_dim != hidden:  # opt-350m: project_in / project_out, no bias
            total += 2 * hidden * embed_dim
        ln = 2 * hidden  # LayerNorm weight + bias
        total += layers * (2 * ln + biases)
        if getattr(cfg, "do_layer_norm_before", True) and not getattr(cfg, "_remove_final_layer_norm", False):
            total += ln
        return total
    total = vocab * hidden + hidden  # embeddings + final RMSNorm
    if not getattr(cfg, "tie_word_embeddings", True):
        total += vocab * hidden
    norms_per_layer = 4 if fam == "gemma" else 2  # Gemma adds pre/post feed-forward norms
    per_layer = norms_per_layer * hidden + biases
    if fam == "gemma" and str(cfg.model_type).startswith("gemma3"):
        head_dim = getattr(cfg, "head_dim", None) or hidden // cfg.num_attention_heads
        per_layer += 2 * head_dim  # q_norm, k_norm
    return total + layers * per_layer


def linear_bits(shape: LinearShape, *, algo: str, bits: int, group_size: int, sym: bool, act_order: bool) -> int:
    """Bits one quantized Linear takes in the packed format (its bias, if any, is counted
    with the other parameters)."""
    if algo == "fp":
        return shape.weights * FP_BITS
    n_groups = 1 if group_size == -1 else math.ceil(shape.in_features / group_size)
    zero_bits = SYM_ZERO_BITS if sym else (FLOAT_ZERO_BITS if algo in FLOAT_ZERO_ALGOS else bits)
    total = shape.weights * bits + shape.out_features * n_groups * (SCALE_BITS + zero_bits)
    if algo == "gptq" and act_order and group_size != -1:
        total += shape.in_features * GROUP_INDEX_BITS
    return total


def dtype_bits(dtype: Any) -> int:
    """Bits per parameter of the model dtype a row was measured in (16 unless fp32)."""
    return 32 if "32" in str(dtype) else 16


def footprint(cfg: Any, quant: dict[str, Any], *, dtype: Any = "float16", model_bytes: int | None = None) -> Footprint:
    """The footprint of `quant` (algo, bits, group_size, sym, act_order) on the model `cfg`
    describes. `model_bytes`, the loaded model's real size at `dtype`, makes the
    non-quantized remainder exact."""
    cfg = text_config(cfg)
    linears = block_linears(cfg)
    layers = cfg.num_hidden_layers
    weights = layers * sum(s.weights for s in linears)
    algo = str(quant.get("algo", "fp"))
    q_bits = layers * sum(
        linear_bits(
            s, algo=algo, bits=int(quant.get("bits", FP_BITS)), group_size=int(quant.get("group_size") or -1),
            sym=bool(quant.get("sym", False)), act_order=bool(quant.get("act_order", False)),
        )
        for s in linears
    )
    per_param = dtype_bits(dtype)
    if model_bytes:
        other = model_bytes * 8 - weights * per_param
        return Footprint(weights, q_bits, max(other, 0), exact=True)
    return Footprint(weights, q_bits, other_params(cfg) * per_param, exact=False)


def load_config(repo: str, revision: str | None = None) -> Any:
    """The Hugging Face config for `repo` (cached under the project's HF home)."""
    from ..models import loader

    return loader._config(repo, revision)


def row_footprint(row: dict[str, Any], configs: dict[tuple[str, str | None], Any]) -> Footprint | None:
    """The footprint of one result row, or None when its config cannot be loaded (an
    unknown repo, no network). `configs` caches configs across rows."""
    repo = str(row.get("loaded_from") or row.get("model") or "")
    if not repo:
        return None
    key = (repo, row.get("model_revision"))
    if key not in configs:
        try:
            configs[key] = load_config(*key)
        except Exception:  # noqa: BLE001 - a missing config costs two columns, never the table
            configs[key] = None
    cfg = configs[key]
    if cfg is None:
        return None
    quant = {k: row.get(k) for k in ("algo", "bits", "group_size", "sym", "act_order")}
    return footprint(cfg, quant, dtype=row.get("dtype", "float16"), model_bytes=row.get("model_bytes"))
