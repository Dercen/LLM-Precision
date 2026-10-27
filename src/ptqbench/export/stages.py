"""Prefill, prefill-KV and decode as three functions with an explicit, static KV cache.

`ptq export-mlir --stages` writes `prefill.mlir`, `prefill_kv.mlir` and `decode.mlir`
(and their linalg forms), following the three-module split of "An MLIR-Based
Compilation Method for Large Language Models" (arXiv 2607.15865):

    prefill(input_ids[1, S])                                  -> logits[1, S, V], k_cache, v_cache
    prefill_kv(input_ids[1, S2], k_cache, v_cache, position)  -> logits[1, S2, V], k_cache, v_cache
    decode(input_ids[1, 1], k_cache, v_cache, position)       -> logits[1, 1, V], k_cache, v_cache

The cache is two tensors of shape `[layers, 1, kv_heads, max_cache_len, head_dim]`
(`CACHE_LAYOUT`), one for keys and one for values, in the model's dtype; `position`
is a 0-d int64 tensor holding the number of tokens already in the cache, which is also
the position of the first new token. Every shape is static: `S` is `--seqlen`, `S2` is
`--prefill-kv-seqlen`, and the cache length is `--max-cache-len`. Writing past the
cache length is undefined (the function has no way to say so), so a consumer keeps
`position + new tokens <= max_cache_len`.

How the cache is made explicit
------------------------------
transformers 5.17 offers `StaticCache` (a `Cache` of `StaticLayer`s, each holding
preallocated key/value tensors it updates **in place** with `index_copy_`) and an
executorch recipe (`TorchExportableModuleWithStaticCache`) that registers those
tensors as mutated buffers of a wrapper module. Neither gives a function with the
cache as inputs and outputs, and an in-place `index_copy` lowers to
`tm_tensor.scatter`, an op torch-mlir's private dialect owns and a stock `mlir-opt`
cannot read. `FunctionalStaticLayer` therefore subclasses `StaticLayer` with an
out-of-place update: the new keys are gathered into position with `index_select`
and merged into the old cache with `torch.where` (plain elementwise ops, so linalg
only). Everything else -- the mask over the full cache length, `get_seq_length`,
`get_mask_sizes` -- is inherited unchanged, and `use_cache=True` is passed only
inside `StageFunction.forward`; the evaluation loader's `use_cache=False` is untouched.

Families: Llama-family models need nothing else. OPT derives its position ids and
mask from `past_seen_tokens` with a data-dependent `torch.ones(...)` that torch.export
refuses, so `StageFunction` passes explicit `position_ids` and an all-ones mask of the
cache length for OPT (and explicit `position_ids` for every family, which is what the
model would compute anyway). Models with sliding-window layers are refused.

Checked, not assumed: `check_stages` runs prefill, prefill_kv and `DECODE_CHECK_STEPS`
decode steps on real text through the exact `nn.Module`s that were exported, feeding
each step the true next token, and compares every logit with the eager model run once
over the whole sequence without a cache. The tolerance is the golden one: the
prefill function's measured fp32-vs-float64 distance times the safety factor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from transformers.cache_utils import Cache, StaticLayer

from ..models import families

#: The three stage functions, in the order they run.
STAGES = ("prefill", "prefill_kv", "decode")
#: Default static cache length: enough for the default 128-token prompt plus the
#: 16-token prefill-KV example plus the decode check, with room to spare, and small
#: enough that the cache tensors stay a few MB for SmolLM2 (30 x 3 x 256 x 64 floats).
DEFAULT_MAX_CACHE_LEN = 256
#: Default length of the second prompt chunk `prefill_kv` is exported with.
DEFAULT_PREFILL_KV_SEQLEN = 16
#: Decode steps the round-trip check runs after prefill and prefill_kv; each feeds
#: the real next token of the text (teacher forcing), so every step is realistic.
DECODE_CHECK_STEPS = 4
#: A decode step always takes exactly one token.
DECODE_SEQLEN = 1

#: Names of the stage functions' inputs and outputs in the golden index and the hand-off spec.
INPUT_IDS = "input_ids"
K_CACHE = "k_cache"
V_CACHE = "v_cache"
POSITION = "position"
LOGITS = "logits"
STAGE_OUTPUTS = (LOGITS, K_CACHE, V_CACHE)
#: What the five cache dimensions mean, written into export.json.
CACHE_LAYOUT = "[layers, batch=1, kv_heads, max_cache_len, head_dim]"


# ---- the functional static cache ----------------------------------------------------------


class FunctionalStaticLayer(StaticLayer):
    """A StaticLayer whose update is out of place, so the cache can be a function input
    and output. `keys`/`values` are rebound to new tensors instead of written in place."""

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        n_new = key_states.shape[-2]
        slot = torch.arange(self.max_cache_len, device=key_states.device)
        # For every cache slot, which of the new tokens lands there (negative or >= n_new: none).
        relative = slot - self.cumulative_length
        valid = ((relative >= 0) & (relative < n_new)).view(1, 1, -1, 1)
        gathered = relative.clamp(0, n_new - 1)
        self.keys = torch.where(valid, key_states.index_select(2, gathered), self.keys)
        self.values = torch.where(valid, value_states.index_select(2, gathered), self.values)
        self.cumulative_length = self.cumulative_length + n_new
        return self.keys, self.values


def cache_shape(config, max_cache_len: int) -> tuple[int, int, int, int, int]:
    """`[layers, 1, kv_heads, max_cache_len, head_dim]` from the model config."""
    cfg = config.get_text_config(decoder=True) if hasattr(config, "get_text_config") else config
    if getattr(cfg, "sliding_window", None) is not None or any(
        t != "full_attention" for t in (getattr(cfg, "layer_types", None) or [])
    ):
        raise NotImplementedError("stage export supports full-attention layers only (no sliding window)")
    heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return (cfg.num_hidden_layers, 1, heads, max_cache_len, head_dim)


def functional_cache(k_cache: torch.Tensor, v_cache: torch.Tensor, position: torch.Tensor) -> Cache:
    """A `Cache` of FunctionalStaticLayers initialised from the function's inputs."""
    layers = []
    max_cache_len = int(k_cache.shape[3])
    for i in range(k_cache.shape[0]):
        layer = FunctionalStaticLayer(max_cache_len)
        layer.keys, layer.values = k_cache[i], v_cache[i]
        layer.dtype, layer.device = k_cache.dtype, k_cache.device
        layer.batch_size, layer.num_heads = int(k_cache.shape[1]), int(k_cache.shape[2])
        layer.k_head_dim = layer.v_head_dim = int(k_cache.shape[-1])
        layer.cumulative_length = position
        layer.is_initialized = True
        layers.append(layer)
    return Cache(layers=layers)


# ---- the stage functions -------------------------------------------------------------------


class StageFunction(nn.Module):
    """One stage of the model with the cache as explicit inputs and outputs.

    `takes_cache=False` is `prefill`: the cache starts empty (zeros, position 0) and
    only `input_ids` is an argument. `takes_cache=True` is `prefill_kv` and `decode`,
    which are the same function at different input lengths.
    """

    def __init__(self, model: nn.Module, family: families.Family, max_cache_len: int, *, takes_cache: bool):
        super().__init__()
        self.model = model
        self.family_name = family.name
        self.max_cache_len = max_cache_len
        self.takes_cache = takes_cache
        self.shape = cache_shape(model.config, max_cache_len)

    def _model_dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def forward(self, input_ids: torch.Tensor, k_cache=None, v_cache=None, position=None):
        if not self.takes_cache:
            k_cache = torch.zeros(self.shape, dtype=self._model_dtype(), device=input_ids.device)
            v_cache = torch.zeros(self.shape, dtype=self._model_dtype(), device=input_ids.device)
            position = torch.zeros((), dtype=torch.int64, device=input_ids.device)
        cache = functional_cache(k_cache, v_cache, position)
        seqlen = input_ids.shape[1]
        kwargs: dict[str, Any] = {
            "position_ids": (torch.arange(seqlen, device=input_ids.device) + position).unsqueeze(0),
            "attention_mask": None,
        }
        if self.family_name == "opt":
            # OPT builds its mask and positions from past_seen_tokens with a data-dependent
            # torch.ones that torch.export refuses; an explicit all-ones mask of the cache
            # length makes every shape static and the causal mask handles the rest.
            kwargs["attention_mask"] = torch.ones((1, self.max_cache_len), dtype=torch.int64, device=input_ids.device)
        out = self.model(input_ids=input_ids, past_key_values=cache, use_cache=True, **kwargs)
        k_new = torch.stack([layer.keys for layer in cache.layers])
        v_new = torch.stack([layer.values for layer in cache.layers])
        return out.logits, k_new, v_new


def build_stages(model: nn.Module, family: families.Family, max_cache_len: int) -> dict[str, StageFunction]:
    """The three stage modules sharing `model`'s weights."""
    with_cache = StageFunction(model, family, max_cache_len, takes_cache=True)
    return {
        "prefill": StageFunction(model, family, max_cache_len, takes_cache=False),
        "prefill_kv": with_cache,
        "decode": with_cache,
    }


# ---- example inputs, the round trip, and the description -------------------------------------


@dataclass
class StageRun:
    """The arguments each stage was traced and checked with, from one real text window."""

    seqlen: int  # S, the prefill prompt
    prefill_kv_seqlen: int  # S2
    decode_steps: int
    ids: torch.Tensor  # [1, S + S2 + decode_steps] real token ids
    #: Example args per stage, in argument order; the caches are real (from the previous stage).
    args: dict[str, tuple[torch.Tensor, ...]] = field(default_factory=dict)
    #: The stage outputs for those args.
    outputs: dict[str, tuple[torch.Tensor, ...]] = field(default_factory=dict)


@torch.no_grad()
def run_stages(stages: dict[str, StageFunction], ids: torch.Tensor, seqlen: int, prefill_kv_seqlen: int) -> StageRun:
    """Run prefill, then prefill_kv, then the first decode step, keeping every argument."""
    s, s2 = seqlen, prefill_kv_seqlen
    if ids.shape[1] < s + s2 + DECODE_SEQLEN:
        raise ValueError(f"need at least {s + s2 + DECODE_SEQLEN} tokens for the stage examples, got {ids.shape[1]}")
    run = StageRun(s, s2, DECODE_CHECK_STEPS, ids)
    run.args["prefill"] = (ids[:, :s],)
    run.outputs["prefill"] = tuple(stages["prefill"](*run.args["prefill"]))
    _, k, v = run.outputs["prefill"]
    run.args["prefill_kv"] = (ids[:, s : s + s2], k, v, torch.tensor(s, dtype=torch.int64))
    run.outputs["prefill_kv"] = tuple(stages["prefill_kv"](*run.args["prefill_kv"]))
    _, k, v = run.outputs["prefill_kv"]
    run.args["decode"] = (ids[:, s + s2 : s + s2 + DECODE_SEQLEN], k, v, torch.tensor(s + s2, dtype=torch.int64))
    run.outputs["decode"] = tuple(stages["decode"](*run.args["decode"]))
    return run


@torch.no_grad()
def check_stages(
    model: nn.Module, stages: dict[str, StageFunction], run: StageRun, tolerance: dict[str, float] | None
) -> dict[str, Any]:
    """prefill -> prefill_kv -> `decode_steps` decode steps against the eager model.

    The eager reference is the quantized model run once over the whole token window
    with no cache (the way perplexity is measured). Every stage's logits are compared
    with the matching positions; `tolerance` is the golden atol/rel_fro_tol of the
    prefill logits, or None when it could not be measured (then nothing fails).
    """
    s, s2, steps = run.seqlen, run.prefill_kv_seqlen, run.decode_steps
    ids = run.ids
    if ids.shape[1] < s + s2 + steps:
        raise ValueError(f"need {s + s2 + steps} tokens for the check, got {ids.shape[1]}")
    full = model(input_ids=ids[:, : s + s2 + steps]).logits
    comparisons: list[dict[str, Any]] = []

    def record(stage: str, step: int | None, got: torch.Tensor, want: torch.Tensor) -> None:
        g, w = got.double(), want.double()
        diff = (g - w).abs()
        entry = {
            "stage": stage, "step": step, "positions": list(want.shape[1:2]),
            "max_abs_diff": float(diff.max()), "rel_fro_diff": float(diff.norm() / w.norm()) if float(w.norm()) else 0.0,
            "argmax_agreement": float((g.argmax(-1) == w.argmax(-1)).double().mean()),
        }
        if tolerance is not None:
            entry["within_tolerance"] = entry["max_abs_diff"] <= tolerance["atol"] and entry["rel_fro_diff"] <= tolerance["rel_fro_tol"]
        comparisons.append(entry)

    logits, k, v = run.outputs["prefill"]
    record("prefill", None, logits, full[:, :s])
    logits, k, v = stages["prefill_kv"](*run.args["prefill_kv"])
    record("prefill_kv", None, logits, full[:, s : s + s2])
    for t in range(steps):
        position = s + s2 + t
        logits, k, v = stages["decode"](ids[:, position : position + 1], k, v, torch.tensor(position, dtype=torch.int64))
        record("decode", t, logits, full[:, position : position + 1])
    worst = max(comparisons, key=lambda c: c["max_abs_diff"])
    return {
        "reference": "the quantized model run once over the whole window with use_cache=False",
        "decode_steps": steps,
        "tolerance": tolerance,
        "max_abs_diff": worst["max_abs_diff"],
        "worst": f"{worst['stage']}" + (f" step {worst['step']}" if worst["step"] is not None else ""),
        "all_within_tolerance": all(c.get("within_tolerance", True) for c in comparisons) if tolerance else None,
        "comparisons": comparisons,
    }


def stage_input_names(stage: str) -> list[str]:
    return [INPUT_IDS] if stage == "prefill" else [INPUT_IDS, K_CACHE, V_CACHE, POSITION]


def describe_signature(stage: str, args: tuple[torch.Tensor, ...], outputs: tuple[torch.Tensor, ...]) -> dict[str, Any]:
    """Names, shapes and dtypes of a stage's inputs and outputs, for export.json."""

    def entry(name, t):
        return {"name": name, "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")}

    return {
        "inputs": [entry(n, t) for n, t in zip(stage_input_names(stage), args)],
        "outputs": [entry(n, t) for n, t in zip(STAGE_OUTPUTS, outputs)],
    }
