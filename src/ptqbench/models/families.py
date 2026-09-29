"""Per-architecture module maps. DESIGN.md 4, 6.

A Family names where the decoder blocks live, which Linears inside a block are
quantized, the order GPTQ's true-sequential mode visits them, and the modules that
run after the last block (final norm, OPT's project_out). Everything that walks a
model -- the quantizers, block streaming, the quant cache -- goes through here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn


def _get(root: Any, path: str) -> Any:
    node = root
    for part in path.split("."):
        node = getattr(node, part, None)
        if node is None:
            return None
    return node


@dataclass(frozen=True)
class Family:
    name: str
    blocks_path: str
    targets: tuple[str, ...]
    sequential_groups: tuple[tuple[str, ...], ...]
    post_block_paths: tuple[str, ...]

    def block_list(self, model: nn.Module) -> list[nn.Module]:
        blocks = _get(model, self.blocks_path)
        if blocks is None:
            raise RuntimeError(f"{type(model).__name__} has no {self.blocks_path!r}; wrong family {self.name!r}?")
        return list(blocks)

    def post_block_modules(self, model: nn.Module) -> list[nn.Module]:
        """Modules applied after the last block and before lm_head, in order; absent ones skipped."""
        return [m for p in self.post_block_paths if (m := _get(model, p)) is not None]


_LLAMA_TARGETS = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)
_LLAMA_SEQUENTIAL = (
    ("self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"),
    ("self_attn.o_proj",),
    ("mlp.up_proj", "mlp.gate_proj"),
    ("mlp.down_proj",),
)

OPT = Family(
    name="opt",
    blocks_path="model.decoder.layers",
    targets=("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj", "fc1", "fc2"),
    sequential_groups=(
        ("self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"),
        ("self_attn.out_proj",),
        ("fc1",),
        ("fc2",),
    ),
    # OPT-350m is post-LN (no final_layer_norm) and projects 1024 -> 512 before lm_head.
    post_block_paths=("model.decoder.final_layer_norm", "model.decoder.project_out"),
)

LLAMA = Family(
    name="llama",
    blocks_path="model.layers",
    targets=_LLAMA_TARGETS,
    sequential_groups=_LLAMA_SEQUENTIAL,
    post_block_paths=("model.norm",),
)

# Same module names as Llama, but Gemma-2/3 blocks carry extra pre/post-feedforward
# norms, so AWQ-lite's Llama scale-group map would fold into the wrong norm. A
# separate name makes awq_lite skip it with a recorded reason instead.
GEMMA = Family(
    name="gemma",
    blocks_path="model.layers",
    targets=_LLAMA_TARGETS,
    sequential_groups=_LLAMA_SEQUENTIAL,
    post_block_paths=("model.norm",),
)

FAMILIES = {f.name: f for f in (OPT, LLAMA, GEMMA)}
_BY_MODEL_TYPE = {"opt": OPT, "llama": LLAMA, "mistral": LLAMA, "qwen2": LLAMA}


def for_model(model: nn.Module) -> Family:
    model_type = getattr(model.config, "model_type", "")
    if model_type in _BY_MODEL_TYPE:
        return _BY_MODEL_TYPE[model_type]
    if model_type.startswith("gemma"):
        return GEMMA
    raise ValueError(f"no family map for model_type {model_type!r}")


def block_targets(block: nn.Module, family: Family) -> dict[str, nn.Linear]:
    """The quantized Linears of one block, by name relative to the block."""
    out: dict[str, nn.Linear] = {}
    for name in family.targets:
        mod = _get(block, name)
        if isinstance(mod, nn.Linear):
            out[name] = mod
    return out


def target_modules(model: nn.Module, family: Family | None = None) -> dict[str, nn.Linear]:
    """Every quantized Linear in the model, by fully qualified name. lm_head is never a target."""
    fam = family or for_model(model)
    out: dict[str, nn.Linear] = {}
    for i, block in enumerate(fam.block_list(model)):
        for name, mod in block_targets(block, fam).items():
            out[f"{fam.blocks_path}.{i}.{name}"] = mod
    return out
