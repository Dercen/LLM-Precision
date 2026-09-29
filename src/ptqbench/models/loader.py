"""Model and tokenizer loading. DESIGN.md 5.1 and 8.

Explicit dtype, SDPA attention with an eager fallback, use_cache off, and the commit
hash of what was actually loaded. `estimate_model_bytes` works from config.json alone
so the resident-vs-streamed decision can be made before any weights are read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .. import device as D
from .. import paths

_DTYPES = {
    "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float32": torch.float32, "fp32": torch.float32, "float": torch.float32,
}


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    dtype: torch.dtype
    revision: str | None
    tokenizer_class: str
    attn_implementation: str
    model_bytes: int


def parse_dtype(name: str | None) -> torch.dtype | None:
    """'float16', 'bf16', 'torch.bfloat16' -> a torch dtype; None or 'auto' -> None."""
    if name is None:
        return None
    key = str(name).replace("torch.", "").lower()
    if key == "auto":
        return None
    if key not in _DTYPES:
        raise ValueError(f"unknown dtype {name!r}")
    return _DTYPES[key]


def _config(repo: str, revision: str | None = None):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(repo, cache_dir=str(paths.hf_home()), revision=revision)
    return getattr(cfg, "text_config", None) or cfg


def estimate_model_bytes(repo: str, dtype: torch.dtype, revision: str | None = None) -> int:
    """Weight bytes from config.json: embeddings, attention, MLP, and an untied lm_head.

    Biases, norms and positional tables are left out; good to a few percent.
    """
    cfg = _config(repo, revision)
    hidden = cfg.hidden_size
    vocab = cfg.vocab_size
    layers = cfg.num_hidden_layers
    heads = cfg.num_attention_heads
    kv_heads = getattr(cfg, "num_key_value_heads", None) or heads
    head_dim = getattr(cfg, "head_dim", None) or hidden // heads
    ffn = getattr(cfg, "intermediate_size", None) or getattr(cfg, "ffn_dim", None) or 4 * hidden
    gated_mlp = getattr(cfg, "model_type", "") != "opt"  # gate/up/down vs fc1/fc2

    attn = 2 * hidden * heads * head_dim + 2 * hidden * kv_heads * head_dim
    mlp = (3 if gated_mlp else 2) * hidden * ffn
    embed_dim = getattr(cfg, "word_embed_proj_dim", None) or hidden
    params = layers * (attn + mlp) + vocab * embed_dim
    if not getattr(cfg, "tie_word_embeddings", True):
        params += vocab * embed_dim
    return int(params * torch.tensor([], dtype=dtype).element_size())


def load(
    repo: str,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
    to_device: bool = True,
    revision: str | None = None,
    attn_implementation: str = "sdpa",
) -> LoadedModel:
    """Load weights into host RAM at `dtype`, then onto `device` unless streaming."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = dtype or D.dtype_for(repo, device)
    cache = str(paths.hf_home())
    tokenizer = AutoTokenizer.from_pretrained(repo, cache_dir=cache, revision=revision)
    kwargs = {"dtype": dtype, "cache_dir": cache, "revision": revision}
    try:
        model = AutoModelForCausalLM.from_pretrained(repo, attn_implementation=attn_implementation, **kwargs)
    except (ValueError, ImportError):
        attn_implementation = "eager"
        model = AutoModelForCausalLM.from_pretrained(repo, attn_implementation=attn_implementation, **kwargs)
    model.eval()
    model.config.use_cache = False
    if to_device:
        model.to(device)
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        dtype=dtype,
        revision=getattr(model.config, "_commit_hash", None) or revision,
        tokenizer_class=type(tokenizer).__name__,
        attn_implementation=getattr(model.config, "_attn_implementation", attn_implementation),
        model_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
    )
