"""On-disk cache of quantized target weights, keyed by quant_key. PLAN.md 9.3.

A crash between datasets must never repeat a long quantization. Entries hold the
fake-quantized weights of every target Linear in the model's own dtype (fp16/bf16 on
the GPU, fp32 in CPU tests -- storing fp16 for an fp32 model was measured to break the
round-trip in the 4th decimal) plus a metadata header naming what produced them. Data-free methods and anything under
`min_seconds` are never cached -- re-running them is cheaper than the disk.

fp16 is 4x larger than packed 4-bit integers; the streamed tier (M5) switches to
packed q/scale/zero when a 6.7B entry would otherwise cost 13 GB.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file

from .. import paths, provenance
from ..models import families

FORMAT_VERSION = 1


def entry_path(quant_key: str) -> Path:
    return paths.quant_cache_dir() / f"{quant_key}.safetensors"


def has(quant_key: str) -> bool:
    return entry_path(quant_key).is_file()


def save(model: Any, quant_key: str, meta: dict[str, Any]) -> Path:
    targets = families.target_modules(model)
    tensors = {name: mod.weight.detach().to("cpu").contiguous() for name, mod in targets.items()}
    header = {
        "format_version": str(FORMAT_VERSION),
        "dtype": str(next(iter(tensors.values())).dtype) if tensors else "none",
        "quant_key": quant_key,
        "n_modules": str(len(tensors)),
        "git_sha": provenance.git_sha(),
        "saved_at": provenance.utc_now(),
        "meta": json.dumps(meta, sort_keys=True, default=str),
    }
    path = entry_path(quant_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".safetensors.tmp")
    save_file(tensors, str(tmp), metadata=header)
    os.replace(tmp, path)
    return path


def load_into(model: Any, quant_key: str) -> dict[str, Any]:
    """Copy cached weights into the model's target Linears; returns the header meta."""
    path = entry_path(quant_key)
    targets = families.target_modules(model)
    with safe_open(str(path), framework="pt", device="cpu") as fh:
        header = fh.metadata() or {}
        keys = set(fh.keys())
        missing = set(targets) - keys
        if missing:
            raise RuntimeError(f"quant cache {path.name} lacks {len(missing)} target modules")
        for name, mod in targets.items():
            mod.weight.data.copy_(fh.get_tensor(name).to(mod.weight.device, mod.weight.dtype))
    return {"quant_cache_hit": True, "quant_cache_saved_at": header.get("saved_at")}


def entries() -> list[tuple[Path, int, float]]:
    """(path, bytes, mtime) for every cache entry, oldest first."""
    out = [(p, p.stat().st_size, p.stat().st_mtime) for p in paths.quant_cache_dir().glob("*.safetensors")]
    return sorted(out, key=lambda t: t[2])


def total_bytes() -> int:
    return sum(b for _, b, _ in entries())


def gc(*, keep_newest: int | None = None, max_gb: float | None = None) -> list[Path]:
    """Evict least-recently-used entries until both limits hold. Returns what was removed."""
    removed: list[Path] = []
    items = entries()
    if keep_newest is not None and len(items) > keep_newest:
        for p, _, _ in items[: len(items) - keep_newest]:
            p.unlink(missing_ok=True)
            removed.append(p)
        items = entries()
    if max_gb is not None:
        limit = int(max_gb * 1024**3)
        size = sum(b for _, b, _ in items)
        for p, b, _ in items:
            if size <= limit:
                break
            p.unlink(missing_ok=True)
            removed.append(p)
            size -= b
    return removed


def touch(quant_key: str) -> None:
    """Mark an entry recently used so LRU eviction keeps what is still needed."""
    p = entry_path(quant_key)
    if p.is_file():
        now = time.time()
        os.utime(p, (now, now))
