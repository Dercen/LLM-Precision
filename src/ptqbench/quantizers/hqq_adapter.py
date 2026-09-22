"""HQQ through its own quantizer, dequantized back into the model. PLAN.md 6.

Data-free, 2 to 8 bits. hqq's optimizer finds a better zero-point than min-max, so
its rows are a useful second data-free baseline next to `rtn`. Verified on CPU with
hqq 0.2.8.post1 (2026-09-21): `group_size=None` is per-row; 128 and 64 tile; nbits
8/4/3/2 all dequantize with `compute_dtype=float32` on CPU.

`HQQLinear` defaults to `compute_dtype=float16, device="cuda"`, which would raise on a
CPU-only test runner, so both are always passed explicitly. hqq asserts
`group_size % 8 == 0` and `in_features % group_size == 0`; -1 is translated to None.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from ..models import families


@dataclass
class HQQReport:
    bits: int
    group_size: int
    n_modules: int
    quant_seconds: float
    mean_relative_error: float
    hqq_version: str

    def as_row_fields(self) -> dict[str, Any]:
        return {
            "algo": "hqq",
            "backend": f"hqq=={self.hqq_version}",
            "bits": self.bits,
            "group_size": self.group_size,
            "sym": False,
            "n_quantized_modules": self.n_modules,
            "quant_seconds": round(self.quant_seconds, 3),
            "mean_relative_error": round(self.mean_relative_error, 6),
        }


def available() -> bool:
    try:
        import hqq.core.quantize  # noqa: F401
    except ImportError:
        return False
    return True


@torch.no_grad()
def apply_hqq(
    model: Any,
    *,
    bits: int,
    group_size: int = -1,
    device: torch.device,
    family: families.Family | None = None,
) -> HQQReport:
    from importlib import metadata

    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

    if bits not in (2, 3, 4, 8):
        raise ValueError(f"hqq supports 2/3/4/8 bits, got {bits}")
    gs = None if group_size == -1 else group_size
    if gs is not None and gs % 8 != 0:
        raise ValueError(f"hqq needs group_size % 8 == 0, got {group_size}")

    started = time.perf_counter()
    targets = families.target_modules(model, family)
    if not targets:
        raise RuntimeError("no target Linear modules found; check the family map")

    compute_dtype = torch.float32 if device.type == "cpu" else next(model.parameters()).dtype
    cfg = BaseQuantizeConfig(nbits=bits, group_size=gs, axis=1)

    errors: list[float] = []
    for module in targets.values():
        if gs is not None and module.in_features % gs != 0:
            raise ValueError(f"in_features {module.in_features} is not divisible by group_size {gs}")
        original = module.weight.data
        hqq_layer = HQQLinear(
            module, quant_config=cfg, compute_dtype=compute_dtype, device=str(device), del_orig=False
        )
        dequantized = hqq_layer.dequantize().to(original.dtype).to(original.device)
        errors.append(
            float((dequantized.float() - original.float()).norm() / original.float().norm().clamp_min(1e-12))
        )
        module.weight.data.copy_(dequantized)
        del hqq_layer

    return HQQReport(
        bits=bits,
        group_size=group_size,
        n_modules=len(targets),
        quant_seconds=time.perf_counter() - started,
        mean_relative_error=sum(errors) / len(errors),
        hqq_version=metadata.version("hqq"),
    )
