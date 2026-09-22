"""Round-to-nearest quantization. DESIGN.md 6.

Data-free: apply the reference grid to every target Linear in place. This is the
baseline every other algorithm is measured against, and the RTN4 number is the M2
hard gate (opt-125m wikitext2, 37.28 +/- 1%).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from ..models import families
from . import fakequant


@dataclass
class QuantReport:
    algo: str
    bits: int
    group_size: int
    sym: bool
    n_modules: int
    quant_seconds: float
    mean_relative_error: float

    def as_row_fields(self) -> dict[str, Any]:
        return {
            "algo": self.algo,
            "bits": self.bits,
            "group_size": self.group_size,
            "sym": self.sym,
            "n_quantized_modules": self.n_modules,
            "quant_seconds": round(self.quant_seconds, 3),
            "mean_relative_error": round(self.mean_relative_error, 6),
        }


@torch.no_grad()
def apply_rtn(
    model: Any,
    *,
    bits: int,
    group_size: int = -1,
    sym: bool = False,
    family: families.Family | None = None,
) -> QuantReport:
    """Fake-quantize every target Linear in place and report the error."""
    started = time.perf_counter()
    targets = families.target_modules(model, family)
    if not targets:
        raise RuntimeError("no target Linear modules found; check the family map")

    errors: list[float] = []
    for module in targets.values():
        original = module.weight.data
        quantized = fakequant.quantize_weight(
            original, bits=bits, sym=sym, group_size=group_size
        )
        errors.append(fakequant.quantization_error(original, quantized))
        module.weight.data.copy_(quantized)

    return QuantReport(
        algo="rtn",
        bits=bits,
        group_size=group_size,
        sym=sym,
        n_modules=len(targets),
        quant_seconds=time.perf_counter() - started,
        mean_relative_error=sum(errors) / len(errors),
    )
