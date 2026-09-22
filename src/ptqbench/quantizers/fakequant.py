"""The reference quantization grid. DESIGN.md 6.

Weights are quantized to integers and immediately dequantized back into the original
nn.Linear ("fake quantization"), which is how the GPTQ, AWQ and OmniQuant papers
measured perplexity. No custom kernels are involved.

The grid reproduces IST-DASLab/gptq's `Quantizer.find_params`, including two quirks
that matter: the min/max range is clamped to include zero, and an all-zero row maps
to [-1, 1] so its scale is never zero. `tests/test_fakequant.py` holds it to the
vendored reference.

Everything runs in fp32 even when the model is fp16. On a CPU-only machine that was
automatic; on a fp16 GPU model it must be explicit, or `xmax - xmin` and the rounding
both lose precision and reference parity fails.

The scale and zero-point divisions are done in float64 and rounded once to float32.
Measured: CUDA compiles a tensor-by-scalar fp32 division into a multiply by the
reciprocal, which is 1 ULP short of correctly rounded, so `span / (2**bits - 1)`
produced a different scale on GPU than on CPU for 83 of 128 groups -- with xmin,
xmax and span all bitwise identical. That made the same weight quantize differently
depending on the device, which would let one `quant_key` name two different sets of
weights and add avoidable noise to any cross-machine comparison. float64 is
correctly rounded on both backends and costs nothing here: the divisions are over a
(rows x n_groups) vector, not the weight matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QuantGrid:
    """The scale/zero-point pair for one weight matrix, plus how it was derived."""

    scale: torch.Tensor  # (rows, n_groups)
    zero: torch.Tensor  # (rows, n_groups)
    bits: int
    sym: bool
    group_size: int

    @property
    def maxq(self) -> int:
        return 2**self.bits - 1


def _find_params(w: torch.Tensor, bits: int, sym: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row scale and zero for a 2-D fp32 tensor. Mirrors the reference exactly."""
    maxq = 2**bits - 1
    zeros = torch.zeros(w.shape[0], device=w.device, dtype=w.dtype)

    xmin = torch.minimum(w.min(dim=1).values, zeros)
    xmax = torch.maximum(w.max(dim=1).values, zeros)

    if sym:
        xmax = torch.maximum(xmin.abs(), xmax)
        negative = xmin < 0
        xmin = torch.where(negative, -xmax, xmin)

    # Dead rows: an all-zero row would otherwise produce scale == 0.
    dead = (xmin == 0) & (xmax == 0)
    xmin = torch.where(dead, torch.full_like(xmin, -1.0), xmin)
    xmax = torch.where(dead, torch.full_like(xmax, 1.0), xmax)

    # The span stays in fp32: it is what the reference computes, and it is already
    # bitwise identical across devices. Only the DIVISION is promoted -- see the
    # module docstring on CUDA's multiply-by-reciprocal shortcut. float64 has 53
    # mantissa bits against float32's 24, so rounding the float64 quotient back to
    # float32 yields the correctly-rounded float32 result, which is what the CPU
    # reference produces.
    span = xmax - xmin
    scale = _div_exact(span, maxq).unsqueeze(1)
    if sym:
        zero = torch.full_like(scale, (maxq + 1) / 2)
    else:
        zero = torch.round(_div_exact(-xmin.unsqueeze(1), scale))
    return scale, zero


def _div_exact(numerator: torch.Tensor, denominator) -> torch.Tensor:
    """fp32 division that is correctly rounded on CPU and CUDA alike."""
    out_dtype = numerator.dtype
    den = denominator.double() if torch.is_tensor(denominator) else float(denominator)
    return (numerator.double() / den).to(out_dtype)


def _apply(w: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, maxq: int) -> torch.Tensor:
    q = torch.clamp(torch.round(w / scale) + zero, 0, maxq)
    return scale * (q - zero)


def _as_groups(w: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    """Fold groups of input columns into the row axis so one kernel handles both modes."""
    rows, cols = w.shape
    n_groups = cols // group_size
    return w.reshape(rows * n_groups, group_size), n_groups


def find_params(
    weight: torch.Tensor, *, bits: int, sym: bool = False, group_size: int = -1
) -> QuantGrid:
    """Scale and zero-point for a weight matrix, per output row or per column group."""
    if weight.ndim != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(weight.shape)}")
    if bits < 2 or bits > 16:
        raise ValueError(f"bits must be in [2, 16], got {bits}")

    w = weight.detach().float()
    rows, cols = w.shape

    if group_size == -1 or group_size >= cols:
        scale, zero = _find_params(w, bits, sym)
        return QuantGrid(scale, zero, bits, sym, -1)

    if group_size <= 0:
        raise ValueError(f"group_size must be -1 or positive, got {group_size}")
    if cols % group_size != 0:
        raise ValueError(
            f"in_features {cols} is not divisible by group_size {group_size}"
        )

    grouped, n_groups = _as_groups(w, group_size)
    scale, zero = _find_params(grouped, bits, sym)
    return QuantGrid(
        scale.reshape(rows, n_groups), zero.reshape(rows, n_groups), bits, sym, group_size
    )


def quantize_weight(
    weight: torch.Tensor,
    *,
    bits: int,
    sym: bool = False,
    group_size: int = -1,
    grid: QuantGrid | None = None,
) -> torch.Tensor:
    """Fake-quantize a weight matrix: quantize to ints, dequantize, keep the dtype.

    Pass an existing `grid` to reuse a scale/zero computed elsewhere (GPTQ does this,
    since its grid is found before the column updates).
    """
    original_dtype = weight.dtype
    w = weight.detach().float()
    rows, cols = w.shape

    if grid is None:
        grid = find_params(weight, bits=bits, sym=sym, group_size=group_size)

    if grid.group_size == -1:
        out = _apply(w, grid.scale, grid.zero, grid.maxq)
    else:
        grouped, n_groups = _as_groups(w, grid.group_size)
        out = _apply(
            grouped,
            grid.scale.reshape(rows * n_groups, 1),
            grid.zero.reshape(rows * n_groups, 1),
            grid.maxq,
        ).reshape(rows, cols)

    return out.to(original_dtype)


def quantization_error(weight: torch.Tensor, quantized: torch.Tensor) -> float:
    """Relative Frobenius error, for diagnostics and sanity assertions."""
    num = torch.linalg.norm((quantized.float() - weight.float()).reshape(-1))
    den = torch.linalg.norm(weight.float().reshape(-1)).clamp_min(1e-12)
    return float(num / den)
