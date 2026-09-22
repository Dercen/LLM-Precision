"""DESIGN.md 6: our grid must equal IST-DASLab/gptq's Quantizer, quirks included."""

from __future__ import annotations

import pytest
import torch

from ptqbench.quantizers import fakequant as fq

from .reference.gptq_quant import Quantizer


def _reference(weight: torch.Tensor, bits: int, sym: bool):
    q = Quantizer()
    q.configure(bits, perchannel=True, sym=sym, mse=False)
    q.find_params(weight, weight=True)
    return q.scale, q.zero, q.quantize(weight)


@pytest.mark.smoke
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("sym", [False, True])
def test_matches_reference_quantizer(bits, sym):
    torch.manual_seed(0)
    w = torch.randn(64, 256, dtype=torch.float32)

    ref_scale, ref_zero, ref_what = _reference(w, bits, sym)
    grid = fq.find_params(w, bits=bits, sym=sym, group_size=-1)
    what = fq.quantize_weight(w, bits=bits, sym=sym, group_size=-1)

    torch.testing.assert_close(grid.scale, ref_scale, rtol=0, atol=0)
    torch.testing.assert_close(grid.zero, ref_zero, rtol=0, atol=0)
    torch.testing.assert_close(what, ref_what, rtol=0, atol=0)


@pytest.mark.smoke
def test_dead_rows_map_to_unit_range():
    """An all-zero row must not produce scale == 0 (the reference's dead-row rule)."""
    w = torch.randn(8, 32)
    w[3] = 0.0
    grid = fq.find_params(w, bits=4)
    assert torch.all(grid.scale > 0)
    ref_scale, _, _ = _reference(w, 4, False)
    torch.testing.assert_close(grid.scale, ref_scale, rtol=0, atol=0)


@pytest.mark.smoke
def test_range_includes_zero():
    """All-positive weights must still admit zero, or the zero-point is wrong."""
    w = torch.rand(8, 32) + 1.0  # strictly positive
    grid = fq.find_params(w, bits=4)
    ref_scale, ref_zero, _ = _reference(w, 4, False)
    torch.testing.assert_close(grid.scale, ref_scale, rtol=0, atol=0)
    torch.testing.assert_close(grid.zero, ref_zero, rtol=0, atol=0)


@pytest.mark.smoke
@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_grouped_equals_reference_applied_per_group(group_size):
    """Grouped quantization must equal running the reference on each column slice."""
    torch.manual_seed(1)
    w = torch.randn(16, 256)
    got = fq.quantize_weight(w, bits=4, group_size=group_size)

    pieces = []
    for start in range(0, w.shape[1], group_size):
        chunk = w[:, start : start + group_size].contiguous()
        _, _, ref_chunk = _reference(chunk, 4, False)
        pieces.append(ref_chunk)
    torch.testing.assert_close(got, torch.cat(pieces, dim=1), rtol=0, atol=0)


@pytest.mark.smoke
def test_more_bits_is_less_error():
    torch.manual_seed(2)
    w = torch.randn(32, 128)
    errors = [
        fq.quantization_error(w, fq.quantize_weight(w, bits=b)) for b in (2, 3, 4, 8)
    ]
    assert errors == sorted(errors, reverse=True), errors


@pytest.mark.smoke
def test_smaller_groups_is_less_error():
    torch.manual_seed(3)
    w = torch.randn(32, 256)
    per_row = fq.quantization_error(w, fq.quantize_weight(w, bits=3, group_size=-1))
    g128 = fq.quantization_error(w, fq.quantize_weight(w, bits=3, group_size=128))
    g64 = fq.quantization_error(w, fq.quantize_weight(w, bits=3, group_size=64))
    assert per_row > g128 > g64


@pytest.mark.smoke
def test_dtype_is_preserved_and_grid_runs_in_fp32():
    """DESIGN.md 6: fp16 in, fp16 out, but the grid itself computed in fp32."""
    torch.manual_seed(4)
    w16 = torch.randn(32, 128, dtype=torch.float16)
    out = fq.quantize_weight(w16, bits=4)
    assert out.dtype is torch.float16
    grid = fq.find_params(w16, bits=4)
    assert grid.scale.dtype is torch.float32


@pytest.mark.smoke
def test_rejects_indivisible_group_size():
    w = torch.randn(8, 100)
    with pytest.raises(ValueError, match="not divisible"):
        fq.find_params(w, bits=4, group_size=32)


@pytest.mark.gpu
@pytest.mark.parametrize("bits,group_size", [(4, 128), (3, -1), (2, 64), (8, 128)])
def test_cuda_matches_cpu_bitwise(bits, group_size):
    """Quantization must be device-independent, or one quant_key names two weights.

    CUDA turns fp32 tensor-by-scalar division into a multiply by the reciprocal,
    which is 1 ULP short of correctly rounded; fakequant does the scale and
    zero-point divisions in float64 to keep both backends on the same answer.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(5)
    w = torch.randn(64, 256, dtype=torch.float16)

    grid_cpu = fq.find_params(w, bits=bits, group_size=group_size)
    grid_gpu = fq.find_params(w.cuda(), bits=bits, group_size=group_size)
    torch.testing.assert_close(grid_cpu.scale, grid_gpu.scale.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(grid_cpu.zero, grid_gpu.zero.cpu(), rtol=0, atol=0)

    cpu = fq.quantize_weight(w, bits=bits, group_size=group_size)
    gpu = fq.quantize_weight(w.cuda(), bits=bits, group_size=group_size).cpu()
    torch.testing.assert_close(cpu, gpu, rtol=0, atol=0)
