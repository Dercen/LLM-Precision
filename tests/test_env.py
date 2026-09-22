"""Paths, device policy and the numerics lockdown of PLAN.md 5a."""

from __future__ import annotations

import pytest
import torch

from ptqbench import device as D
from ptqbench import paths


@pytest.mark.smoke
def test_repo_root_has_pyproject():
    assert (paths.repo_root() / "pyproject.toml").is_file()


@pytest.mark.smoke
def test_describe_lists_every_location():
    described = paths.describe()
    for key in ("repo_root", "hf_home", "ptq_cache_dir", "quant_cache", "results_dir"):
        assert described[key]


@pytest.mark.smoke
def test_lock_numerics_turns_tf32_off():
    """On torch 2.14 cudnn.conv/rnn ship as 'tf32'; the lock must clear every knob."""
    state = D.lock_numerics()
    assert not D.tf32_is_live(), D.read_fp32_precision()
    assert state.tf32 is False
    assert state.float32_matmul_precision == "highest"
    assert torch.get_float32_matmul_precision() == "highest"


@pytest.mark.smoke
def test_dtype_policy():
    cpu = torch.device("cpu")
    assert D.dtype_for("facebook/opt-125m", cpu) is torch.float32
    if torch.cuda.is_available():
        gpu = D.resolve("auto")
        assert D.dtype_for("facebook/opt-125m", gpu) is torch.float16
        assert D.dtype_for("meta-llama/Llama-2-7b-hf", gpu) is torch.float16
        assert D.dtype_for("meta-llama/Llama-3.1-8B", gpu) is torch.bfloat16


@pytest.mark.gpu
def test_stream_boundary_matches_plan():
    """PLAN.md 2a: opt-2.7b resident, opt-6.7b streamed, on an 8 GB card."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    dev = D.resolve("auto")
    # "capacity" basis: a question about the card, not about what is loaded right now.
    # Unambiguous cases only -- opt-1.3b resident, opt-6.7b and Llama-2-7b streamed.
    assert D.should_stream(int(2.63 * 1024**3), dev, basis="capacity") is False
    assert D.should_stream(int(13.3 * 1024**3), dev, basis="capacity") is True
    assert D.should_stream(int(13.5 * 1024**3), dev, basis="capacity") is True


@pytest.mark.gpu
def test_opt_2_7b_is_resident_by_measurement():
    """PLAN.md 2a's marginal case, resolved.

    The plan's 5.3 GB figure for opt-2.7b was rounded; config.json gives 2.65B params
    = 4.93 GB fp16, and 1.3 x 4.93 = 6.41 GB sits under the 6.82 GB capacity, so the
    rule says resident. Measured 2026-09-21: resident fp16 eval peaks at 5.07 GB with
    ~2.5 GB to spare. opt-6.7b at 13.3 GB remains the first streamed model.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    dev = D.resolve("auto")
    opt_2_7b = int(4.93 * 1024**3)
    assert D.should_stream(opt_2_7b, dev, basis="capacity") is False
    assert D.vram_capacity_bytes(dev) > 1.3 * opt_2_7b
