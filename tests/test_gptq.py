"""PLAN.md 6 and 10 (M3): the GPTQ port reproduces the paper and is deterministic.

Measured 2026-09-21 on opt-125m, C4 128x2048 seed 0: GPTQ4 per-row 31.5475 (paper
31.12, +1.37%), GPTQ3 per-row 53.0273 (paper 53.85, -1.53%). Two independent runs,
and a resident run against an offloaded one, agree on all 84,934,656 weights exactly.
"""

from __future__ import annotations

import pytest
import torch

from ptqbench import device as D
from ptqbench.data import calibration as C
from ptqbench.eval.run_eval import run_single_eval
from ptqbench.models import families
from ptqbench.models import loader as ml
from ptqbench.quantizers import gptq as G

pytestmark = pytest.mark.gpu

MODEL = "facebook/opt-125m"
CALIB = C.CalibSpec("c4", 128, 2048, 0)


def _need_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


def _ppl(**kw) -> float:
    _need_cuda()
    return run_single_eval(model_id=MODEL, write=False, calib=CALIB, **kw)["ppl"]


@pytest.mark.parametrize(
    "bits,paper,tol_pct",
    [(4, 31.12, 3.0), (3, 53.85, 5.0)],
)
def test_gptq_matches_published(bits, paper, tol_pct):
    got = _ppl(dataset_key="wikitext2", algo="gptq", bits=bits, group_size=-1)
    delta = (got - paper) / paper * 100
    assert abs(delta) <= tol_pct, f"GPTQ{bits}: {got:.4f} vs paper {paper} ({delta:+.2f}%)"


def test_gptq_beats_rtn_at_every_width():
    for bits in (4, 3):
        rtn = _ppl(dataset_key="wikitext2", algo="rtn", bits=bits, max_windows=40)
        gptq = _ppl(dataset_key="wikitext2", algo="gptq", bits=bits, max_windows=40)
        assert gptq < rtn, f"{bits}-bit: gptq {gptq} >= rtn {rtn}"


def _quantized_weights(offload: bool):
    dev = D.resolve("auto")
    lm = ml.load(MODEL, device=dev, to_device=not offload)
    windows = C.build(lm.tokenizer, CALIB)
    report = G.apply_gptq(
        lm.model, windows, spec=G.GPTQSpec(bits=4), device=dev, offload=offload, progress=False
    )
    weights = {n: m.weight.detach().cpu().clone() for n, m in families.target_modules(lm.model).items()}
    del lm
    torch.cuda.empty_cache()
    return weights, report


def test_gptq_is_deterministic():
    """PLAN.md 5a for the quantizer: the same inputs must give the same weights."""
    _need_cuda()
    D.lock_numerics()
    a, ra = _quantized_weights(offload=False)
    b, rb = _quantized_weights(offload=False)
    assert ra.total_loss == rb.total_loss
    for name in a:
        assert torch.equal(a[name], b[name]), f"{name} differs between identical runs"


def test_offloaded_gptq_equals_resident_gptq():
    """The streamed tier's GPTQ must be the resident computation, not an approximation."""
    _need_cuda()
    D.lock_numerics()
    resident, rr = _quantized_weights(offload=False)
    offloaded, ro = _quantized_weights(offload=True)
    assert rr.eval_mode == "resident" and ro.eval_mode == "streamed"
    assert ro.total_loss == rr.total_loss
    for name in resident:
        assert torch.equal(resident[name], offloaded[name]), f"{name} differs under offload"


def test_true_sequential_runs_and_is_close():
    """Not the paper's OPT setting, but the Llama path uses it; it must work and be sane."""
    plain = _ppl(dataset_key="wikitext2", algo="gptq", bits=4, max_windows=40)
    seq = _ppl(dataset_key="wikitext2", algo="gptq", bits=4, true_sequential=True, max_windows=40)
    assert abs(seq - plain) / plain < 0.05, (seq, plain)


def test_grouped_gptq_beats_per_row():
    per_row = _ppl(dataset_key="wikitext2", algo="gptq", bits=3, group_size=-1, max_windows=40)
    g128 = _ppl(dataset_key="wikitext2", algo="gptq", bits=3, group_size=128, max_windows=40)
    assert g128 < per_row, (g128, per_row)


# ---- Llama family ----------------------------------------------------------------
# Measured 2026-09-21 on SmolLM2-135M, wikitext2, C4 128x2048 seed 0, 4-bit per-row:
#   RTN 26.6056 | GPTQ 27.9133 | GPTQ+act_order 24.1774 | GPTQ+act_order+true_seq 23.8548
# GPTQ's per-layer output error beat RTN's in 28/28 layers, so the port is right; the
# perplexity loss is how fixed-order local errors compose through 30 outlier-heavy
# layers. act_order (quantize the largest-Hessian-diagonal columns first) is the fix,
# and is the default for the Llama family in every experiment config.

SMOL = "HuggingFaceTB/SmolLM2-135M"


def _smol(**kw) -> float:
    _need_cuda()
    return run_single_eval(model_id=SMOL, dataset_key="wikitext2", write=False, calib=CALIB, **kw)["ppl"]


def test_llama_family_gptq_needs_act_order():
    """The M3 Llama gate: GPTQ4 must beat RTN4 -- and on this family only act_order does."""
    rtn = _smol(algo="rtn", bits=4, max_windows=60)
    plain = _smol(algo="gptq", bits=4, max_windows=60)
    ordered = _smol(algo="gptq", bits=4, act_order=True, max_windows=60)
    assert ordered < rtn, f"act_order GPTQ {ordered} must beat RTN {rtn}"
    # Pinned so a change that quietly makes plain GPTQ better here is noticed, not assumed.
    assert plain > ordered, f"plain GPTQ {plain} unexpectedly beat act_order {ordered}"


def test_group_size_must_divide_in_features():
    """SmolLM2's hidden size is 576 = 4.5 x 128: g128 cannot tile it and must refuse loudly.

    A silent partial trailing group would corrupt a paper-comparable row, so the
    runner marks such cells skipped rather than bending the grid (PLAN.md 9).
    """
    with pytest.raises(ValueError, match="not divisible"):
        _smol(algo="rtn", bits=4, group_size=128, max_windows=2)
