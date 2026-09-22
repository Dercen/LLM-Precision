"""DESIGN.md 6 and 10 (M4): AWQ-lite beats RTN at the same grid, on both families.

Measured 2026-09-21, wikitext2, C4 128x2048 seed 0: opt-125m 4-bit g128 RTN 30.4803 vs
AWQ-lite 29.3020; 3-bit g128 51.2001 vs 36.9665; SmolLM2-135M 4-bit g64 19.9461 vs 17.4723
with the 30 GQA o_proj groups skipped (v_proj has 3 kv heads, o_proj reads 9).
"""

from __future__ import annotations

import pytest
import torch

from ptqbench import device as D
from ptqbench.data import calibration as C
from ptqbench.data import datasets as ds
from ptqbench.eval import perplexity as P
from ptqbench.models import families
from ptqbench.models import loader as ml
from ptqbench.quantizers import awq_lite as A
from ptqbench.quantizers import rtn as R

pytestmark = pytest.mark.gpu
CALIB = C.CalibSpec("c4", 128, 2048, 0)


def _need_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


def _ppl_after(model_id, quantize, max_windows=40):
    dev = D.resolve("auto")
    lm = ml.load(model_id, device=dev)
    stream = ds.build("wikitext2", lm.tokenizer)
    report = quantize(lm)
    ppl = P.evaluate(lm.model, stream, device=dev, max_windows=max_windows, progress=False).ppl
    del lm
    torch.cuda.empty_cache()
    return ppl, report


@pytest.mark.parametrize("model_id,bits,gs", [
    ("facebook/opt-125m", 4, 128),
    ("facebook/opt-125m", 3, 128),
    ("HuggingFaceTB/SmolLM2-135M", 4, 64),
])
def test_awq_lite_beats_rtn(model_id, bits, gs):
    _need_cuda()
    D.lock_numerics()
    dev = D.resolve("auto")
    rtn, _ = _ppl_after(model_id, lambda lm: R.apply_rtn(lm.model, bits=bits, group_size=gs))
    awq, rep = _ppl_after(
        model_id,
        lambda lm: A.apply_awq_lite(
            lm.model, C.build(lm.tokenizer, CALIB), spec=A.AWQSpec(bits=bits, group_size=gs), device=dev
        ),
    )
    assert awq < rtn, f"{model_id} {bits}b g{gs}: awq {awq} >= rtn {rtn}"
    assert rep.n_groups_scaled > 0


def test_scale_groups_skip_inexact_folds():
    """GQA o_proj is skipped on SmolLM2; every OPT-125m group is exact."""
    _need_cuda()
    dev = D.resolve("auto")
    smol = ml.load("HuggingFaceTB/SmolLM2-135M", device=dev)
    groups, skipped = A.scale_groups(families.for_model(smol.model).block_list(smol.model)[0], families.for_model(smol.model))
    assert [g.name for g in groups] == ["qkv", "gate_up", "down_proj"]
    assert any("GQA" in s for s in skipped)
    opt = ml.load("facebook/opt-125m", device=dev)
    groups, skipped = A.scale_groups(families.for_model(opt.model).block_list(opt.model)[0], families.for_model(opt.model))
    assert [g.name for g in groups] == ["qkv", "out_proj", "fc1", "fc2"] and skipped == []


def test_scale_fold_is_exact():
    """Folding s must leave the block's function unchanged before quantization."""
    _need_cuda()
    D.lock_numerics()
    dev = D.resolve("auto")
    lm = ml.load("facebook/opt-125m", device=dev)
    fam = families.for_model(lm.model)
    block = fam.block_list(lm.model)[0]
    x = torch.randn(1, 16, lm.model.config.hidden_size, device=dev, dtype=torch.float16)
    pos = torch.arange(16, device=dev).unsqueeze(0)
    before = block(x, attention_mask=None, position_ids=pos)
    for g, _ in [A.scale_groups(block, fam)]:
        for grp in g:
            s = torch.rand(next(iter(grp.linears.values())).in_features, device=dev) + 0.5
            A.apply_scale(grp, s)
    after = block(x, attention_mask=None, position_ids=pos)
    torch.testing.assert_close(before, after, rtol=2e-2, atol=2e-2)
