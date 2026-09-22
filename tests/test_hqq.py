"""DESIGN.md 6: the hqq adapter runs on CPU with CUDA monkeypatched away, on one Linear."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.smoke


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_hqq_dequantizes_one_linear_on_cpu(no_cuda):
    pytest.importorskip("hqq.core.quantize")
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

    torch.manual_seed(0)
    lin = torch.nn.Linear(768, 256)
    cfg = BaseQuantizeConfig(nbits=4, group_size=None, axis=1)
    h = HQQLinear(lin, quant_config=cfg, compute_dtype=torch.float32, device="cpu", del_orig=False)
    w = h.dequantize()
    assert w.shape == lin.weight.shape
    rel = ((w.float() - lin.weight.float()).norm() / lin.weight.float().norm()).item()
    assert 0.01 < rel < 0.2


def test_apply_hqq_on_opt125m_cpu(no_cuda):
    pytest.importorskip("hqq.core.quantize")
    from ptqbench.models import loader as ml
    from ptqbench.quantizers import hqq_adapter as H

    lm = ml.load("facebook/opt-125m", device=torch.device("cpu"))
    before = lm.model.model.decoder.layers[0].fc1.weight.detach().clone()
    rep = H.apply_hqq(lm.model, bits=4, group_size=128, device=torch.device("cpu"))
    after = lm.model.model.decoder.layers[0].fc1.weight.detach()
    assert rep.n_modules == 72
    assert not torch.equal(before, after)
    assert 0.01 < rep.mean_relative_error < 0.3
    assert rep.as_row_fields()["backend"].startswith("hqq==")


def test_group_size_rules():
    pytest.importorskip("hqq.core.quantize")
    from ptqbench.quantizers import hqq_adapter as H

    with pytest.raises(ValueError, match="% 8"):
        H.apply_hqq(torch.nn.Module(), bits=4, group_size=12, device=torch.device("cpu"))
