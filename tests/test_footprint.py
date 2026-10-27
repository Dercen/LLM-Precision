"""Footprint (weight bytes and bits per weight from config.json): the Linear shapes and
the parameter count match the real tiny models exactly, and the packed-format
arithmetic is checked by hand."""

from __future__ import annotations

import pytest

from ptqbench.analysis import footprint as F
from ptqbench.models import families
from tests.tiny_models import GROUP_SIZE, TINY_MODELS

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("family", sorted(TINY_MODELS))
def test_shapes_and_parameter_count_match_the_real_model(family):
    model = TINY_MODELS[family]()
    cfg = model.config
    shapes = {s.name: s for s in F.block_linears(cfg)}
    fam = families.for_model(model)
    assert F.family_name(cfg) == fam.name
    real = families.block_targets(fam.block_list(model)[0], fam)
    assert set(real) == set(shapes)
    for name, lin in real.items():
        assert (shapes[name].out_features, shapes[name].in_features) == tuple(lin.weight.shape), name
        assert shapes[name].bias == (lin.bias is not None), name
    n_linear = cfg.num_hidden_layers * sum(s.weights for s in shapes.values())
    assert F.other_params(cfg) + n_linear == sum(p.numel() for p in model.parameters())


def test_bits_per_weight_by_hand():
    s = F.LinearShape("x", 64, 64, False)  # 64 inputs: four groups of 16
    g16 = {"group_size": 16, "sym": False, "act_order": False}
    assert F.linear_bits(s, algo="rtn", bits=4, **g16) / s.weights == 4 + (16 + 4) / 16
    assert F.linear_bits(s, algo="rtn", bits=4, group_size=16, sym=True, act_order=False) / s.weights == 4 + 16 / 16
    assert F.linear_bits(s, algo="hqq", bits=4, **g16) / s.weights == 4 + (16 + 16) / 16
    assert F.linear_bits(s, algo="rtn", bits=4, group_size=-1, sym=False, act_order=False) / s.weights == 4 + 20 / 64
    assert F.linear_bits(s, algo="rtn", bits=3, **g16) / s.weights == 3 + (16 + 3) / 16
    plain = F.linear_bits(s, algo="gptq", bits=4, **g16)
    act = F.linear_bits(s, algo="gptq", bits=4, group_size=16, sym=False, act_order=True)
    assert act - plain == s.in_features * F.GROUP_INDEX_BITS
    assert F.linear_bits(s, algo="gptq", bits=4, group_size=-1, sym=False, act_order=True) == F.linear_bits(
        s, algo="gptq", bits=4, group_size=-1, sym=False, act_order=False
    ), "per-row act-order needs no group index"
    assert F.linear_bits(s, algo="fp", bits=16, group_size=-1, sym=False, act_order=False) == s.weights * F.FP_BITS


def test_footprint_totals_and_exact_remainder():
    model = TINY_MODELS["llama"]()
    cfg = model.config
    n_params = sum(p.numel() for p in model.parameters())
    fp = F.footprint(cfg, {"algo": "fp", "bits": 16, "group_size": -1})
    assert fp.weight_bytes == n_params * 2 and fp.bits_per_weight == 16 and not fp.exact
    q = F.footprint(cfg, {"algo": "rtn", "bits": 4, "group_size": GROUP_SIZE}, model_bytes=n_params * 2)
    assert q.exact and q.bits_per_weight == pytest.approx(4 + (16 + 4) / GROUP_SIZE)
    assert q.weight_bytes < fp.weight_bytes
    # The config-derived remainder equals the exact one for a model the config describes fully.
    assert F.footprint(cfg, {"algo": "rtn", "bits": 4, "group_size": GROUP_SIZE}).other_bits == q.other_bits
    assert F.dtype_bits("torch.float32") == 32 and F.dtype_bits("torch.bfloat16") == 16
    assert F.footprint(cfg, {"algo": "rtn", "bits": 4, "group_size": GROUP_SIZE}, dtype="torch.float32").other_bits == 2 * q.other_bits


def test_row_footprint_caches_and_tolerates_unknown_repos(monkeypatch):
    tiny = TINY_MODELS["opt"]().config

    def fake_load(repo, revision=None):
        if repo == "org/tiny-opt":
            return tiny
        raise OSError("no such repo")

    monkeypatch.setattr(F, "load_config", fake_load)
    configs: dict = {}
    row = {"model": "org/missing", "algo": "rtn", "bits": 4, "group_size": 16, "status": "ok"}
    assert F.row_footprint(row, configs) is None
    assert configs == {("org/missing", None): None}
    good = {"model": "org/tiny-opt", "model_revision": "abc", "algo": "rtn", "bits": 4, "group_size": 16, "dtype": "torch.float16"}
    fp = F.row_footprint(good, configs)
    assert fp is not None and fp.bits_per_weight == pytest.approx(4 + 20 / 16) and ("org/tiny-opt", "abc") in configs
    assert F.row_footprint({"algo": "rtn"}, configs) is None, "no repo, no footprint"
