"""DESIGN.md 9: matrix expansion, stable ids, extends, family overrides."""

from __future__ import annotations

import pytest

from ptqbench import config as C
from ptqbench import paths

pytestmark = pytest.mark.smoke

EXP = paths.configs_dir() / "experiments"


def test_smoke_config_expands_to_four_rows():
    runs = C.expand(C.load_experiment(EXP / "smoke_opt125m.yaml"))
    assert [r.quant.algo for r in runs] == ["fp", "rtn", "gptq", "hqq"]
    assert all(r.dataset == "wikitext2" for r in runs)
    assert runs[2].calib is not None and runs[2].calib.dataset == "wikitext2"
    assert runs[1].calib is None, "data-free rows carry no calibration"


def test_ids_are_stable_and_distinct():
    runs = C.expand(C.load_experiment(EXP / "smoke_opt125m.yaml"))
    ids = [r.run_id for r in runs]
    assert len(set(ids)) == len(ids)
    again = C.expand(C.load_experiment(EXP / "smoke_opt125m.yaml"))
    assert [r.run_id for r in again] == ids
    assert all(len(i) == 12 for i in ids)


def test_dtype_is_part_of_quant_key():
    runs = C.expand(C.load_experiment(EXP / "smoke_opt125m.yaml"))
    a = runs[1]
    b = a.model_copy(update={"dtype": "float32"})
    assert a.quant_key != b.quant_key


def test_eval_mode_is_not_part_of_run_id():
    runs = C.expand(C.load_experiment(EXP / "smoke_opt125m.yaml"))
    a = runs[0]
    b = a.model_copy(update={"eval": a.eval.model_copy(update={"eval_mode": "streamed"})})
    assert a.run_id == b.run_id


def test_resident_full_matrix_shape():
    exp = C.load_experiment(EXP / "resident_full.yaml")
    runs = C.expand(exp)
    per_model = {}
    for r in runs:
        per_model.setdefault(r.model.key, set()).add((r.quant.algo, r.quant.bits, r.quant.group_size))
    # fp(1) + rtn(9) + gptq(9) + hqq(9) + awq(2) = 30 quant configs for an OPT model
    assert len(per_model["opt-125m"]) == 30
    # SmolLM2 adds g64 wherever the grid has per-row: rtn/gptq/hqq at 8,4,3 bits
    assert len(per_model["smollm2-135m"]) == 30 + 9
    assert len(runs) == (30 * 4 + 39) * 3


def test_family_override_sets_act_order_for_llama_only():
    runs = C.expand(C.load_experiment(EXP / "resident_full.yaml"))
    smol = [r for r in runs if r.model.key == "smollm2-135m" and r.quant.algo == "gptq"]
    opt = [r for r in runs if r.model.key == "opt-125m" and r.quant.algo == "gptq"]
    assert all(r.quant.act_order for r in smol)
    assert not any(r.quant.act_order for r in opt)


def test_extends_chain_and_override():
    exp = C.load_experiment(EXP / "streamed_opt67b.yaml")
    assert exp.models == ["opt-6.7b"]
    assert exp.eval.eval_mode == "streamed"
    assert exp.datasets == ["wikitext2", "c4_new", "ptb_new"]  # inherited through two levels
    assert len(exp.grid) == 5


def test_fp_rejects_low_bits():
    with pytest.raises(ValueError):
        C.QuantSpec(algo="fp", bits=4)


def test_every_model_config_loads():
    for path in sorted((paths.configs_dir() / "models").glob("*.yaml")):
        spec = C.load_model(path.stem)
        assert spec.repo and spec.tokenizer_class


def test_mirror_substitution_keeps_the_canonical_id():
    spec = C.load_model("llama-2-7b")
    assert spec.gated and spec.mirror and spec.mirror_revision
    m = spec.with_mirror()
    assert m.repo == spec.mirror and m.revision == spec.mirror_revision
    assert m.canonical_repo == "meta-llama/Llama-2-7b-hf"
    official = C.RunSpec(model=spec, quant=C.QuantSpec(algo="fp"), dataset="wikitext2", dtype="torch.float16")
    mirrored = official.model_copy(update={"model": m})
    assert official.quant_key != mirrored.quant_key, "mirror weights are different weights until verified"
