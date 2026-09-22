"""The wizard's logic layer, without prompts: answers -> RunSpecs -> rows."""

from __future__ import annotations

import pytest

from ptqbench import wizard as W

pytestmark = pytest.mark.smoke


def test_build_runs_matches_matrix_ids():
    from ptqbench import config as C
    from ptqbench import paths

    a = W.Answers(model_key="opt-125m", datasets=["wikitext2"], algo="rtn", bits=4, group_size=-1)
    runs = W.build_runs(a)
    matrix = [r for r in C.expand(C.load_experiment(paths.configs_dir() / "experiments" / "resident_full.yaml"))
              if r.model.key == "opt-125m" and r.quant.algo == "rtn" and r.quant.bits == 4
              and r.quant.group_size == -1 and r.dataset == "wikitext2"]
    assert runs[0].run_id == matrix[0].run_id, "a wizard row must be the same row the matrix produces"


def test_llama_defaults_to_act_order_and_fp_ignores_bits():
    llama = W.build_runs(W.Answers(model_key="smollm2-135m", algo="gptq", bits=4))[0]
    assert llama.quant.act_order is True and llama.calib is not None
    opt = W.build_runs(W.Answers(model_key="opt-125m", algo="gptq", bits=4))[0]
    assert opt.quant.act_order is False
    fp = W.build_runs(W.Answers(model_key="opt-125m", algo="fp", bits=3, group_size=64))[0]
    assert fp.quant.bits == 16 and fp.quant.group_size == -1 and fp.calib is None


def test_model_choices_lists_every_config():
    keys = {m["key"] for m in W.model_choices()}
    assert {"opt-125m", "opt-6.7b", "llama-2-7b"} <= keys
    assert all(isinstance(m["cached"], bool) for m in W.model_choices())


def test_execute_answers_end_to_end_on_cpu():
    a = W.Answers(model_key="opt-125m", datasets=["wikitext2"], algo="rtn", bits=8, quick=True)
    rows = W.execute_answers(a, device_spec="cpu", write=False)
    assert len(rows) == 1 and rows[0]["partial"] is True and rows[0]["ppl"] > 1
    paper, _src = W.paper_number({**rows[0], "partial": False})
    assert paper is None or paper > 1


def test_summary_reads_well():
    text = W._summary(W.Answers(model_key="opt-125m", datasets=["wikitext2", "c4_new"], algo="gptq", bits=4, group_size=128))
    assert text == "opt-125m: gptq 4-bit g128, calibration c4 on wikitext2, c4_new"
