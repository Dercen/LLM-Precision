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
    assert text == ("opt-125m: GPTQ, 4 bits, groups of 128, calibration text c4; "
                    "tested on WikiText-2 (Wikipedia articles), C4 (web pages)")
    fp = W._summary(W.Answers(model_key="opt-125m", datasets=["wikitext2"], algo="fp", quick=True))
    assert fp == "opt-125m: full precision (no quantization); tested on WikiText-2 (Wikipedia articles), quick preview"


def test_recommended_group_size_respects_model_shape():
    assert W.recommended_group_size("opt-125m") == 128
    assert W.recommended_group_size("smollm2-135m") == 64, "128 does not divide SmolLM2's hidden size"


def test_demo_is_baseline_then_four_bit_quick_previews():
    demo = W.demo_answers()
    assert [a.algo for a in demo] == ["fp", "rtn"] and all(a.quick for a in demo)
    assert demo[1].bits == 4 and demo[1].group_size == 128 and demo[1].datasets == ["wikitext2"]
    runs = W.build_runs(demo[1])
    assert runs[0].eval.max_windows == 20


def test_estimate_reads_like_a_sentence():
    assert W._minutes(30) == "under a minute"
    assert W._minutes(170) == "about 3 minutes"
    assert W._minutes(5 * 3600) == "about 5.0 hours"
    text = W.estimate(W.demo_answers()[1], "cpu")
    assert text.startswith(("under a minute", "about "))
    big = W.estimate(W.Answers(model_key="opt-6.7b", datasets=["wikitext2"], algo="gptq", bits=4, group_size=128), "cpu")
    assert "about" in big and ("hour" in big or "minute" in big)
