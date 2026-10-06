"""The plain-language layer: reason codes become sentences with a fix, and a run gets
one verdict sentence against the measured full-precision baseline."""

from __future__ import annotations

import json

import pytest

from ptqbench import explain as E

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize(
    "reason,needle,fix",
    [
        ("import_error:hqq.core.quantize:ModuleNotFoundError", "`hqq` library is not installed", "uv sync --extra hqq"),
        ("import_error:somepkg:ImportError", "`somepkg` library", "install the `somepkg` package"),
        ("group_size_indivisible:128:[576, 1536]", "Groups of 128 do not divide", "groups of 64 or per row"),
        ("bits_unsupported:5", "cannot do 5-bit", "2, 3, 4 or 8"),
        ("device_unsupported:cpu", "cannot run on the cpu", "NVIDIA GPU"),
        ("unknown_algo:magic", "no method called `magic`", "fp, rtn, gptq, awq_lite or hqq"),
        ("oom", "GPU ran out of memory", "--device cpu"),
        ("OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB", "GPU ran out of memory", "--device cpu"),
        ("OSError: nobody/x is not a valid model identifier listed on huggingface.co", "could not be downloaded", "huggingface.co"),
        ("ConnectionError: HTTPSConnectionPool max retries exceeded", "machine is offline", "prefetch"),
        ("KeyboardInterrupt: ", "Stopped by the user", "Ctrl-C"),
        ("RuntimeError: something odd", "Failed with `RuntimeError: something odd`", ".log"),
        (None, "No reason was recorded", ""),
    ],
)
def test_every_reason_code_gets_a_sentence_and_a_fix(reason, needle, fix):
    text = E.explain_reason(reason)
    assert needle in text and fix in text
    assert text.endswith(".")


def _row(algo="rtn", bits=4, gs=128, ppl=29.45, partial=False, dataset="wikitext2"):
    return {"model": "facebook/opt-125m", "model_key": "opt-125m", "dataset": dataset, "algo": algo,
            "bits": bits, "group_size": gs, "ppl": ppl, "partial": partial}


def test_verdict_thresholds_follow_the_results_guide():
    base = (27.66, False)
    assert "hard to notice" in E.verdict(_row(ppl=27.7), base)
    assert "noticeable but usable" in E.verdict(_row(ppl=29.45), base)
    assert "clear loss" in E.verdict(_row(ppl=37.28, gs=-1), base)
    assert "effectively broken" in E.verdict(_row(ppl=1277.0, bits=3), base)
    text = E.verdict(_row(ppl=29.45), base)
    assert text.startswith("At 4 bits with RTN (plain rounding), groups of 128, opt-125m is 6.5% worse")
    assert "WikiText-2 (Wikipedia articles)" in text and "29.45 vs 27.66" in text


def test_verdict_for_baseline_missing_and_previews():
    assert "Baseline: opt-125m scores 27.66" in E.verdict(_row(algo="fp", bits=16, gs=-1, ppl=27.66), None)
    missing = E.verdict(_row(), None)
    assert "No full-precision baseline" in missing and "Full precision" in missing
    preview = E.verdict(_row(partial=True), (27.66, True))
    assert "quick preview, approximate" in preview and "quick-preview baseline" in preview
    assert "0.4% better" in E.verdict(_row(ppl=27.55), (27.66, False))


def test_baseline_lookup_prefers_full_rows(tmp_path):
    def write(name, **kw):
        row = {"model": "facebook/opt-125m", "dataset": "wikitext2", "algo": "fp", "status": "ok",
               "ppl": 27.0, "partial": False, "finished_at": "2026-01-01T00:00:00Z", **kw}
        (tmp_path / f"{name}.json").write_text(json.dumps(row))

    assert E.baseline_ppl("facebook/opt-125m", "wikitext2", runs_dir=tmp_path) is None
    write("preview", ppl=28.0, partial=True)
    assert E.baseline_ppl("facebook/opt-125m", "wikitext2", runs_dir=tmp_path) is None
    assert E.baseline_ppl("facebook/opt-125m", "wikitext2", runs_dir=tmp_path, allow_partial=True) == (28.0, True)
    write("old", ppl=27.5, finished_at="2026-01-01T00:00:00Z")
    write("new", ppl=27.66, finished_at="2026-02-01T00:00:00Z")
    write("other-model", ppl=5.0, model="meta-llama/Llama-2-7b-hf")
    write("failed", ppl=1.0, status="failed")
    (tmp_path / "junk.json").write_text("{not json")
    assert E.baseline_ppl("facebook/opt-125m", "wikitext2", runs_dir=tmp_path) == (27.66, False)
    assert E.baseline_ppl("facebook/opt-125m", "c4_new", runs_dir=tmp_path) is None


def test_labels_are_shared_with_the_by_model_pages():
    from ptqbench.analysis import aggregate as A

    assert A.ALGO_NAMES is E.ALGO_NAMES and A.DATASET_NAMES is E.DATASET_NAMES
    assert E.grouping(-1) == "per row" and E.grouping(None) == "per row" and E.grouping(64) == "groups of 64"
    assert E.method_name("fp") == "Full precision" and E.method_name("rtn") == "RTN (plain rounding)"
    assert [k for k, _, _ in E.METHOD_MENU] == ["fp", "rtn", "gptq", "awq_lite", "hqq"]
