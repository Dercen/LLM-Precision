"""Stages (item 5 of the vault hand-off): prefill, prefill_kv and decode with an explicit
static cache, for both tiny families. The functional cache reproduces the eager model
over prefill + prefill_kv + several decode steps, every stage exports with the
documented signature and lowers to linalg with nothing left behind, and the CLI writes
three files, three golden cases, one shared weights.bin, and a passing round trip.
"""

from __future__ import annotations

import json

import pytest
import torch

from ptqbench.export import golden as G
from ptqbench.export import mlir as M
from ptqbench.export import stages as S
from ptqbench.export import swap_in_quantized
from ptqbench.models import families
from ptqbench.quantizers import fakequant as fq
from ptqbench.quantizers import rtn
from tests.test_export_golden import fake_stream
from tests.tiny_models import (
    GROUP_SIZE,
    N_BLOCKS,
    N_HEADS,
    N_KV_HEADS,
    TINY_MODELS,
    VOCAB,
    tiny_llama,
    tiny_tokenizer,
)

pytestmark = pytest.mark.smoke

BITS = 4
#: Tiny stage geometry: an 8-token prompt, a 4-token second chunk, a 32-slot cache.
SEQLEN = 8
KV_SEQLEN = 4
MAX_CACHE = 32
HEAD_DIM = 16


def quantized_tiny(family: str):
    model = TINY_MODELS[family]()
    rec = fq.Recorder()
    with torch.no_grad():
        rtn.apply_rtn(model, bits=BITS, group_size=GROUP_SIZE, record=rec)
    swap_in_quantized(model, rec.records)
    return model


@pytest.fixture(scope="module")
def ids() -> torch.Tensor:
    torch.manual_seed(11)
    return torch.randint(3, VOCAB, (1, SEQLEN + KV_SEQLEN + S.DECODE_CHECK_STEPS))


@pytest.mark.parametrize("family", sorted(TINY_MODELS))
def test_stages_match_the_eager_model(family, ids):
    model = quantized_tiny(family)
    fam = families.for_model(model)
    stages = S.build_stages(model, fam, MAX_CACHE)
    kv_heads = N_KV_HEADS if family == "llama" else N_HEADS
    assert stages["prefill"].shape == (N_BLOCKS, 1, kv_heads, MAX_CACHE, HEAD_DIM)
    assert stages["prefill_kv"] is stages["decode"], "the same function at two input lengths"
    run = S.run_stages(stages, ids, SEQLEN, KV_SEQLEN)
    logits, k, v = run.outputs["prefill"]
    assert logits.shape == (1, SEQLEN, VOCAB) and k.shape == stages["prefill"].shape and v.shape == k.shape
    assert run.args["prefill_kv"][3].item() == SEQLEN and run.args["decode"][3].item() == SEQLEN + KV_SEQLEN
    assert run.outputs["decode"][0].shape == (1, 1, VOCAB)
    assert torch.count_nonzero(k[:, :, :, SEQLEN:]) == 0, "slots past the prompt stay empty"
    check = S.check_stages(model, stages, run, tolerance=None)
    assert len(check["comparisons"]) == 2 + S.DECODE_CHECK_STEPS
    assert check["max_abs_diff"] < 1e-5 and all(c["argmax_agreement"] == 1.0 for c in check["comparisons"])
    assert check["all_within_tolerance"] is None  # no tolerance given, nothing judged
    # With the golden tolerance of the prefill logits, the round trip is judged and passes.
    case = G.build_case("prefill", stages["prefill"], [("input_ids", run.args["prefill"][0])], list(S.STAGE_OUTPUTS), source={})
    tol = {"atol": case.tolerance[0]["atol"], "rel_fro_tol": case.tolerance[0]["rel_fro_tol"]}
    judged = S.check_stages(model, stages, run, tol)
    assert judged["all_within_tolerance"] is True and judged["tolerance"] == tol
    assert [c["name"] for c in case.tolerance] == ["logits", "k_cache", "v_cache"]


@pytest.mark.parametrize("family", sorted(TINY_MODELS))
def test_stage_functions_export_and_lower(family, ids):
    pytest.importorskip("torch_mlir")
    model = quantized_tiny(family)
    stages = S.build_stages(model, families.for_model(model), MAX_CACHE)
    run = S.run_stages(stages, ids, SEQLEN, KV_SEQLEN)
    for stage in S.STAGES:
        imported = M.import_function(stages[stage], run.args[stage], func_name=stage)
        n_in = 1 if stage == "prefill" else 4
        assert imported.n_user_inputs == n_in
        text = str(imported.module)
        assert f"func.func @{stage}(" in text
        head = text[text.index(f"func.func @{stage}(") : text.index("{", text.index(f"func.func @{stage}("))]
        assert head.count("%arg") == n_in and head.count("-> (") == 1 and head.count("!torch.vtensor") == n_in + 3
        if stage != "prefill":
            assert "!torch.vtensor<[],si64>" in head, "position is a 0-d int64"
        # The exported graph computes the same thing as the module it came from.
        prog = M.export_program(stages[stage], run.args[stage])
        with torch.no_grad():
            got = prog.module()(*run.args[stage])
        for g, w in zip(got, run.outputs[stage]):
            assert float((g - w).abs().max()) < 1e-5
        M.lower_to_linalg(imported.module)
        report = M.check_lowered(imported.module)
        assert report.tm_tensor_ops == {}, f"{family} {stage}: a stock mlir-opt must read it"
        assert report.linalg_ops.get("linalg.matmul", 0) + report.linalg_ops.get("linalg.batch_matmul", 0) > 0


def test_cache_shape_refuses_sliding_windows():
    cfg = tiny_llama().config
    cfg.sliding_window = 4
    with pytest.raises(NotImplementedError, match="sliding window"):
        S.cache_shape(cfg, MAX_CACHE)


def test_run_stages_needs_enough_tokens(ids):
    model = quantized_tiny("opt")
    stages = S.build_stages(model, families.OPT, MAX_CACHE)
    with pytest.raises(ValueError, match="need at least"):
        S.run_stages(stages, ids[:, :SEQLEN], SEQLEN, KV_SEQLEN)


def _save_tiny(tmp_path, monkeypatch, family="llama"):
    from ptqbench.data import datasets as ds

    monkeypatch.setattr(ds, "build", fake_stream)
    src = tmp_path / f"tiny-{family}"
    TINY_MODELS[family]().save_pretrained(src)
    tiny_tokenizer().save_pretrained(src)
    monkeypatch.setattr("ptqbench.paths.runs_dir", lambda: tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    return src


def test_cli_stages_with_golden_and_external_weights(tmp_path, monkeypatch):
    pytest.importorskip("torch_mlir")
    from ptqbench import cli
    from ptqbench.export import weights as W

    src = _save_tiny(tmp_path, monkeypatch)
    out = tmp_path / "out"
    argv = [
        "--device", "cpu", "export-mlir", "--model", str(src), "--bits", str(BITS), "--group-size", str(GROUP_SIZE),
        "--seqlen", str(SEQLEN), "--stages", "--max-cache-len", str(MAX_CACHE), "--prefill-kv-seqlen", str(KV_SEQLEN),
        "--golden", "--weights", "external", "--output-type", "both", "--out", str(out),
    ]
    assert cli.main(argv) == cli.OK
    manifest = json.loads((out / "export.json").read_text())
    st = manifest["stages"]
    assert st["max_cache_len"] == MAX_CACHE and st["cache_shape"] == [N_BLOCKS, 1, N_KV_HEADS, MAX_CACHE, HEAD_DIM]
    assert st["check"]["all_within_tolerance"] is True and st["check"]["decode_steps"] == S.DECODE_CHECK_STEPS
    sig = st["signatures"]
    assert [e["name"] for e in sig["prefill"]["inputs"]] == ["input_ids"]
    assert [e["name"] for e in sig["decode"]["inputs"]] == ["input_ids", "k_cache", "v_cache", "position"]
    assert sig["decode"]["inputs"][0]["shape"] == [1, 1] and sig["decode"]["inputs"][3]["shape"] == []
    assert sig["prefill_kv"]["inputs"][0]["shape"] == [1, KV_SEQLEN] and sig["prefill"]["outputs"][0]["shape"] == [1, SEQLEN, VOCAB]
    assert manifest["files"]["mlir"] is None and set(manifest["files"]["stages"]) == set(S.STAGES)
    for stage in S.STAGES:
        f = manifest["files"]["stages"][stage]
        assert f["mlir"] == f"{stage}.mlir" and f["mlir_linalg"] == f"{stage}.linalg.mlir"
        assert manifest["mlir"]["functions"][stage]["tm_tensor_ops"] == {}
        assert f"func.func @{stage}(" in (out / f"{stage}.mlir").read_text()
        M.parse(out / f"{stage}.linalg.mlir")
    # Three golden cases: the caches of prefill_kv and decode are the previous stage's outputs.
    index = json.loads((out / "golden" / "index.json").read_text())
    assert set(index["functions"]) == set(S.STAGES)
    for stage in S.STAGES:
        fn = index["functions"][stage]
        assert [e["name"] for e in fn["outputs"]] == ["logits", "k_cache", "v_cache"]
        assert fn["tolerance"] and fn["tolerance"][0]["max_abs_err_vs_fp64"] > 0
    k_out = G.read_tensor(out / "golden" / "prefill", index["functions"]["prefill"]["outputs"][1])
    k_in = G.read_tensor(out / "golden" / "prefill_kv", index["functions"]["prefill_kv"]["inputs"][1])
    assert torch.equal(k_out, k_in) and index["functions"]["prefill_kv"]["source"]["cache_from"] == "prefill"
    assert G.read_tensor(out / "golden" / "decode", index["functions"]["decode"]["inputs"][3]).item() == SEQLEN + KV_SEQLEN
    # One weights.bin shared by the three functions, each with its own argument order.
    w = json.loads((out / "weights.json").read_text())
    assert set(w["functions"]) == set(S.STAGES) and w["functions"]["prefill"]["n_user_inputs"] == 1 and w["functions"]["decode"]["n_user_inputs"] == 4
    assert all(len(f["arguments"]) == w["n_tensors"] for f in w["functions"].values())
    tensors, widx = W.read_weights(out)
    assert len(tensors) == w["n_tensors"]
    # The weights rebuild a model whose stage functions reproduce the golden decode logits.
    blank = quantized_tiny("llama")
    with torch.no_grad():
        for t in list(blank.parameters()) + list(blank.buffers()):
            t.zero_()
    W.load_into(blank, tensors, widx)
    stages = S.build_stages(blank, families.LLAMA, MAX_CACHE)
    dec = index["functions"]["decode"]
    args = [G.read_tensor(out / "golden" / "decode", e) for e in dec["inputs"]]
    with torch.no_grad():
        logits = stages["decode"](*args)[0]
    assert torch.equal(logits, G.read_tensor(out / "golden" / "decode", dec["outputs"][0]))


def test_cli_stages_rejections(tmp_path, monkeypatch):
    from ptqbench import cli

    src = _save_tiny(tmp_path, monkeypatch, family="opt")
    base = [
        "--device", "cpu", "export-mlir", "--model", str(src), "--bits", str(BITS), "--group-size", str(GROUP_SIZE),
        "--seqlen", str(SEQLEN), "--stages", "--max-cache-len", str(MAX_CACHE), "--prefill-kv-seqlen", str(KV_SEQLEN),
        "--no-mlir", "--out", str(tmp_path / "out"),
    ]
    with pytest.raises(ValueError, match="mutually exclusive"):
        cli.main(base + ["--slice", "block"])
    with pytest.raises(ValueError, match="static shapes"):
        cli.main(base + ["--dynamic-seqlen"])
    too_small = list(base)
    too_small[too_small.index("--max-cache-len") + 1] = str(SEQLEN)
    with pytest.raises(ValueError, match="cannot hold"):
        cli.main(too_small)
