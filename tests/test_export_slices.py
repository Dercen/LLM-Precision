"""Slices (item 3 of the vault hand-off): one block, its attention or its MLP as a
function of the hidden states it really receives, for both tiny families. The capture
replays bitwise, exports to the raw torch dialect, lowers to linalg with nothing left
behind, and the CLI writes the files, the golden vectors and the report.
"""

from __future__ import annotations

import json

import pytest
import torch

from ptqbench.export import golden as G
from ptqbench.export import mlir as M
from ptqbench.export import slices as S
from ptqbench.export import swap_in_quantized
from ptqbench.models import families
from ptqbench.quantizers import fakequant as fq
from ptqbench.quantizers import rtn
from tests.test_export_golden import fake_stream
from tests.tiny_models import (
    GROUP_SIZE,
    HIDDEN,
    N_BLOCKS,
    TINY_MODELS,
    VOCAB,
    tiny_opt,
    tiny_tokenizer,
)

pytestmark = pytest.mark.smoke

SEQLEN = 16
BITS = 4


def quantized_tiny(family: str):
    model = TINY_MODELS[family]()
    rec = fq.Recorder()
    with torch.no_grad():
        rtn.apply_rtn(model, bits=BITS, group_size=GROUP_SIZE, record=rec)
    swap_in_quantized(model, rec.records)
    return model


@pytest.fixture(scope="module")
def ids() -> torch.Tensor:
    torch.manual_seed(5)
    return torch.randint(3, VOCAB, (1, SEQLEN))


def test_family_map_names_the_pieces():
    opt, llama = tiny_opt(), TINY_MODELS["llama"]()
    ob, lb = families.OPT.block_list(opt)[0], families.LLAMA.block_list(llama)[0]
    assert families.OPT.attention_module(ob) is ob.self_attn and families.LLAMA.attention_module(lb) is lb.self_attn
    assert families.LLAMA.mlp_module(lb) is lb.mlp and families.LLAMA.mlp_bounds(lb) == (lb.mlp, lb.mlp)
    first, last = families.OPT.mlp_bounds(ob)
    assert first is ob.fc1 and last is ob.fc2
    mlp = families.OPT.mlp_module(ob)
    assert isinstance(mlp, families.OPTMLP) and mlp.fc1 is ob.fc1 and mlp.fc2 is ob.fc2, "shares the block's weights"
    x = torch.randn(SEQLEN, HIDDEN)
    with torch.no_grad():
        assert torch.equal(mlp(x), ob.fc2(ob.activation_fn(ob.fc1(x))))


@pytest.mark.parametrize("kind", S.SLICE_KINDS)
@pytest.mark.parametrize("family", sorted(TINY_MODELS))
def test_capture_replays_bitwise_and_lowers(family, kind, ids):
    model = quantized_tiny(family)
    fam = families.for_model(model)
    index = N_BLOCKS - 1
    captured = S.capture_piece(model, fam, kind, index, ids)
    assert captured.function == f"{kind}_{index}" and captured.args == ()
    if family == "opt" and kind == "mlp":
        assert captured.hidden.shape == (SEQLEN, HIDDEN), "OPT flattens before fc1"
    else:
        assert captured.hidden.shape == (1, SEQLEN, HIDDEN)
    assert captured.output.shape == captured.hidden.shape
    replay, report = S.build_replay(captured)
    assert report.replay_bitwise and report.function == captured.function
    assert report.quantized_modules, "the piece holds quantized layers"
    with torch.no_grad():
        assert torch.equal(replay(captured.hidden), captured.output)
    if kind == "mlp":
        assert report.captured_kwargs == {}
    else:
        assert "attention_mask" in report.captured_kwargs and report.captured_kwargs["attention_mask"] is None
        if family == "llama":
            cos_sin = report.captured_kwargs["position_embeddings"]
            assert [c["shape"] for c in cos_sin] == [[1, SEQLEN, HIDDEN // 4]] * 2, "rotary cos/sin frozen in"
    # The piece is a pure function of its hidden states: it exports and lowers cleanly.
    pytest.importorskip("torch_mlir")
    imported = M.import_function(replay, (captured.hidden,), func_name=captured.function)
    assert f"func.func @{captured.function}" in str(imported.module)
    M.lower_to_linalg(imported.module)
    lowered = M.check_lowered(imported.module)
    assert lowered.tm_tensor_ops == {} and lowered.linalg_ops.get("linalg.matmul", 0) + lowered.linalg_ops.get("linalg.batch_matmul", 0) > 0


def test_capture_rejects_bad_block(ids):
    model = quantized_tiny("opt")
    with pytest.raises(IndexError, match="out of range"):
        S.capture_piece(model, families.OPT, "block", N_BLOCKS, ids)
    with pytest.raises(ValueError, match="slice must be"):
        S.function_name("norm", 0)


def test_slice_golden_case_measures_the_piece(ids):
    model = quantized_tiny("llama")
    captured = S.capture_piece(model, families.LLAMA, "block", 0, ids)
    replay, _ = S.build_replay(captured)
    case = G.build_case("block_0", replay, [("hidden_states", captured.hidden)], ["hidden_states_out"], source={"kind": "captured_activations"})
    assert torch.equal(case.outputs[0][1], captured.output)
    (tol,) = case.tolerance
    assert 0 < tol["max_abs_err_vs_fp64"] < 1e-4 and tol["atol"] >= G.TOLERANCE_SAFETY_FACTOR * tol["max_abs_err_vs_fp64"]


@pytest.mark.parametrize("kind", ["mlp", "attention"])
def test_cli_slice(tmp_path, monkeypatch, kind):
    """`ptq export-mlir --slice <kind> --block-index 1 --golden` on the tiny OPT."""
    from ptqbench import cli
    from ptqbench.data import datasets as ds

    monkeypatch.setattr(ds, "build", fake_stream)
    src = tmp_path / "tiny-opt"
    tiny_opt().save_pretrained(src)
    tiny_tokenizer().save_pretrained(src)
    monkeypatch.setattr("ptqbench.paths.runs_dir", lambda: tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    out = tmp_path / "out"
    argv = [
        "--device", "cpu", "export-mlir", "--model", str(src), "--bits", str(BITS), "--group-size", str(GROUP_SIZE),
        "--seqlen", str(SEQLEN), "--slice", kind, "--block-index", "1", "--golden", "--out", str(out),
    ]
    have_mlir = M.available()
    if have_mlir:
        argv += ["--output-type", "both"]
    else:
        argv.append("--no-mlir")
    assert cli.main(argv) == cli.OK
    manifest = json.loads((out / "export.json").read_text())
    function = f"{kind}_1"
    sl = manifest["slice"]
    assert sl["kind"] == kind and sl["block_index"] == 1 and sl["function"] == function and sl["replay_bitwise"] is True
    expected_shape = [SEQLEN, HIDDEN] if kind == "mlp" else [1, SEQLEN, HIDDEN]
    assert sl["input_shape"] == expected_shape and sl["output_shape"] == expected_shape
    assert sl["quantized_modules"] == (["fc1", "fc2"] if kind == "mlp" else ["k_proj", "out_proj", "q_proj", "v_proj"])
    assert manifest["n_quantized_modules"] == 12, "layers.json still describes the whole quantization"
    if have_mlir:
        assert manifest["files"]["mlir"] == f"{function}.mlir" and manifest["files"]["mlir_linalg"] == f"{function}.linalg.mlir"
        assert manifest["mlir"]["function"] == function and manifest["mlir"]["tm_tensor_ops"] == {}
        assert not (out / "model.mlir").exists()
        assert f"func.func @{function}" in (out / f"{function}.mlir").read_text()
        M.parse(out / f"{function}.linalg.mlir")
    index = json.loads((out / "golden" / "index.json").read_text())
    fn = index["functions"][function]
    assert [e["name"] for e in fn["inputs"]] == ["hidden_states"]
    assert [e["name"] for e in fn["outputs"]] == [S.SLICE_OUTPUT_NAMES[kind]]
    assert fn["source"]["kind"] == "captured_activations" and fn["source"]["text"]["dataset"] == "wikitext2"
    assert fn["inputs"][0]["shape"] == expected_shape and fn["tolerance"][0]["max_abs_err_vs_fp64"] > 0
    # The golden input really is what the piece saw: recompute the capture from the same text.
    model = quantized_tiny("opt")
    ids = fake_stream("wikitext2", tiny_tokenizer()).window(fn["source"]["text"]["window_index"])[:, :SEQLEN]
    captured = S.capture_piece(model, families.OPT, kind, 1, ids)
    assert torch.equal(G.read_tensor(out / "golden" / function, fn["inputs"][0]), captured.hidden)
    assert torch.equal(G.read_tensor(out / "golden" / function, fn["outputs"][0]), captured.output)


def test_cli_slice_rejects_dynamic_seqlen(tmp_path, monkeypatch):
    from ptqbench import cli
    from ptqbench.data import datasets as ds

    monkeypatch.setattr(ds, "build", fake_stream)
    src = tmp_path / "tiny-opt"
    tiny_opt().save_pretrained(src)
    tiny_tokenizer().save_pretrained(src)
    monkeypatch.setattr("ptqbench.paths.runs_dir", lambda: tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    argv = [
        "--device", "cpu", "export-mlir", "--model", str(src), "--bits", str(BITS), "--group-size", str(GROUP_SIZE),
        "--seqlen", str(SEQLEN), "--slice", "block", "--dynamic-seqlen", "--no-mlir", "--out", str(tmp_path / "out"),
    ]
    with pytest.raises(ValueError, match="dynamic-seqlen"):
        cli.main(argv)
