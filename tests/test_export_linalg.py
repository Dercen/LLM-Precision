"""The linalg output form of the export (item 1 of the vault hand-off).

Both tiny families: the raw import succeeds (the Llama case is the regression for the
frozen importer's "Could not find state mapping for buffer"), the lowering verifies,
contains linalg ops and no torch-dialect or tm_tensor leftovers, and dequantization has
become plain integer-to-float arithmetic. The CLI writes both forms and records them.
"""

from __future__ import annotations

import json

import pytest
import torch

from ptqbench.export import mlir as M
from ptqbench.export import swap_in_quantized
from ptqbench.quantizers import fakequant as fq
from ptqbench.quantizers import rtn
from tests.tiny_models import GROUP_SIZE, TINY_MODELS, VOCAB, tiny_opt, tiny_tokenizer

pytestmark = pytest.mark.smoke

#: Example length for the tiny exports; anything shorter than the position table works.
SEQLEN = 16
#: Bits for the tiny quantizations: 4 keeps every code in a `ui8` container visibly sub-byte.
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
    torch.manual_seed(2)
    return torch.randint(3, VOCAB, (1, SEQLEN))


@pytest.mark.parametrize("explicit_positions", [False, True])
@pytest.mark.parametrize("family", sorted(TINY_MODELS))
def test_lowering_is_linalg_only_with_arithmetic_dequant(family, explicit_positions, ids):
    pytest.importorskip("torch_mlir")
    from ptqbench.export.stage import ExplicitPositions

    model = quantized_tiny(family)
    fn = ExplicitPositions(model) if explicit_positions else model
    with torch.no_grad():
        plain = model(ids).logits
        assert torch.equal(fn(ids) if explicit_positions else plain, plain), "the wrapper is the same model"
    imported = M.import_function(fn, (ids,), func_name="main")
    raw_ops = M.count_ops(imported.module, M.RAW_OP_PREFIXES)
    n_quantized = 12 if family == "opt" else 14
    assert raw_ops["torch.quantized_decomposed.dequantize_per_channel_group"] == n_quantized
    assert imported.n_user_inputs == 1 and imported.lifted == []

    M.lower_to_linalg(imported.module)
    report = M.check_lowered(imported.module)  # verifies, raises on torch-dialect leftovers
    assert report.linalg_ops["linalg.matmul"] >= n_quantized and "linalg.generic" in report.linalg_ops
    text = str(imported.module)
    assert "dequantize" not in text  # ops only: resource names still read `torch_tensor_..._torch.float32`
    assert not any(op.startswith("torch.") for op in M.count_ops(imported.module))
    # The quantized-op lowering: widen the ui8 code, int -> float, subtract zero, multiply scale.
    for op in ("arith.extui", "arith.sitofp", "arith.subf", "arith.mulf"):
        assert op in text, op
    assert "xi8>" in text  # the ui8 codes became a signless i8 tensor
    if explicit_positions:
        assert report.tm_tensor_ops == {}
    else:
        # The model's own position handling carries a cumsum: documented, counted, noted.
        assert report.tm_tensor_ops == {"tm_tensor.scan": 1, "tm_tensor.yield": 1}


def test_check_lowered_rejects_an_unlowered_module(ids):
    pytest.importorskip("torch_mlir")
    imported = M.import_function(quantized_tiny("opt"), (ids,), func_name="main")
    with pytest.raises(ValueError, match="no linalg ops"):
        M.check_lowered(imported.module)


def test_write_forms_both(tmp_path, ids):
    pytest.importorskip("torch_mlir")
    from ptqbench.export.stage import ExplicitPositions

    imported = M.import_function(ExplicitPositions(quantized_tiny("llama")), (ids,), func_name="main")
    files, stats = M.write_forms(imported, tmp_path, "model", output_type="both", bytecode=True)
    assert files == {
        "mlir": "model.mlir", "mlir_bytecode": "model.mlirbc",
        "mlir_linalg": "model.linalg.mlir", "mlir_linalg_bytecode": "model.linalg.mlirbc",
    }
    assert stats["forms"] == ["raw", "linalg"] and stats["weights"] == "embedded"
    assert stats["ops"]["torch.quantized_decomposed.dequantize_per_channel_group"] == 14
    assert stats["linalg_ops"]["linalg.matmul"] >= 14 and stats["tm_tensor_ops"] == {}
    for name in files.values():  # every form re-parses with the dialects the wheel ships
        assert M.parse(tmp_path / name).operation.verify()
    raw = (tmp_path / "model.mlir").read_text()
    lowered = (tmp_path / "model.linalg.mlir").read_text()
    assert "dequantize_per_channel_group" in raw and "func.func @main" in lowered
    assert "dense_resource<torch_tensor_" in lowered, "weights stay embedded through the lowering"


def test_write_forms_rejects_unknown_output_type(tmp_path, ids):
    pytest.importorskip("torch_mlir")
    imported = M.import_function(quantized_tiny("opt"), (ids,), func_name="main")
    with pytest.raises(ValueError, match="output_type"):
        M.write_forms(imported, tmp_path, "model", output_type="tosa")


def test_cli_output_type_both(tmp_path, monkeypatch):
    """`ptq export-mlir --output-type both`: both files, both recorded, both verify."""
    pytest.importorskip("torch_mlir")
    from ptqbench import cli

    src = tmp_path / "tiny-opt"
    tiny_opt().save_pretrained(src)
    tiny_tokenizer().save_pretrained(src)
    monkeypatch.setattr("ptqbench.paths.runs_dir", lambda: tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    out = tmp_path / "out"
    argv = [
        "--device", "cpu", "export-mlir", "--model", str(src), "--bits", str(BITS),
        "--group-size", str(GROUP_SIZE), "--output-type", "both", "--explicit-positions",
        "--seqlen", str(SEQLEN), "--out", str(out),
    ]
    assert cli.main(argv) == cli.OK
    manifest = json.loads((out / "export.json").read_text())
    assert manifest["files"]["mlir"] == "model.mlir" and manifest["files"]["mlir_linalg"] == "model.linalg.mlir"
    assert manifest["mlir"]["forms"] == ["raw", "linalg"] and manifest["mlir"]["explicit_positions"] is True
    assert manifest["mlir"]["ops"]["torch.quantized_decomposed.dequantize_per_channel_group"] == 12
    assert manifest["mlir"]["linalg_ops"]["linalg.matmul"] >= 12
    assert manifest["mlir"]["tm_tensor_ops"] == {} and manifest["notes"] == []
    assert manifest["summary"]["all_bitwise_equal_to_evaluated"] is True
    M.parse(out / "model.mlir")
    M.parse(out / "model.linalg.mlir")


def test_cli_default_linalg_notes_the_scan(tmp_path, monkeypatch):
    """Without --explicit-positions the lowered whole model keeps one tm_tensor.scan, and says so."""
    pytest.importorskip("torch_mlir")
    from ptqbench import cli

    src = tmp_path / "tiny-opt"
    tiny_opt().save_pretrained(src)
    tiny_tokenizer().save_pretrained(src)
    monkeypatch.setattr("ptqbench.paths.runs_dir", lambda: tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    out = tmp_path / "out"
    argv = [
        "--device", "cpu", "export-mlir", "--model", str(src), "--bits", str(BITS),
        "--group-size", str(GROUP_SIZE), "--output-type", "linalg", "--seqlen", str(SEQLEN), "--out", str(out),
    ]
    assert cli.main(argv) == cli.OK
    manifest = json.loads((out / "export.json").read_text())
    assert manifest["files"]["mlir"] is None and manifest["files"]["mlir_linalg"] == "model.linalg.mlir"
    assert manifest["mlir"]["forms"] == ["linalg"] and manifest["mlir"]["tm_tensor_ops"] == {"tm_tensor.scan": 1, "tm_tensor.yield": 1}
    assert any("tm_tensor.scan" in note and "--explicit-positions" in note for note in manifest["notes"])
    assert not (out / "model.mlir").exists()
    # The wheel ships no tm_tensor Python binding: such a file is torch-mlir-opt-only.
    with pytest.raises(Exception, match="tm_tensor"):
        M.parse(out / "model.linalg.mlir")
