"""External packed weights (item 4 of the vault hand-off): pack/unpack round-trips at
every width with the documented bit order, the lifted graph takes its weights as named
arguments and still lowers, weights.bin is aligned and described, and a model rebuilt
from the file pair reproduces the golden outputs bit for bit.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from ptqbench.export import golden as G
from ptqbench.export import mlir as M
from ptqbench.export import swap_in_quantized
from ptqbench.export import weights as W
from ptqbench.quantizers import fakequant as fq
from ptqbench.quantizers import rtn
from tests.test_export_golden import fake_stream
from tests.tiny_models import GROUP_SIZE, TINY_MODELS, VOCAB, tiny_llama, tiny_opt, tiny_tokenizer

pytestmark = pytest.mark.smoke

SEQLEN = 16
BITS = 4


def quantized_tiny(family: str = "opt", bits: int = BITS):
    model = TINY_MODELS[family]()
    rec = fq.Recorder()
    with torch.no_grad():
        rtn.apply_rtn(model, bits=bits, group_size=GROUP_SIZE, record=rec)
    swap_in_quantized(model, rec.records)
    return model


@pytest.mark.parametrize("bits", range(W.MIN_BITS, W.MAX_BITS + 1))
@pytest.mark.parametrize("cols", [13, 64])
def test_unpack_pack_roundtrip(bits, cols):
    rng = np.random.default_rng(bits * 100 + cols)
    codes = rng.integers(0, 2**bits, size=(7, cols), dtype=np.uint8)
    packed = W.pack(codes, bits)
    assert packed.dtype == np.uint8 and packed.shape == (7, W.row_stride_bytes(cols, bits))
    assert W.row_stride_bytes(cols, bits) == -(-cols * bits // 8)
    assert np.array_equal(W.unpack(packed, bits, cols), codes)
    assert np.array_equal(W.unpack(W.pack(torch.from_numpy(codes), bits), bits, cols), codes), "tensors accepted"


def test_pack_bit_order_is_documented():
    # 4 bits: code 0 in the low nibble, code 1 in the high nibble.
    assert W.pack(np.array([[1, 2, 3, 4]], dtype=np.uint8), 4).tolist() == [[0x21, 0x43]]
    # 3 bits: codes 1, 2, 3 -> bit stream 1,0,0 | 0,1,0 | 1,1,0 -> byte 0 = 0b11010001, byte 1 = 0b0.
    assert W.pack(np.array([[1, 2, 3]], dtype=np.uint8), 3).tolist() == [[0xD1, 0x00]]
    # 2 bits: four codes per byte, first code lowest.
    assert W.pack(np.array([[1, 2, 3, 0]], dtype=np.uint8), 2).tolist() == [[0b00111001]]
    # 8 bits: untouched.
    assert W.pack(np.array([[200, 7]], dtype=np.uint8), 8).tolist() == [[200, 7]]
    # Every row is byte aligned on its own: a second row starts in a fresh byte.
    two = W.pack(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8), 3)
    assert two.shape == (2, 2) and two[0].tolist() == [0xD1, 0x00]
    rule = W.packing_rule()
    assert rule["bit_order"] == "lsb-first" and "0xD1" in rule["example_3bit"]


def test_pack_rejects_bad_input():
    with pytest.raises(ValueError, match="bits must be"):
        W.pack(np.zeros((1, 4), dtype=np.uint8), 1)
    with pytest.raises(ValueError, match="must lie in"):
        W.pack(np.array([[16]], dtype=np.uint8), 4)
    with pytest.raises(ValueError, match="2-D"):
        W.pack(np.zeros(4, dtype=np.uint8), 4)
    with pytest.raises(ValueError, match="packed rows must be"):
        W.unpack(np.zeros((1, 3), dtype=np.uint8), 4, 4)


@pytest.fixture(scope="module")
def ids() -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randint(3, VOCAB, (1, SEQLEN))


@pytest.mark.parametrize("family", sorted(TINY_MODELS))
def test_lifted_graph_takes_named_arguments_and_lowers(family, ids, tmp_path):
    pytest.importorskip("torch_mlir")
    model = quantized_tiny(family)
    modules = {n: m for n, m in model.named_modules() if hasattr(m, "weight_q")}
    imported = M.import_function(model, (ids,), func_name="main", lift_weights=True)
    assert imported.n_user_inputs == 1 and imported.stats["weights"] == "lifted"
    names = [a.name for a in imported.lifted]
    assert all(a.index == i + 1 for i, a in enumerate(imported.lifted)), "arguments follow the user input in order"
    for module_name in modules:
        for leaf in ("weight_q", "scale", "zero"):
            assert f"{module_name}.{leaf}" in names
    assert len(names) == len(set(names)), "every tensor lifted once"
    text = str(imported.module)
    assert "dense_resource" not in text and f"%arg{len(imported.lifted)}" in text
    assert "!torch.vtensor<[64,64],ui8>" in text or "!torch.vtensor<[64,64],ui8>" in text.replace(" ", "")

    writer = W.WeightsWriter(model, modules)
    writer.add_function("main", imported.n_user_inputs, imported.lifted)
    bin_path, json_path, index = writer.write(tmp_path)
    assert bin_path.stat().st_size == index["total_bytes"] and index["alignment"] == W.WEIGHT_ALIGNMENT_BYTES
    assert json.loads(json_path.read_text())["functions"] == index["functions"]
    entries = {e["name"]: e for e in index["tensors"]}
    for e in index["tensors"]:
        assert e["offset"] % W.WEIGHT_ALIGNMENT_BYTES == 0
    codes = entries[f"{next(iter(modules))}.weight_q"]
    rows, cols = codes["shape"]
    assert codes["role"] == "codes" and codes["packed"] and codes["bits"] == BITS and codes["group_size"] == GROUP_SIZE
    assert codes["scheme"] == "per_group" and codes["length"] == rows * W.row_stride_bytes(cols, BITS)
    assert codes["row_stride_bytes"] == W.row_stride_bytes(cols, BITS) and codes["dtype"] == "uint8"
    scale = entries[f"{next(iter(modules))}.scale"]
    assert scale["role"] == "scale" and scale["dtype"] == "float32" and not scale["packed"] and scale["bits"] == BITS
    zero = entries[f"{next(iter(modules))}.zero"]
    assert zero["role"] == "zero" and zero["dtype"] == "int32"
    roles = {e["role"] for e in index["tensors"]}
    assert {"codes", "scale", "zero", "bias" if family == "opt" else "float", "float"} <= roles
    assert index["functions"]["main"]["n_user_inputs"] == 1
    assert [a["index"] for a in index["functions"]["main"]["arguments"]] == list(range(1, len(names) + 1))
    # A tied lm_head (OPT) is one argument with an alias, not two tensors; the tiny Llama
    # is untied (LlamaConfig's default), so there its lm_head is a float tensor of its own.
    if model.lm_head.weight is model.get_input_embeddings().weight:
        embed = "model.decoder.embed_tokens.weight"
        assert entries[embed]["aliases"] == ["lm_head.weight"] and "lm_head.weight" not in entries
    else:
        assert entries["lm_head.weight"]["role"] == "float" and entries["lm_head.weight"]["aliases"] == []

    # The graph-only module still lowers, with the lifted arguments typed as plain tensors.
    M.lower_to_linalg(imported.module)
    report = M.check_lowered(imported.module)
    assert report.linalg_ops["linalg.matmul"] >= 12
    assert "tensor<64x64xi8>" in str(imported.module)


def test_read_weights_rebuilds_the_model(ids, tmp_path):
    pytest.importorskip("torch_mlir")
    for bits in (3, 8):
        model = quantized_tiny("llama", bits=bits)
        modules = {n: m for n, m in model.named_modules() if hasattr(m, "weight_q")}
        imported = M.import_function(model, (ids,), func_name="main", lift_weights=True)
        writer = W.WeightsWriter(model, modules)
        writer.add_function("main", 1, imported.lifted)
        out = tmp_path / f"b{bits}"
        writer.write(out)
        tensors, index = W.read_weights(out)
        packed_flags = {e["packed"] for e in index["tensors"] if e["role"] == "codes"}
        assert packed_flags == {bits < 8}
        blank = quantized_tiny("llama", bits=bits)
        with torch.no_grad():
            for t in list(blank.parameters()) + list(blank.buffers()):
                t.zero_()
            assert not torch.equal(blank(ids).logits, model(ids).logits)
        loaded = W.load_into(blank, tensors, index)
        assert "lm_head.weight" in loaded and "model.embed_tokens.weight" in loaded, "untied: both loaded by name"
        with torch.no_grad():
            assert torch.equal(blank(ids).logits, model(ids).logits), f"{bits}-bit rebuild is bitwise"


def test_cli_external_weights_with_golden(tmp_path, monkeypatch):
    """`--weights external --golden --output-type both`: graph-only IR, the file pair,
    and a model rebuilt from weights.bin reproducing the golden logits bitwise."""
    pytest.importorskip("torch_mlir")
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
        "--seqlen", str(SEQLEN), "--weights", "external", "--golden", "--output-type", "both",
        "--explicit-positions", "--out", str(out),
    ]
    assert cli.main(argv) == cli.OK
    manifest = json.loads((out / "export.json").read_text())
    assert manifest["files"]["weights"] == "weights.bin" and manifest["files"]["weights_index"] == "weights.json"
    w = manifest["weights"]
    assert w["mode"] == "external" and w["alignment"] == 64 and w["functions"] == {"main": w["n_tensors"]}
    assert w["roles"]["codes"] == 12 and w["roles"]["scale"] == 12 and w["roles"]["zero"] == 12
    assert manifest["mlir"]["weights"] == "lifted"
    raw = (out / "model.mlir").read_text()
    assert "dense_resource" not in raw, "graph only"
    assert "dense_resource" not in (out / "model.linalg.mlir").read_text()
    index = json.loads((out / "weights.json").read_text())
    names = {e["name"] for e in index["tensors"]}
    assert "model.decoder.layers.0.fc1.weight_q" in names, "named after the model, not the wrapper"
    assert all(not n.startswith("model.model.") for n in names)
    # Rebuild from the file pair and reproduce the golden output bitwise.
    tensors, index = W.read_weights(out)
    blank = quantized_tiny("opt")
    with torch.no_grad():
        for t in list(blank.parameters()) + list(blank.buffers()):
            t.zero_()
    W.load_into(blank, tensors, index)
    gi = json.loads((out / "golden" / "index.json").read_text())["functions"]["main"]
    golden_in = G.read_tensor(out / "golden" / "main", gi["inputs"][0])
    golden_out = G.read_tensor(out / "golden" / "main", gi["outputs"][0])
    with torch.no_grad():
        assert torch.equal(blank(golden_in).logits, golden_out)


def test_cli_external_weights_slice_names(tmp_path, monkeypatch):
    """A Llama attention slice with external weights: model-relative names for the
    block's tensors, and the captured rotary cos/sin pair as named float constants."""
    pytest.importorskip("torch_mlir")
    from ptqbench import cli
    from ptqbench.data import datasets as ds

    monkeypatch.setattr(ds, "build", fake_stream)
    src = tmp_path / "tiny-llama"
    tiny_llama().save_pretrained(src)
    tiny_tokenizer().save_pretrained(src)
    monkeypatch.setattr("ptqbench.paths.runs_dir", lambda: tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    out = tmp_path / "out"
    argv = [
        "--device", "cpu", "export-mlir", "--model", str(src), "--bits", str(BITS), "--group-size", str(GROUP_SIZE),
        "--seqlen", str(SEQLEN), "--weights", "external", "--slice", "attention", "--block-index", "1", "--out", str(out),
    ]
    assert cli.main(argv) == cli.OK
    index = json.loads((out / "weights.json").read_text())
    entries = {e["name"]: e for e in index["tensors"]}
    assert "model.layers.1.self_attn.q_proj.weight_q" in entries, "named after the model, not the Replay wrapper"
    for name in ("position_embeddings_0", "position_embeddings_1"):  # the rotary cos/sin the block received
        assert entries[name]["role"] == "float" and entries[name]["shape"] == [1, SEQLEN, 16]
    assert "position_ids" not in entries, "unused by the attention, so torch.export drops it and nothing lifts it"
    assert index["functions"]["attention_1"]["n_user_inputs"] == 1
    assert len(index["functions"]["attention_1"]["arguments"]) == len(entries)
    assert "dense_resource" not in (out / "attention_1.mlir").read_text()


def test_cli_external_weights_need_mlir(tmp_path, monkeypatch):
    from ptqbench import cli

    src = tmp_path / "tiny-opt"
    tiny_opt().save_pretrained(src)
    tiny_tokenizer().save_pretrained(src)
    monkeypatch.setattr("ptqbench.paths.runs_dir", lambda: tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    argv = [
        "--device", "cpu", "export-mlir", "--model", str(src), "--bits", str(BITS), "--group-size", str(GROUP_SIZE),
        "--weights", "external", "--no-mlir", "--out", str(tmp_path / "out"),
    ]
    with pytest.raises(ValueError, match="weights external"):
        cli.main(argv)
