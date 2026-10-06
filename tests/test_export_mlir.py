"""The export stage: a quantized model as a torch.nn.Module with explicit dequantization
and as torch-dialect MLIR, bitwise the weights the evaluation ran with.

Everything here runs on a 2-layer OPT built from a config (no download), so the
assertions are about faithfulness, not accuracy: recording changes nothing the
quantizer writes, the QuantizedLinear reproduces the fake-quantized weight bit for
bit, and the MLIR file re-parses with the dequantization ops in it.
"""

from __future__ import annotations

import copy
import json

import pytest
import torch
from transformers import OPTConfig, OPTForCausalLM, PreTrainedTokenizerFast

from ptqbench.export import QuantizedLinear, swap_in_quantized
from ptqbench.export import mlir as M
from ptqbench.models import families
from ptqbench.quantizers import fakequant as fq
from ptqbench.quantizers import gptq, rtn

pytestmark = pytest.mark.smoke
CPU = torch.device("cpu")
VOCAB = 256


def tiny_opt() -> OPTForCausalLM:
    torch.manual_seed(0)
    cfg = OPTConfig(
        vocab_size=VOCAB, hidden_size=64, ffn_dim=128, num_hidden_layers=2, num_attention_heads=4,
        max_position_embeddings=64, word_embed_proj_dim=64, bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    model = OPTForCausalLM(cfg).eval()
    model.config.use_cache = False
    return model


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2, **{f"w{i}": i for i in range(3, VOCAB)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", bos_token="</s>", eos_token="</s>")


@pytest.fixture(scope="module")
def windows() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(3, VOCAB, (4, 32))


@pytest.fixture(scope="module")
def ids() -> torch.Tensor:
    torch.manual_seed(2)
    return torch.randint(3, VOCAB, (1, 16))


def _quantize_twice(fn):
    """Run a quantizer with and without a Recorder on two copies of the tiny model."""
    a, b = tiny_opt(), tiny_opt()
    rec = fq.Recorder(keep_reference=True)
    with torch.no_grad():
        fn(a, rec)
        fn(b, None)
    return a, b, rec


def _assert_faithful(a, b, rec, ids, *, representation):
    """Recording changed nothing; the export module reproduces the evaluated model exactly."""
    ta, tb = families.target_modules(a), families.target_modules(b)
    assert set(rec.records) == set(ta) and len(ta) == 12
    for name in ta:
        assert torch.equal(ta[name].weight, tb[name].weight), f"{name}: recording changed the weight"
    c = copy.deepcopy(a)
    mods = swap_in_quantized(c, rec.records, representation=representation)
    for name, mod in mods.items():
        assert isinstance(mod, QuantizedLinear)
        assert mod.weight_q.dtype is torch.uint8
        assert int(mod.weight_q.max()) <= mod.quant_max
        assert torch.equal(mod.dequantize(), ta[name].weight), f"{name}: dequantize is not bitwise"
        assert torch.equal(rec.records[name].dequantize().to(ta[name].weight.dtype), ta[name].weight)
    with torch.no_grad():
        assert torch.equal(c(ids).logits, a(ids).logits)
    return mods


@pytest.mark.parametrize("representation", ["decomposed", "aten"])
@pytest.mark.parametrize("bits,group_size,sym", [(4, -1, False), (3, 16, True), (8, 32, False), (2, -1, False)])
def test_rtn_export_is_bitwise(bits, group_size, sym, representation, ids):
    a, b, rec = _quantize_twice(lambda m, r: rtn.apply_rtn(m, bits=bits, group_size=group_size, sym=sym, record=r))
    mods = _assert_faithful(a, b, rec, ids, representation=representation)
    expect_scheme = "per_row" if group_size == -1 else "per_group"
    assert {m.scheme for m in mods.values()} == {expect_scheme}
    if representation == "decomposed":
        op = "dequantize_per_channel" if group_size == -1 else "dequantize_per_channel_group"
        assert {m.dequant_op for m in mods.values()} == {f"quantized_decomposed.{op}"}
        assert all(m.zero.dtype is torch.int32 for m in mods.values())
    else:
        assert {m.dequant_op for m in mods.values()} == {"aten"}


def test_codes_match_apply():
    """`record_grid` must round exactly as `_apply` does, grouped or not."""
    torch.manual_seed(3)
    w = torch.randn(16, 64)
    for gs in (-1, 16):
        grid = fq.find_params(w, bits=4, group_size=gs)
        rec = fq.record_grid(w, grid, algo="rtn")
        assert torch.equal(rec.dequantize(), fq.quantize_weight(w, bits=4, group_size=gs, grid=grid))
        assert rec.codes.dtype is torch.uint8 and rec.zero_is_integer
        assert 0 < rec.rel_error < 0.2


@pytest.mark.parametrize(
    "spec,scheme",
    [
        (gptq.GPTQSpec(bits=4), "per_row"),
        (gptq.GPTQSpec(bits=4, group_size=16), "per_group"),
        (gptq.GPTQSpec(bits=4, group_size=16, act_order=True), "per_group_permuted"),
        (gptq.GPTQSpec(bits=4, group_size=16, act_order=True, static_groups=True), "per_group"),
        (gptq.GPTQSpec(bits=3, group_size=16, true_sequential=True), "per_group"),
    ],
)
def test_gptq_export_is_bitwise(spec, scheme, windows, ids):
    a, b, rec = _quantize_twice(
        lambda m, r: gptq.apply_gptq(m, windows, spec=spec, device=CPU, progress=False, record=r)
    )
    mods = _assert_faithful(a, b, rec, ids, representation="decomposed")
    assert {m.scheme for m in mods.values()} == {scheme}
    if scheme == "per_group_permuted":
        # act_order scatters a group's columns: no quantized_decomposed op takes a
        # per-column group index, so these layers are explicit ATen ops.
        assert all(m.g_idx is not None and m.dequant_op == "aten" for m in mods.values())
        g = next(iter(mods.values())).g_idx
        assert not torch.equal(g, torch.arange(g.numel()) // 16)
    else:
        assert all(m.g_idx is None for m in mods.values())


def test_awq_lite_export_is_bitwise(windows, ids):
    from ptqbench.quantizers import awq_lite

    spec = awq_lite.AWQSpec(bits=4, group_size=16)
    a, b, rec = _quantize_twice(lambda m, r: awq_lite.apply_awq_lite(m, windows, spec=spec, device=CPU, record=r))
    mods = _assert_faithful(a, b, rec, ids, representation="decomposed")
    assert {r.algo for r in rec.records.values()} == {"awq_lite"}
    assert {m.dequant_op for m in mods.values()} == {"quantized_decomposed.dequantize_per_channel_group"}


@pytest.mark.parametrize("bits,group_size", [(4, 16), (3, -1), (2, 32)])
def test_hqq_export_is_bitwise(bits, group_size, ids, monkeypatch):
    pytest.importorskip("hqq.core.quantize")
    from ptqbench.quantizers import hqq_adapter as H

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    a, b, rec = _quantize_twice(lambda m, r: H.apply_hqq(m, bits=bits, group_size=group_size, device=CPU, record=r))
    mods = _assert_faithful(a, b, rec, ids, representation="decomposed")
    # hqq's zero-point is a float: representable only as explicit ATen ops.
    assert all(not r.zero_is_integer for r in rec.records.values())
    assert {m.dequant_op for m in mods.values()} == {"aten"}
    assert {m.zero_point_domain for m in mods.values()} == {"float"}


def test_mlir_round_trip(tmp_path, ids):
    pytest.importorskip("torch_mlir")
    model = tiny_opt()
    rec = fq.Recorder()
    with torch.no_grad():
        rtn.apply_rtn(model, bits=4, group_size=16, record=rec)
    swap_in_quantized(model, rec.records)

    module, stats = M.export_torch_dialect(model, ids)
    path = M.write_module(module, tmp_path / "model.mlir")
    text = path.read_text()
    assert text.startswith("module") and "func.func @main" in text
    counts = M.count_dequant_ops(text)
    assert counts["torch.quantized_decomposed.dequantize_per_channel_group"] == 12
    assert counts["torch.operator"] == 0, "every op imported as a first-class torch-dialect op"
    assert "ui8" in text and "dense_resource<torch_tensor_" in text
    assert stats["literals"] >= 12 and stats["torch_mlir_version"]

    reparsed = M.parse(path)  # parses and verifies, or raises
    assert reparsed.operation.verify()
    bc = M.write_module(module, tmp_path / "model.mlirbc", bytecode=True)
    assert M.parse(bc).operation.verify()

    # And torch-mlir can take it the rest of the way: backend contract, then linalg.
    from torch_mlir import fx

    lowered = fx.export_and_import(model, ids, output_type="linalg-on-tensors", func_name="main")
    lowered_text = str(lowered)
    assert "linalg.matmul" in lowered_text and "dequantize" not in lowered_text


def test_mlir_dynamic_seqlen(ids):
    pytest.importorskip("torch_mlir")
    model = tiny_opt()
    rec = fq.Recorder()
    with torch.no_grad():
        rtn.apply_rtn(model, bits=4, record=rec)
    swap_in_quantized(model, rec.records)
    module, stats = M.export_torch_dialect(model, ids, dynamic_seqlen=True, max_seqlen=64)
    assert stats["dynamic_seqlen"]
    assert "!torch.vtensor<[1,?],si64>" in str(module)


def test_cli_export_mlir(tmp_path, monkeypatch):
    """`ptq export-mlir` on a local directory: files, JSON keyed by module name, bitwise."""
    from ptqbench import cli

    src = tmp_path / "tiny-opt"
    tiny_opt().save_pretrained(src)
    tiny_tokenizer().save_pretrained(src)
    monkeypatch.setattr("ptqbench.paths.runs_dir", lambda: tmp_path / "runs")  # no result rows to join
    (tmp_path / "runs").mkdir()
    out = tmp_path / "out"
    have_mlir = M.available()
    argv = ["--device", "cpu", "export-mlir", "--model", str(src), "--bits", "4", "--group-size", "16", "--out", str(out)]
    if not have_mlir:
        argv.append("--no-mlir")
    assert cli.main(argv) == cli.OK

    layers = json.loads((out / "layers.json").read_text())
    expected = {f"model.decoder.layers.{i}.{n}" for i in range(2) for n in families.OPT.targets}
    assert set(layers) == expected
    for entry in layers.values():
        assert entry["bits"] == 4 and entry["group_size"] == 16 and entry["scheme"] == "per_group"
        assert entry["bitwise_equal_to_evaluated"] is True and entry["roundtrip_max_abs_diff"] == 0.0
        assert 0 < entry["rel_error"] < 0.5 and entry["codes_used"] <= 16
    manifest = json.loads((out / "export.json").read_text())
    assert manifest["n_quantized_modules"] == 12 and manifest["algo"] == "rtn"
    assert manifest["summary"]["all_bitwise_equal_to_evaluated"] is True
    assert manifest["dtype"] == "torch.float32" and manifest["perplexity_rows"] == []
    if have_mlir:
        mlir = out / "model.mlir"
        assert mlir.is_file() and manifest["files"]["mlir"] == "model.mlir"
        assert manifest["mlir"]["ops"]["torch.quantized_decomposed.dequantize_per_channel_group"] == 12
        M.parse(mlir)
    else:
        assert manifest["files"]["mlir"] is None


def test_cli_export_without_torch_mlir_is_a_clear_failure(tmp_path, monkeypatch, capsys):
    from ptqbench import cli

    monkeypatch.setattr(M, "available", lambda: False)
    rc = cli.main(["export-mlir", "--model", "x", "--bits", "4", "--out", str(tmp_path)])
    assert rc == cli.FAILED_CHECK
    assert "uv sync --extra mlir" in capsys.readouterr().err
