"""Golden test vectors (item 2 of the vault hand-off): tensors land on disk as .npy and
raw little-endian .bin with an index that describes them, inputs are real seeded text
(a fake stream stands in for wikitext2 so nothing downloads), the tolerance is derived
from a float64 rerun, and the CLI wires it all up.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from ptqbench.data import datasets as ds
from ptqbench.export import golden as G
from ptqbench.export import swap_in_quantized
from ptqbench.quantizers import fakequant as fq
from ptqbench.quantizers import rtn
from tests.tiny_models import GROUP_SIZE, VOCAB, tiny_opt, tiny_tokenizer

pytestmark = pytest.mark.smoke

SEQLEN = 16
BITS = 4
#: The fake stream holds this many source windows, so different seeds can pick different ones.
FAKE_WINDOWS = 4


def quantized_tiny():
    model = tiny_opt()
    rec = fq.Recorder()
    with torch.no_grad():
        rtn.apply_rtn(model, bits=BITS, group_size=GROUP_SIZE, record=rec)
    swap_in_quantized(model, rec.records)
    return model


def fake_stream(key, tokenizer, seqlen=G.GOLDEN_SOURCE_SEQLEN) -> ds.TokenStream:
    """Stands in for `datasets.build`: deterministic ids in the tiny vocabulary, cut
    into `FAKE_WINDOWS` windows of the real source length."""
    ids = (torch.arange(FAKE_WINDOWS * seqlen) * 7 % (VOCAB - 3) + 3).unsqueeze(0)
    return ds.TokenStream(key=key, ids=ids, seqlen=seqlen, split="fake", join="fake", n_source_rows=1)


@pytest.fixture
def no_download(monkeypatch):
    monkeypatch.setattr(ds, "build", fake_stream)


def test_write_and_read_tensor(tmp_path):
    cases = {
        "f32": torch.randn(3, 5), "i64": torch.arange(12).reshape(2, 6), "f16": torch.randn(4).half(),
        "bf16": torch.randn(2, 2).bfloat16(), "scalar": torch.tensor(7),
    }
    for name, t in cases.items():
        entry = G.write_tensor(tmp_path, name, t, role=G.ROLE_INPUT)
        assert entry["shape"] == list(t.shape) and entry["byte_length"] == t.numel() * t.element_size()
        assert entry["dtype"] == str(t.dtype).replace("torch.", "") and entry["role"] == "input"
        raw = (tmp_path / entry["bin"]).read_bytes()
        assert len(raw) == entry["byte_length"] and G.F.sha256_of(raw) == entry["sha256"]
        from_npy = np.load(tmp_path / entry["npy"])
        assert from_npy.tobytes() == raw, "the .npy and the .bin hold the same bytes"
        assert from_npy.dtype.byteorder in ("<", "|", "=") and (from_npy.dtype.byteorder != "=" or np.little_endian)
        assert torch.equal(G.read_tensor(tmp_path, entry), t)
    assert cases["bf16"].element_size() == 2 and G.write_tensor(tmp_path, "b", cases["bf16"], role="input")["storage_dtype"] == "uint16"


def test_realistic_window_is_seeded_text(no_download):
    tok = tiny_tokenizer()
    ids, source = G.realistic_window(tok, "wikitext2", seed=0, length=SEQLEN)
    again, _ = G.realistic_window(tok, "wikitext2", seed=0, length=SEQLEN)
    assert ids.shape == (1, SEQLEN) and ids.dtype is torch.int64 and torch.equal(ids, again)
    stream = fake_stream("wikitext2", tok)
    assert torch.equal(ids, stream.window(source["window_index"])[:, :SEQLEN]), "a prefix of a real window"
    assert source["dataset"] == "wikitext2" and source["n_windows"] == FAKE_WINDOWS and source["length"] == SEQLEN
    assert source["first_token_hash"] == stream.first_token_hash()
    others = {G.realistic_window(tok, "wikitext2", seed=s, length=SEQLEN)[1]["window_index"] for s in range(8)}
    assert len(others) > 1, "the seed picks the window"
    with pytest.raises(ValueError, match="exceeds"):
        G.realistic_window(tok, "wikitext2", seed=0, length=G.GOLDEN_SOURCE_SEQLEN + 1)


def test_tolerance_is_derived_from_a_float64_rerun(no_download):
    model = quantized_tiny()
    ids, source = G.realistic_window(tiny_tokenizer(), "wikitext2", seed=0, length=SEQLEN)
    case = G.build_case("main", model, [("input_ids", ids)], ["logits"], source=source)
    assert [n for n, _ in case.outputs] == ["logits"]
    with torch.no_grad():
        assert torch.equal(case.outputs[0][1], model(ids).logits)
    (tol,) = case.tolerance
    err, big = tol["max_abs_err_vs_fp64"], tol["max_abs_value"]
    assert 0 < err < 1e-4, "fp32 rounding is small but not zero"
    assert 0 < tol["rel_fro_err_vs_fp64"] < 1e-4
    assert tol["atol"] == max(G.TOLERANCE_SAFETY_FACTOR * err, G.TOLERANCE_SAFETY_FACTOR * G.FP32_EPSILON * big)
    assert tol["rel_fro_tol"] == G.TOLERANCE_SAFETY_FACTOR * tol["rel_fro_err_vs_fp64"]
    assert tol["argmax_agreement"] == 1.0
    # The float64 copy really holds the same weights: its Linear equals the dequantized one.
    ref = G.fp64_reference(model)
    q = model.model.decoder.layers[0].fc1
    assert ref.model.decoder.layers[0].fc1.weight.dtype is torch.float64
    assert torch.equal(ref.model.decoder.layers[0].fc1.weight.float(), q.dequantize())
    # The consumer-side check: the reference passes its own tolerance, a perturbed copy does not.
    logits = case.outputs[0][1]
    assert G.check_within(logits, logits, tol)["within_tolerance"]
    nudged = logits.clone()
    nudged[0, 0, 0] += 10 * tol["atol"]
    assert not G.check_within(nudged, logits, tol)["within_tolerance"]


def test_write_golden_index(tmp_path, no_download):
    model = quantized_tiny()
    ids, source = G.realistic_window(tiny_tokenizer(), "wikitext2", seed=1, length=SEQLEN)
    case = G.build_case("main", model, [("input_ids", ids)], ["logits"], source=source)
    path, index = G.write_golden(tmp_path, [case])
    assert path == tmp_path / "golden" / "index.json" and json.loads(path.read_text()) == json.loads(json.dumps(index))
    fn = index["functions"]["main"]
    assert [e["name"] for e in fn["inputs"]] == ["input_ids"] and [e["name"] for e in fn["outputs"]] == ["logits"]
    assert fn["outputs"][0]["shape"] == [1, SEQLEN, VOCAB] and fn["outputs"][0]["dtype"] == "float32"
    assert fn["outputs"][0]["byte_length"] == SEQLEN * VOCAB * 4 and index["byte_order"] == "little"
    assert index["tolerance_policy"]["safety_factor"] == G.TOLERANCE_SAFETY_FACTOR
    directory = tmp_path / "golden" / fn["directory"]
    assert torch.equal(G.read_tensor(directory, fn["inputs"][0]), ids)
    assert torch.equal(G.read_tensor(directory, fn["outputs"][0]), case.outputs[0][1])
    assert G.summary(index)["main"]["worst_output"] == "logits"


def test_cli_golden(tmp_path, monkeypatch, no_download):
    """`ptq export-mlir --golden`: the vectors, the index, and export.json's summary."""
    from ptqbench import cli
    from ptqbench.export import mlir as M

    src = tmp_path / "tiny-opt"
    tiny_opt().save_pretrained(src)
    tiny_tokenizer().save_pretrained(src)
    monkeypatch.setattr("ptqbench.paths.runs_dir", lambda: tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    out = tmp_path / "out"
    argv = [
        "--device", "cpu", "export-mlir", "--model", str(src), "--bits", str(BITS), "--group-size", str(GROUP_SIZE),
        "--seqlen", str(SEQLEN), "--golden", "--golden-seed", "2", "--out", str(out),
    ]
    if not M.available():
        argv.append("--no-mlir")
    assert cli.main(argv) == cli.OK
    manifest = json.loads((out / "export.json").read_text())
    assert manifest["files"]["golden"] == "golden/index.json"
    g = manifest["golden"]
    assert g["seed"] == 2 and g["dataset"] == "wikitext2" and g["functions"]["main"]["worst_output"] == "logits"
    assert 0 < g["functions"]["main"]["max_abs_err_vs_fp64"] < g["functions"]["main"]["atol"]
    index = json.loads((out / "golden" / "index.json").read_text())
    fn = index["functions"]["main"]
    assert fn["source"]["seed"] == 2 and fn["source"]["length"] == SEQLEN
    for entry in fn["inputs"] + fn["outputs"]:
        f = out / "golden" / "main" / entry["bin"]
        assert f.stat().st_size == entry["byte_length"]
        assert np.load(out / "golden" / "main" / entry["npy"]).tobytes() == f.read_bytes()
    # The golden input is the fake stream's text, not noise, and the output is the model's.
    ids = G.read_tensor(out / "golden" / "main", fn["inputs"][0])
    assert torch.equal(ids, fake_stream("wikitext2", tiny_tokenizer()).window(fn["source"]["window_index"])[:, :SEQLEN])
    logits = G.read_tensor(out / "golden" / "main", fn["outputs"][0])
    model = quantized_tiny()  # same seed, same quantizer, same tiny model as the CLI loaded from `src`
    with torch.no_grad():
        assert torch.equal(logits, model(ids).logits)
