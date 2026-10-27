"""Per-layer sensitivity: tables and heatmap from synthetic exports (export.json +
layers.json), the metric fallback, and the CLI."""

from __future__ import annotations

import json

import pytest

from ptqbench import paths
from ptqbench.analysis import sensitivity as S

pytestmark = pytest.mark.smoke

BLOCKS = 2
PROJECTIONS = ("self_attn.q_proj", "self_attn.k_proj", "mlp.down_proj")


def _export(root, algo, bits, *, scored=True, model="org/tiny-125m", key="tiny-125m", prefix="model.layers"):
    """One fake export: the error grows with the block index and shrinks with bits;
    down_proj is the most sensitive projection."""
    d = root / f"{key}-{algo}{bits}"
    d.mkdir(parents=True)
    quant = {"algo": algo, "bits": bits, "group_size": 16, "act_order": algo == "gptq"}
    (d / "export.json").write_text(json.dumps({"model": model, "model_key": key, "quant": quant}))
    layers = {}
    for b in range(BLOCKS):
        for i, proj in enumerate(PROJECTIONS):
            err = (1 + b) * (1 + i) * 0.01 * (16 / bits)
            layers[f"{prefix}.{b}.{proj}"] = {
                "rel_error": err / 2, "snr_db": 30.0, "shape": [64, 64],
                **({"output_rel_error": err} if scored else {}),
            }
    (d / "layers.json").write_text(json.dumps(layers))
    return d / "layers.json"


def test_tables_name_the_sensitive_layers_and_order_configs(tmp_path):
    a = _export(tmp_path, "rtn", 4)
    b = _export(tmp_path, "rtn", 3)
    c = _export(tmp_path, "gptq", 4)
    result = S.run([b, c, a], out_dir=tmp_path / "sens", plots_dir=tmp_path / "plots")
    assert result["model_key"] == "tiny-125m" and result["metric"] == S.METRIC_OUTPUT
    assert result["n_configs"] == 3 and result["n_layers"] == 3 * BLOCKS * len(PROJECTIONS)
    md = result["md"].read_text(encoding="utf-8")
    assert "## Average over blocks, per projection" in md
    # method order first, then bits descending: rtn 4, rtn 3, gptq-ao 4
    assert md.index("## rtn 4-bit g16") < md.index("## rtn 3-bit g16") < md.index("## gptq-ao 4-bit g16")
    assert "Most sensitive layers: block 1 `down_proj`" in md
    means = result["means"]
    assert list(means.columns) == ["q_proj", "k_proj", "down_proj"]
    assert means.loc["rtn 3-bit g16", "down_proj"] > means.loc["rtn 4-bit g16", "down_proj"]
    assert result["png"].is_file() and result["png"].stat().st_size > 5_000
    assert result["csv"].read_text().count("\n") == result["n_layers"] + 1


def test_weight_error_fallback_and_opt_names(tmp_path):
    path = _export(tmp_path, "rtn", 4, scored=False, prefix="model.decoder.layers")
    key, df = S.profile([path])
    assert key == "tiny-125m" and S.metric_for(df) == S.METRIC_WEIGHT
    assert sorted(df["projection"].unique()) == ["down_proj", "k_proj", "q_proj"] and df["block"].max() == BLOCKS - 1
    _csv, md = S.write_tables(key, df, tmp_path / "out")
    assert "without `--score-windows`" in md.read_text(encoding="utf-8")


def test_one_model_at_a_time(tmp_path):
    a = _export(tmp_path, "rtn", 4)
    b = _export(tmp_path / "other", "rtn", 4, model="org/other-350m", key="other-350m")
    with pytest.raises(ValueError, match="one model at a time"):
        S.profile([a, b])


def test_cli_writes_the_three_files(tmp_path, monkeypatch, capsys):
    from ptqbench import cli

    monkeypatch.setattr(paths, "plots_dir", lambda: tmp_path / "plots")
    a = _export(tmp_path, "rtn", 4)
    b = _export(tmp_path, "rtn", 2)
    assert cli.main(["sensitivity", str(a), str(b), "--out", str(tmp_path / "sens")]) == cli.OK
    out = capsys.readouterr().out
    assert "2 configuration(s)" in out and "output error on real text" in out and "down_proj" in out
    assert (tmp_path / "sens" / "tiny-125m.md").is_file() and (tmp_path / "plots" / "tiny-125m_sensitivity.png").is_file()
