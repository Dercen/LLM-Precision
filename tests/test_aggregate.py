"""results/by-model/ pages: one per model, readable labels, sensible comparisons."""

from __future__ import annotations

import pandas as pd
import pytest

from ptqbench.analysis import aggregate as A

pytestmark = pytest.mark.smoke


def _row(model, algo, bits, gs, ppl, dataset="wikitext2", status="ok"):
    return {
        "model": model, "dataset": dataset, "algo": algo, "bits": bits, "group_size": gs,
        "ppl": ppl, "partial": False, "status": status, "reason": None if status == "ok" else "oom",
        "paper_ppl": None, "paper_source": None, "delta_vs_paper_pct": None,
    }


@pytest.fixture
def df():
    return pd.DataFrame([
        _row("org/big-1.3b", "fp", 16, -1, 10.0),
        _row("org/big-1.3b", "rtn", 4, -1, 12.0),
        _row("org/big-1.3b", "gptq", 4, 128, 10.5),
        _row("org/big-1.3b", "rtn", 2, -1, 500.0),
        _row("org/big-1.3b", "hqq", 3, -1, None, status="failed"),
        _row("org/small-125m", "fp", 16, -1, 20.0),
    ])


def test_worse_formats_percent_then_multiple():
    assert A._worse(10.5, 10.0) == "+5.0%"
    assert A._worse(9.9, 10.0) == "-1.0%"
    assert A._worse(35.0, 10.0) == "3.5× worse"
    assert A._worse(500.0, 10.0) == "50× worse"
    assert A._worse(10.0, None) == ""


def test_one_page_per_model_and_index_smallest_first(df, tmp_path):
    written = A.write_by_model(df, tmp_path)
    assert sorted(p.name for p in written) == ["big-1.3b.md", "small-125m.md"]
    index = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert index.index("small-125m") < index.index("big-1.3b")


def test_page_picks_best_method_and_lists_failures(df, tmp_path):
    A.write_by_model(df, tmp_path)
    page = (tmp_path / "big-1.3b.md").read_text(encoding="utf-8")
    best = page.split("## Best result")[1].split("##")[0]
    assert "| 4 | GPTQ | groups of 128 | 10.5000 | +5.0% |" in best
    assert "RTN (plain rounding) | per row" in page
    assert "`oom` (1)" in page


def test_table_carries_sizes_when_the_config_is_known(monkeypatch):
    from ptqbench.analysis import footprint as F
    from tests.tiny_models import tiny_opt

    tiny = tiny_opt().config
    monkeypatch.setattr(F, "load_config", lambda repo, revision=None: tiny if repo == "org/tiny-opt" else (_ for _ in ()).throw(OSError(repo)))
    rows = [
        {"run_id": "a", "model": "org/tiny-opt", "dataset": "wikitext2", "algo": "fp", "bits": 16, "group_size": -1,
         "ppl": 20.0, "status": "ok", "partial": False, "dtype": "torch.float16"},
        {"run_id": "b", "model": "org/tiny-opt", "dataset": "wikitext2", "algo": "rtn", "bits": 4, "group_size": 16,
         "ppl": 22.0, "status": "ok", "partial": False, "dtype": "torch.float16"},
        {"run_id": "c", "model": "org/unknown", "dataset": "wikitext2", "algo": "rtn", "bits": 4, "group_size": 16,
         "ppl": 22.0, "status": "ok", "partial": False},
        {"run_id": "d", "model": "org/tiny-opt", "dataset": "wikitext2", "algo": "rtn", "bits": 3, "group_size": 16,
         "ppl": None, "status": "failed", "partial": False, "reason": "oom"},
    ]
    df = A.build_table(rows).set_index("run_id")
    assert df.loc["a", "bits_per_weight"] == 16.0 and df.loc["a", "weight_bytes"] == 87680 * 2
    assert df.loc["b", "bits_per_weight"] == pytest.approx(4 + 20 / 16) and df.loc["b", "weight_bytes"] < df.loc["a", "weight_bytes"]
    assert pd.isna(df.loc["c", "bits_per_weight"]) and pd.isna(df.loc["d", "weight_bytes"])


def test_page_shows_sizes_when_present(tmp_path):
    rows = pd.DataFrame([
        {**_row("org/big-1.3b", "fp", 16, -1, 10.0), "bits_per_weight": 16.0, "weight_bytes": 2_600_000_000},
        {**_row("org/big-1.3b", "rtn", 4, 128, 11.0), "bits_per_weight": 4.15625, "weight_bytes": 900_000_000},
    ])
    A.write_by_model(rows, tmp_path)
    page = (tmp_path / "big-1.3b.md").read_text(encoding="utf-8")
    assert "take 2,600 MB at 16 bits" in page
    assert "| 4 | RTN (plain rounding) | groups of 128 | 11.0000 | +10.0% | 4.16 | 900 |" in page
