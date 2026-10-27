"""The charts carry their reading guide: plain dataset names, 'lower is better', the
within-5% band, and a caption, on a tiny synthetic results table."""

from __future__ import annotations

import pandas as pd
import pytest

from ptqbench.analysis import plots as P

pytestmark = pytest.mark.smoke


def _rows():
    out = []
    for dataset in ("wikitext2", "c4_new"):
        out.append({"model": "org/tiny-125m", "dataset": dataset, "algo": "fp", "bits": 16, "group_size": -1,
                    "ppl": 20.0, "status": "ok", "partial": False, "paper_ppl": None, "delta_vs_fp16": None})
        for algo, base in (("rtn", 21.0), ("gptq", 20.5)):
            for bits, mult in ((8, 1.0), (4, 1.1), (3, 1.8), (2, 20.0)):
                out.append({"model": "org/tiny-125m", "dataset": dataset, "algo": algo, "bits": bits, "group_size": 128,
                            "ppl": base * mult, "status": "ok", "partial": False, "paper_ppl": 20.6 if bits == 4 else None,
                            "delta_vs_fp16": base * mult - 20.0})
    return pd.DataFrame(out)


def _sweep_rows():
    """A group-size sweep with sizes: six groupings at 4 and 3 bits, two methods, the
    fp16 row, and one setting far enough off to fall outside the frontier chart."""
    out = [{"model": "org/tiny-125m", "dataset": "wikitext2", "algo": "fp", "bits": 16, "group_size": -1, "ppl": 20.0,
            "status": "ok", "partial": False, "paper_ppl": None, "delta_vs_fp16": None,
            "bits_per_weight": 16.0, "weight_bytes": 200_000_000}]
    for algo, base in (("rtn", 21.0), ("gptq", 20.5)):
        for bits, mult in ((4, 1.1), (3, 1.8)):
            for gs in (-1, 192, 96, 64, 48, 32):
                bpw = bits + (0 if gs == -1 else 20 / gs)
                out.append({"model": "org/tiny-125m", "dataset": "wikitext2", "algo": algo, "bits": bits, "group_size": gs,
                            "ppl": base * mult * (1 + (gs if gs > 0 else 600) / 2000), "status": "ok", "partial": False,
                            "paper_ppl": None, "delta_vs_fp16": 1.0, "bits_per_weight": bpw,
                            "weight_bytes": int(200_000_000 * bpw / 16)})
    out.append({"model": "org/tiny-125m", "dataset": "wikitext2", "algo": "hqq", "bits": 2, "group_size": 64, "ppl": 9_000.0,
                "status": "ok", "partial": False, "paper_ppl": None, "delta_vs_fp16": 8_980.0,
                "bits_per_weight": 2.5, "weight_bytes": 31_250_000})
    return pd.DataFrame(out)


def test_frontier_and_groupsize_render_with_captions(tmp_path, monkeypatch):
    captions: list[str] = []
    import matplotlib.figure

    real_text = matplotlib.figure.Figure.text

    def spy(self, *args, **kwargs):
        captions.append(args[2] if len(args) > 2 else kwargs.get("s", ""))
        return real_text(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "text", spy)
    P._style()
    ok = P._complete(_sweep_rows())
    f = P.plot_frontier(ok, "org/tiny-125m", tmp_path / "f.png")
    g = P.plot_groupsize(ok, "org/tiny-125m", tmp_path / "g.png")
    assert f and f.stat().st_size > 10_000 and g and g.stat().st_size > 10_000
    assert any("Lower-left is better" in c and "frontier" in c for c in captions)
    assert any("Smaller groups" in c and "bits per weight" in c for c in captions)
    # The old-style table (no sizes, three groupings at most) draws neither.
    plain = P._complete(_rows())
    assert P.plot_frontier(plain, "org/tiny-125m", tmp_path / "none.png") is None
    assert P.plot_groupsize(plain, "org/tiny-125m", tmp_path / "none2.png") is None
    assert P._gs_label(-1) == "per-row" and P._gs_label(128) == "g128" and P._gs_label(48) == "g48"


def test_both_figures_render_with_captions(tmp_path, monkeypatch):
    captions: list[str] = []
    import matplotlib.figure

    real_text = matplotlib.figure.Figure.text

    def spy(self, *args, **kwargs):
        captions.append(args[2] if len(args) > 2 else kwargs.get("s", ""))
        return real_text(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "text", spy)
    P._style()
    ok = P._complete(_rows())
    p1 = P.plot_perplexity(ok, "org/tiny-125m", tmp_path / "p.png")
    p2 = P.plot_delta_heatmap(ok, "org/tiny-125m", tmp_path / "d.png")
    assert p1 and p1.stat().st_size > 10_000 and p2 and p2.stat().st_size > 5_000
    assert any("Lower is better" in c and "within 5%" in c for c in captions)
    assert any("how much perplexity rose" in c for c in captions)
    assert P.DATASET_TITLE["wikitext2"] == "WikiText-2 (Wikipedia)"
