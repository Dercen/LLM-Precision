"""`ptq plot`: static figures from results.csv. PLAN.md 9.5.

Two figures per model:

* perplexity: three panels (wikitext2, c4_new, ptb_new), bits categorical on x, log-y
  perplexity, one line per algorithm in a FIXED colour (slot order rtn, gptq, awq_lite,
  hqq -- colour follows the entity, never the row count), solid for g128, dashed for
  per-row, dotted for g64; the fp16 baseline as a muted horizontal rule; published
  numbers as hollow markers in the series colour.
* delta heatmap: (algorithm, group size) rows by bits, coloured by log10(1 + ppl - fp16)
  on a single-hue sequential ramp, the value printed in each cell in ink chosen by the
  fill's luminance.

Colours are the validated reference palette; text always wears text tokens, marks
carry the series colour; gridlines are solid hairlines. results.csv and summary.md are
the table twins of every figure. Light mode only: these are export figures.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .. import paths

# Reference palette (light). Categorical slots in fixed order; algorithm -> slot.
SERIES = {"rtn": "#2a78d6", "gptq": "#eb6834", "awq_lite": "#1baf7a", "hqq": "#eda100"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781"}
CHROME = {"surface": "#fcfcfb", "page": "#f9f9f7", "grid": "#e1e0d9", "axis": "#c3c2b7"}
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
BITS = [8, 4, 3, 2]
DATASETS = ["wikitext2", "c4_new", "ptb_new"]
GS_STYLE = {128: "-", -1: "--", 64: ":"}
GS_LABEL = {128: "g128", -1: "per-row", 64: "g64"}
ALGO_LABEL = {"rtn": "RTN", "gptq": "GPTQ", "awq_lite": "AWQ-lite", "hqq": "HQQ"}


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Helvetica", "Arial"],
        "axes.facecolor": CHROME["surface"], "figure.facecolor": CHROME["page"],
        "axes.edgecolor": CHROME["axis"], "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": CHROME["grid"], "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK["muted"], "ytick.color": INK["muted"], "axes.labelcolor": INK["secondary"],
        "text.color": INK["primary"], "axes.titlecolor": INK["primary"],
        "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False,
        "font.size": 10, "axes.titlesize": 11, "legend.fontsize": 9,
    })


def _complete(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[(df["status"] == "ok") & (~df["partial"].astype(bool)) & df["ppl"].notna()].copy()
    ok["group_size"] = ok["group_size"].fillna(-1).astype(int)
    ok["bits"] = ok["bits"].astype(int)
    return ok


def _fmt_ppl(v: float) -> str:
    if v >= 100:
        return f"{v:,.0f}"
    return f"{v:.1f}" if v >= 10 else f"{v:.2f}"


def _luminance(hex_color: str) -> float:
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def plot_perplexity(ok: pd.DataFrame, model: str, out: Path) -> Path | None:
    sub = ok[ok["model"] == model]
    if sub.empty:
        return None
    datasets = [d for d in DATASETS if d in set(sub["dataset"])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 4.2), squeeze=False)
    handles: dict[str, Line2D] = {}
    for ax, dataset in zip(axes[0], datasets, strict=True):
        d = sub[sub["dataset"] == dataset]
        fp = d[d["algo"] == "fp"]["ppl"]
        ax.set_xticks(range(len(BITS)))
        ax.set_xticklabels([f"{b}-bit" for b in BITS])
        ax.set_yscale("log")
        ax.grid(axis="x", visible=False)
        if not fp.empty:
            ax.axhline(fp.iloc[0], color=CHROME["axis"], linewidth=1.2, zorder=1)
            ax.annotate(f"fp16 {fp.iloc[0]:.2f}", xy=(len(BITS) - 1 + 0.3, fp.iloc[0]),
                        xytext=(0, 3), textcoords="offset points", ha="right", va="bottom",
                        fontsize=8, color=INK["secondary"])
        end_labels: list[tuple[int, float, str]] = []
        for algo, colour in SERIES.items():
            for gs, ls in GS_STYLE.items():
                q = d[(d["algo"] == algo) & (d["group_size"] == gs)].sort_values("bits", ascending=False)
                if q.empty:
                    continue
                xs = [BITS.index(b) for b in q["bits"] if b in BITS]
                ys = [float(p) for b, p in zip(q["bits"], q["ppl"], strict=True) if b in BITS]
                line, = ax.plot(xs, ys, ls, color=colour, linewidth=2, marker="o", markersize=8,
                                markerfacecolor=colour, markeredgecolor=CHROME["surface"], markeredgewidth=2,
                                solid_capstyle="round", zorder=3)
                handles.setdefault(f"{algo}|{gs}", line)
                paper = q[q["paper_ppl"].notna()]
                if not paper.empty:
                    ax.plot([BITS.index(b) for b in paper["bits"]], paper["paper_ppl"].astype(float), "o",
                            markersize=8, markerfacecolor=CHROME["surface"], markeredgecolor=colour,
                            markeredgewidth=1.6, linestyle="none", zorder=4)
                if xs and (gs == 128 or (gs == -1 and not (d["group_size"] == 128).any())):
                    end_labels.append((xs[-1], ys[-1], ALGO_LABEL[algo]))
        # Selective direct labels at each line's real endpoint, skipping collisions.
        placed: list[tuple[int, float]] = []
        for x, y, text in sorted(end_labels, key=lambda t: t[1]):
            if any(px == x and abs(math.log10(y) - math.log10(py)) < 0.06 for px, py in placed):
                continue
            placed.append((x, y))
            ax.annotate(text, xy=(x, y), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK["secondary"])
        ax.set_title(dataset, loc="left")
        ax.set_xlim(-0.4, len(BITS) - 0.4)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _fmt_ppl(v)))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0][0].set_ylabel("perplexity (log)")
    legend_items = [handles[k] for k in sorted(handles, key=lambda k: (list(SERIES).index(k.split("|")[0]), k))]
    legend_labels = [f"{ALGO_LABEL[k.split('|')[0]]} {GS_LABEL[int(k.split('|')[1])]}" for k in sorted(handles, key=lambda k: (list(SERIES).index(k.split("|")[0]), k))]
    legend_items.append(Line2D([], [], marker="o", linestyle="none", markersize=8, markerfacecolor=CHROME["surface"], markeredgecolor=INK["muted"], markeredgewidth=1.6))
    legend_labels.append("published")
    fig.legend(legend_items, legend_labels, loc="lower center", ncol=min(6, len(legend_items)), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{model} — perplexity by weight precision", x=0.01, ha="left", fontsize=12, color=INK["primary"])
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_delta_heatmap(ok: pd.DataFrame, model: str, out: Path) -> Path | None:
    sub = ok[(ok["model"] == model) & (ok["algo"] != "fp") & ok["delta_vs_fp16"].notna()]
    if sub.empty:
        return None
    datasets = [d for d in DATASETS if d in set(sub["dataset"])]
    rows = sorted({(a, g) for a, g in zip(sub["algo"], sub["group_size"], strict=True)},
                  key=lambda t: (list(SERIES).index(t[0]) if t[0] in SERIES else 9, -t[1]))
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets) + 0.8, 0.42 * len(rows) + 1.6), squeeze=False)
    vmax = math.log10(1 + max(1.0, float(sub["delta_vs_fp16"].max())))
    for ax, dataset in zip(axes[0], datasets, strict=True):
        d = sub[sub["dataset"] == dataset]
        ax.set_xlim(0, len(BITS))
        ax.set_ylim(0, len(rows))
        ax.invert_yaxis()
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for ri, (algo, gs) in enumerate(rows):
            for ci, bits in enumerate(BITS):
                cell = d[(d["algo"] == algo) & (d["group_size"] == gs) & (d["bits"] == bits)]
                if cell.empty:
                    continue
                delta = float(cell["delta_vs_fp16"].iloc[0])
                t = max(0.0, math.log10(1 + max(delta, 0.0))) / vmax if vmax > 0 else 0.0
                colour = SEQ[min(len(SEQ) - 1, round(t * (len(SEQ) - 1)))]
                ax.add_patch(plt.Rectangle((ci + 0.03, ri + 0.03), 0.94, 0.94, facecolor=colour, edgecolor="none"))
                ink = "#ffffff" if _luminance(colour) < 0.45 else INK["primary"]
                ax.text(ci + 0.5, ri + 0.5, f"+{_fmt_ppl(delta)}" if delta >= 0 else f"{delta:.2f}",
                        ha="center", va="center", fontsize=8.5, color=ink)
        ax.set_xticks([i + 0.5 for i in range(len(BITS))])
        ax.set_xticklabels([f"{b}-bit" for b in BITS])
        ax.set_yticks([i + 0.5 for i in range(len(rows))])
        ax.set_yticklabels([f"{ALGO_LABEL.get(a, a)} {GS_LABEL.get(g, g)}" for a, g in rows] if ax is axes[0][0] else [])
        ax.set_title(dataset, loc="left")
    fig.suptitle(f"{model} — perplexity increase over fp16 (log colour scale)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_all(models: list[str] | None = None, csv_path: Path | None = None) -> list[Path]:
    _style()
    csv_path = csv_path or paths.results_dir() / "results.csv"
    if not csv_path.is_file():
        from .aggregate import aggregate

        aggregate()
    df = pd.read_csv(csv_path)
    ok = _complete(df)
    written: list[Path] = []
    wanted = set(models or [])
    for model in sorted(ok["model"].unique()):
        key = model.split("/")[-1].lower()
        if wanted and key not in wanted and model not in wanted:
            continue
        for fn, suffix in ((plot_perplexity, "perplexity"), (plot_delta_heatmap, "delta_vs_fp16")):
            out = fn(ok, model, paths.plots_dir() / f"{key}_{suffix}.png")
            if out is not None:
                written.append(out)
    return written
