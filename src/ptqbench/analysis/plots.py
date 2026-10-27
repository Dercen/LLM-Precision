"""`ptq plot`: static figures from results.csv. DESIGN.md 9.5.

Four figures per model:

* perplexity: three panels (wikitext2, c4_new, ptb_new), bits categorical on x, log-y
  perplexity, one line per algorithm in a FIXED colour (slot order rtn, gptq, awq_lite,
  hqq -- colour follows the entity, never the row count), solid for g128, dashed for
  per-row, dotted for g64; the fp16 baseline as a muted horizontal rule; published
  numbers as hollow markers in the series colour.
* delta heatmap: (algorithm, group size) rows by bits, coloured by log10(1 + ppl - fp16)
  on a single-hue sequential ramp, the value printed in each cell in ink chosen by the
  fill's luminance.
* frontier: perplexity against weight bytes (`analysis.footprint`) on WikiText-2, every
  setting a point, the Pareto frontier a grey step, fp16 a square.
* group size: one panel per bit width, perplexity against grouping from per-row down to
  the smallest group, the bits per weight of each grouping under its label. Drawn only
  for models with three or more groupings at one bit width (the group-size sweeps).

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
DATASET_TITLE = {"wikitext2": "WikiText-2 (Wikipedia)", "c4_new": "C4 (web pages)", "ptb_new": "PTB (news)"}
BAND = "#e8efe6"  # within 5% of fp16: the loss that is hard to notice in use
CAPTION_PPL = ("Lower is better. The grey line is the unmodified 16-bit model (fp16) and the shaded band is "
               "within 5% of it, a loss that is hard to notice in use. Hollow circles are numbers published in papers. "
               "Rows: how many weights share one scale factor (g128 = groups of 128; per-row = a whole row).")
CAPTION_DELTA = ("Each cell is how much perplexity rose above the unmodified 16-bit model: 0 means no loss, darker is "
                 "worse. Rows are method and grouping (g128 = groups of 128).")


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
    """Small multiples: one row per group size, one column per dataset, <= 4 lines each.

    Four methods x three group sizes on one axis is eleven series and unreadable; a
    group size per row keeps every panel at one line per method. Group sizes with a
    single bit width (g64, 2-bit only) have no line to draw and live in the heatmap
    and the table instead.
    """
    sub = ok[ok["model"] == model]
    if sub.empty:
        return None
    datasets = [d for d in DATASETS if d in set(sub["dataset"])]
    gs_rows = [gs for gs in (-1, 128, 64)
               if sub[(sub["group_size"] == gs) & (sub["algo"] != "fp")]["bits"].nunique() >= 2]
    if not gs_rows:
        return None
    fig, axes = plt.subplots(len(gs_rows), len(datasets), figsize=(4.4 * len(datasets), 3.6 * len(gs_rows)),
                             squeeze=False, sharey="row")
    handles: dict[str, Line2D] = {}
    for ri, gs in enumerate(gs_rows):
        for ci, dataset in enumerate(datasets):
            ax = axes[ri][ci]
            d = sub[sub["dataset"] == dataset]
            fp = d[d["algo"] == "fp"]["ppl"]
            ax.set_xticks(range(len(BITS)))
            ax.set_xticklabels([f"{b}-bit" for b in BITS])
            ax.set_yscale("log")
            ax.grid(axis="x", visible=False)
            if not fp.empty:
                ax.axhspan(fp.iloc[0], fp.iloc[0] * 1.05, color=BAND, zorder=0, linewidth=0)
                ax.axhline(fp.iloc[0], color=CHROME["axis"], linewidth=1.2, zorder=1)
                ax.annotate(f"fp16 {fp.iloc[0]:.2f}", xy=(-0.35, fp.iloc[0]), xytext=(0, 3),
                            textcoords="offset points", ha="left", va="bottom", fontsize=8, color=INK["secondary"])
            end_labels: list[tuple[int, float, str]] = []
            for algo, colour in SERIES.items():
                q = d[(d["algo"] == algo) & (d["group_size"] == gs)].sort_values("bits", ascending=False)
                if q.empty:
                    continue
                xs = [BITS.index(b) for b in q["bits"] if b in BITS]
                ys = [float(pv) for b, pv in zip(q["bits"], q["ppl"], strict=True) if b in BITS]
                line, = ax.plot(xs, ys, "-", color=colour, linewidth=2, marker="o", markersize=8,
                                markerfacecolor=colour, markeredgecolor=CHROME["surface"], markeredgewidth=2,
                                solid_capstyle="round", zorder=3)
                handles.setdefault(algo, line)
                paper = q[q["paper_ppl"].notna()]
                if not paper.empty:
                    ax.plot([BITS.index(b) for b in paper["bits"]], paper["paper_ppl"].astype(float), "o",
                            markersize=8, markerfacecolor=CHROME["surface"], markeredgecolor=colour,
                            markeredgewidth=1.6, linestyle="none", zorder=4)
                if xs:
                    end_labels.append((xs[-1], ys[-1], ALGO_LABEL[algo]))
            placed: list[tuple[int, float]] = []
            for x, y, text in sorted(end_labels, key=lambda t: t[1]):
                if any(px == x and abs(math.log10(y) - math.log10(py)) < 0.08 for px, py in placed):
                    continue
                placed.append((x, y))
                ax.annotate(text, xy=(x, y), xytext=(8, 0), textcoords="offset points",
                            va="center", fontsize=8.5, color=INK["secondary"])
            if ri == 0:
                ax.set_title(DATASET_TITLE.get(dataset, dataset), loc="left")
            ax.set_xlim(-0.4, len(BITS) - 0.4)
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _fmt_ppl(v)))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            if ci == 0:
                ax.set_ylabel(f"{GS_LABEL[gs]}\nperplexity (log), lower is better")
    legend_items = [handles[a] for a in SERIES if a in handles]
    legend_labels = [ALGO_LABEL[a] for a in SERIES if a in handles]
    legend_items.append(Line2D([], [], marker="o", linestyle="none", markersize=8, markerfacecolor=CHROME["surface"],
                               markeredgecolor=INK["muted"], markeredgewidth=1.6))
    legend_labels.append("published")
    legend_items.append(plt.Rectangle((0, 0), 1, 1, facecolor=BAND, edgecolor="none"))
    legend_labels.append("within 5% of fp16")
    fig.legend(legend_items, legend_labels, loc="lower center", ncol=len(legend_items), bbox_to_anchor=(0.5, 0.035))
    fig.suptitle(f"{model} — perplexity by weight precision", x=0.01, ha="left", fontsize=12, color=INK["primary"])
    fig.text(0.01, 0.006, CAPTION_PPL, ha="left", va="bottom", fontsize=8.2, color=INK["secondary"],
             wrap=True, transform=fig.transFigure)
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
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
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets) + 0.8, 0.42 * len(rows) + 2.1), squeeze=False)
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
        ax.set_title(DATASET_TITLE.get(dataset, dataset), loc="left")
    fig.suptitle(f"{model} — how much worse than the unmodified model (log colour scale)", x=0.01, ha="left", fontsize=12)
    fig.text(0.01, 0.01, CAPTION_DELTA, ha="left", va="bottom", fontsize=8.2, color=INK["secondary"], wrap=True)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


CAPTION_FRONTIER = ("Lower-left is better: less memory and less loss. Bytes are the packed weights (codes plus a 16-bit "
                    "scale and a zero-point per group) with embeddings, norms and biases left at 16 bits. The grey step is "
                    "the frontier, and only the settings on it are labelled; the square is the unmodified fp16 model. "
                    "Settings more than 20x worse than fp16 are left off.")
CAPTION_GROUPSIZE = ("Lower is better. Smaller groups give every weight a closer scale factor but cost memory: the number "
                     "under each grouping is its bits per weight for plain rounding, scales and zero-points included "
                     "(HQQ's float zero-points and act-order GPTQ's group index add a little). The grey line is the "
                     "unmodified fp16 model and the band is within 5% of it.")
#: Settings this many times worse than fp16 are off the useful scale of the frontier chart.
FRONTIER_CUTOFF = 20.0
#: A frontier needs at least this many settings to be worth a chart.
MIN_POINTS = 3
#: Frontier labels: the next label waits for the loss to fall by this much (log10; 0.08 is 20%).
LABEL_GAP = 0.08
#: Inches: narrower figures wrap the title and caption into each other.
MIN_FIGURE_WIDTH = 7.6
#: Inches per grouping on a group-size panel: eight groupings need room for two-line labels.
PANEL_WIDTH_PER_GROUPING = 0.62
#: A group-size chart needs at least this many groupings at one bit width: the sweeps
#: have six to eight, the plain matrix at most three (per-row, g128 and 2-bit g64).
MIN_GROUP_SIZES = 4


def _gs_label(gs: int) -> str:
    return GS_LABEL.get(gs, "per-row" if gs == -1 else f"g{gs}")


def _fp16_rule(ax, base: float) -> None:
    """The unmodified model as a rule, with the within-5% band above it."""
    ax.axhspan(base, base * 1.05, color=BAND, zorder=0, linewidth=0)
    ax.axhline(base, color=CHROME["axis"], linewidth=1.2, zorder=1)


def _ppl_axis(ax) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _fmt_ppl(v)))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def _label_narrow_axis(ax) -> None:
    """A log axis spanning less than a decade may show no major tick at all (15 to 27
    sits between 10 and 100): label its minor ticks instead."""
    lo, hi = ax.get_ylim()
    if hi / lo < 10:
        ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _fmt_ppl(v)))


def plot_frontier(ok: pd.DataFrame, model: str, out: Path, dataset: str = "wikitext2") -> Path | None:
    """Perplexity against weight bytes on one dataset: every setting as a point, the
    settings nothing beats on both axes joined into a step (the Pareto frontier)."""
    if "weight_bytes" not in ok.columns:
        return None
    sub = ok[(ok["model"] == model) & (ok["dataset"] == dataset) & ok["weight_bytes"].notna()]
    fp = sub[sub["algo"] == "fp"]
    if fp.empty or len(sub) < MIN_POINTS:
        return None
    base = float(fp["ppl"].iloc[0])
    sub = sub[sub["ppl"] <= base * FRONTIER_CUTOFF]
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    _ppl_axis(ax)
    _fp16_rule(ax, base)
    fp_mb = float(fp["weight_bytes"].iloc[0]) / 1e6
    ax.plot([fp_mb], [base], "s", markersize=9, color=INK["secondary"], linestyle="none", zorder=4)
    ax.annotate("fp16", xy=(fp_mb, base), xytext=(6, 5), textcoords="offset points", fontsize=8.5, color=INK["secondary"])
    handles: dict[str, Line2D] = {}
    for algo, colour in SERIES.items():
        q = sub[sub["algo"] == algo]
        if q.empty:
            continue
        line, = ax.plot(q["weight_bytes"].astype(float) / 1e6, q["ppl"].astype(float), "o", color=colour, markersize=7,
                        markeredgecolor=CHROME["surface"], markeredgewidth=1.2, linestyle="none", zorder=3)
        handles[algo] = line
    quant = sub[sub["algo"] != "fp"]
    points = sorted(
        (float(r["weight_bytes"]) / 1e6, float(r["ppl"]), f"{int(r['bits'])}b {_gs_label(int(r['group_size']))}")
        for _, r in quant.iterrows()
    )
    front: list[tuple[float, float, str]] = []
    best = math.inf
    for x, y, label in points:  # left to right, keep each new low: the staircase nothing improves on
        if y < best:
            best = y
            front.append((x, y, label))
    last_labelled = math.inf
    for x, y, label in front:
        # The frontier descends in clusters, one point per grouping. Label a point only
        # once the loss has dropped LABEL_GAP below the last labelled one.
        if math.log10(last_labelled) - math.log10(y) < LABEL_GAP:
            continue
        last_labelled = y
        ax.annotate(label, xy=(x, y), xytext=(6, 3), textcoords="offset points", fontsize=7.5, color=INK["secondary"])
    if base < best:  # the unmodified model ends the staircase (it is labelled above)
        front.append((fp_mb, base, "fp16"))
    if len(front) > 1:
        ax.step([x for x, _, _ in front], [y for _, y, _ in front], where="post", color=INK["muted"], linewidth=1.2, zorder=2)
    ax.set_xlabel("weights (MB)")
    ax.set_ylabel("perplexity (log), lower is better")
    ax.set_xlim(left=0.9 * float(sub["weight_bytes"].min()) / 1e6)
    _label_narrow_axis(ax)
    items = [handles[a] for a in SERIES if a in handles]
    labels = [ALGO_LABEL[a] for a in SERIES if a in handles]
    items.append(Line2D([], [], marker="s", linestyle="none", markersize=8, color=INK["secondary"]))
    labels.append("fp16")
    items.append(plt.Rectangle((0, 0), 1, 1, facecolor=BAND, edgecolor="none"))
    labels.append("within 5% of fp16")
    fig.legend(items, labels, loc="lower center", ncol=len(items), bbox_to_anchor=(0.5, 0.1))
    fig.suptitle(f"{model} — perplexity against model size, {DATASET_TITLE.get(dataset, dataset)}",
                 x=0.01, ha="left", fontsize=12, color=INK["primary"])
    fig.text(0.01, 0.006, CAPTION_FRONTIER, ha="left", va="bottom", fontsize=8.2, color=INK["secondary"], wrap=True)
    fig.tight_layout(rect=(0, 0.17, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_groupsize(ok: pd.DataFrame, model: str, out: Path, dataset: str = "wikitext2") -> Path | None:
    """One panel per bit width: perplexity against grouping, per-row then large groups to
    small, one line per method; the bits per weight each grouping costs under its label."""
    sub = ok[(ok["model"] == model) & (ok["dataset"] == dataset)]
    quant = sub[sub["algo"] != "fp"]
    panels = [b for b in (4, 3, 2) if quant[quant["bits"] == b]["group_size"].nunique() >= MIN_GROUP_SIZES]
    if not panels:
        return None
    gs_values = sorted({int(g) for g in quant[quant["bits"].isin(panels)]["group_size"]}, key=lambda g: (g != -1, -g))
    fp = sub[sub["algo"] == "fp"]["ppl"]
    # Each bit width gets its own y range: 2-bit collapses to 10^6 and more, which on a
    # shared axis would flatten the 4-bit panel, the one that carries the decision.
    panel_width = max(4.6, PANEL_WIDTH_PER_GROUPING * len(gs_values) + 1.6)
    fig, axes = plt.subplots(1, len(panels), figsize=(max(MIN_FIGURE_WIDTH, panel_width * len(panels)), 4.8), squeeze=False)
    handles: dict[str, Line2D] = {}
    for ax, bits in zip(axes[0], panels, strict=True):
        d = quant[quant["bits"] == bits]
        labels = []
        for g in gs_values:
            cell = d[d["group_size"] == g]
            label = _gs_label(g)
            if "bits_per_weight" in cell.columns and cell["bits_per_weight"].notna().any():
                plain = cell[cell["algo"] == "rtn"]["bits_per_weight"].dropna()
                bpw = float(plain.iloc[0]) if not plain.empty else float(cell["bits_per_weight"].dropna().min())
                label += f"\n{bpw:.2f}"
            labels.append(label)
        ax.set_xticks(range(len(gs_values)))
        ax.set_xticklabels(labels, fontsize=8)
        _ppl_axis(ax)
        ax.grid(axis="x", visible=False)
        if not fp.empty:
            _fp16_rule(ax, float(fp.iloc[0]))
            ax.annotate(f"fp16 {float(fp.iloc[0]):.2f}", xy=(-0.35, float(fp.iloc[0])), xytext=(0, 3),
                        textcoords="offset points", ha="left", va="bottom", fontsize=8, color=INK["secondary"])
        for algo, colour in SERIES.items():
            q = d[d["algo"] == algo]
            if q.empty:
                continue
            pts = sorted((gs_values.index(int(g)), float(p)) for g, p in zip(q["group_size"], q["ppl"], strict=True))
            line, = ax.plot([x for x, _ in pts], [y for _, y in pts], "-", color=colour, linewidth=2, marker="o",
                            markersize=7, markerfacecolor=colour, markeredgecolor=CHROME["surface"], markeredgewidth=2,
                            solid_capstyle="round", zorder=3)
            handles.setdefault(algo, line)
        ax.set_title(f"{bits}-bit", loc="left")
        ax.set_xlim(-0.4, len(gs_values) - 0.6)
        _label_narrow_axis(ax)
    for ax in axes[0]:
        ax.set_ylabel("perplexity (log), lower is better")
    items = [handles[a] for a in SERIES if a in handles]
    labels = [ALGO_LABEL[a] for a in SERIES if a in handles]
    items.append(plt.Rectangle((0, 0), 1, 1, facecolor=BAND, edgecolor="none"))
    labels.append("within 5% of fp16")
    fig.legend(items, labels, loc="lower center", ncol=len(items), bbox_to_anchor=(0.5, 0.1))
    fig.suptitle(f"{model} — perplexity by group size, {DATASET_TITLE.get(dataset, dataset)}",
                 x=0.01, ha="left", fontsize=12, color=INK["primary"])
    fig.text(0.01, 0.006, CAPTION_GROUPSIZE, ha="left", va="bottom", fontsize=8.2, color=INK["secondary"], wrap=True)
    fig.tight_layout(rect=(0, 0.18, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_all(models: list[str] | None = None, csv_path: Path | None = None) -> list[Path]:
    _style()
    csv_path = csv_path or paths.tables_dir() / "results.csv"
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
        for fn, suffix in ((plot_perplexity, "perplexity"), (plot_delta_heatmap, "delta_vs_fp16"),
                           (plot_frontier, "frontier"), (plot_groupsize, "groupsize")):
            target = paths.plots_dir() / f"{key}_{suffix}.png"
            out = fn(ok, model, target)
            if out is not None:
                written.append(out)
            else:  # nothing to draw any more: a stale chart must not stay on the model's page
                target.unlink(missing_ok=True)
    return written
