"""Per-layer sensitivity: which layers tolerate fewer bits.

`ptq export-mlir --score-windows N` leaves, in `layers.json`, every quantized layer's
relative output error on N windows of real text, ||Q(W) x - W x|| / ||W x||: the
per-layer objective GPTQ and AWQ minimise, measured after the fact for any method. This
module reads the `layers.json` of one or more exports of the same model (one export per
method, bit width and group size) and writes, per model:

* `results/sensitivity/<model>.csv`: one line per layer and configuration;
* `results/sensitivity/<model>.md`: the average over blocks per projection, then one
  table per configuration (blocks down, projections across) with its most sensitive
  layers named;
* `results/plots/<model>_sensitivity.png`: one heatmap per configuration on a shared
  log colour scale, which the by-model page and the website pick up like any chart.

A dark row says a projection needs more bits than its neighbours (mixed precision by
layer type); a dark column says a block does (mixed precision by block). Exports made
without `--score-windows` fall back to the weight-space error ||Q(W) - W|| / ||W||.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .. import paths

#: results/<this>/<model>.{md,csv}; the chart goes to the plots directory as <model>_<PLOT_SUFFIX>.png.
SENSITIVITY_DIRNAME = "sensitivity"
PLOT_SUFFIX = "sensitivity"
#: The two sensitivity measures layers.json can carry, preferred first.
METRIC_OUTPUT = "output_rel_error"
METRIC_WEIGHT = "rel_error"
METRIC_TEXT = {
    METRIC_OUTPUT: (
        "Sensitivity is each quantized layer's relative output error on real text, "
        "‖Q(W)x − Wx‖ / ‖Wx‖ over the scoring windows (`--score-windows`): 0.05 means the "
        "layer's output moved by 5%. Lower is better. A layer well above its neighbours "
        "needs more bits than they do."
    ),
    METRIC_WEIGHT: (
        "Sensitivity is each quantized layer's relative weight error ‖Q(W) − W‖ / ‖W‖. "
        "These exports were made without `--score-windows`, so the output error on real "
        "text is not available; the weight error ignores what the layer's inputs look like."
    ),
}
#: Layers named per configuration in the markdown.
TOP_N = 10
#: Cell values are printed on the heatmap only while the grid stays readable.
MAX_BLOCKS_WITH_VALUES = 16
#: Panels per row of the heatmap figure.
PANEL_COLUMNS = 3
#: Methods in the order the tables and panels list them.
ALGO_ORDER = ("rtn", "gptq", "awq_lite", "hqq")
#: "<prefix>.layers.<block>.<path inside the block>"; the projection is the path's last part.
_MODULE_RE = re.compile(r"\.(\d+)\.(.+)$")
CAPTION = ("Each cell is one quantized layer: darker means its output moved more under quantization, on a log "
           "colour scale shared by every panel so bit widths can be compared. Blocks run left to right through "
           "the model; rows are the projections inside a block. A dark row says that projection needs more "
           "bits; a dark column says that block does.")


@dataclass(frozen=True)
class ExportMeta:
    """What one export quantized, from its export.json."""

    model: str
    model_key: str
    algo: str
    bits: int
    group_size: int
    act_order: bool
    path: Path

    @property
    def label(self) -> str:
        algo = self.algo + ("-ao" if self.algo == "gptq" and self.act_order else "")
        gs = "per-row" if self.group_size == -1 else f"g{self.group_size}"
        return f"{algo} {self.bits}-bit {gs}"

    @property
    def order(self) -> tuple[int, int, int]:
        return (ALGO_ORDER.index(self.algo) if self.algo in ALGO_ORDER else len(ALGO_ORDER), -self.bits, self.group_size)


def read_export(layers_json: Path) -> tuple[ExportMeta, pd.DataFrame]:
    """One export's layers as a table: module, block, projection, the error measures."""
    layers_json = Path(layers_json)
    manifest = json.loads((layers_json.parent / "export.json").read_text(encoding="utf-8"))
    layers = json.loads(layers_json.read_text(encoding="utf-8"))
    quant = manifest.get("quant") or {}
    meta = ExportMeta(
        model=str(manifest["model"]),
        model_key=str(manifest.get("model_key") or manifest["model"].split("/")[-1].lower()),
        algo=str(quant.get("algo", manifest.get("algo"))),
        bits=int(quant.get("bits", manifest.get("bits"))),
        group_size=int(quant.get("group_size", manifest.get("group_size")) or -1),
        act_order=bool(quant.get("act_order", False)),
        path=layers_json,
    )
    rows = []
    for name, info in layers.items():
        m = _MODULE_RE.search(name)
        if m is None:
            continue  # not inside the block list
        shape = info.get("shape") or [None, None]
        rows.append({
            "position": len(rows),  # file order: the order the model lists its projections
            "module": name, "block": int(m.group(1)), "projection": m.group(2).split(".")[-1],
            METRIC_OUTPUT: info.get(METRIC_OUTPUT), METRIC_WEIGHT: info.get(METRIC_WEIGHT), "snr_db": info.get("snr_db"),
            "out_features": shape[0], "in_features": shape[1],
        })
    if not rows:
        raise ValueError(f"{layers_json}: no block-level layers")
    df = pd.DataFrame(rows)
    for field in ("model", "algo", "bits", "group_size", "act_order"):
        df[field] = getattr(meta, field)
    df["config"] = meta.label
    return meta, df


def profile(layers_jsons: Iterable[Path]) -> tuple[str, pd.DataFrame]:
    """All exports of one model as one table, configurations in method/bits/group order."""
    metas: list[ExportMeta] = []
    frames: list[pd.DataFrame] = []
    for path in layers_jsons:
        meta, df = read_export(Path(path))
        metas.append(meta)
        frames.append(df)
    if not metas:
        raise ValueError("no layers.json given")
    keys = sorted({m.model_key for m in metas})
    if len(keys) > 1:
        raise ValueError(f"one model at a time; these exports cover {keys}")
    order = {m.label: m.order for m in metas}
    df = pd.concat(frames, ignore_index=True)
    df["_order"] = df["config"].map(order)
    df = df.sort_values(["_order", "block", "position"]).drop(columns="_order").reset_index(drop=True)
    return keys[0], df


def metric_for(df: pd.DataFrame) -> str:
    return METRIC_OUTPUT if df[METRIC_OUTPUT].notna().any() else METRIC_WEIGHT


def projection_order(df: pd.DataFrame) -> list[str]:
    """Projections as the first block lists them (q, k, v, o, gate, up, down), then any others."""
    first = df[df["block"] == df["block"].min()]
    seen = list(dict.fromkeys(first["projection"]))
    return seen + sorted(set(df["projection"]) - set(seen))


def configs_in(df: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(df["config"]))


def block_table(df: pd.DataFrame, metric: str, projections: list[str]) -> pd.DataFrame:
    """Blocks down, projections across, for the rows of one configuration."""
    return df.pivot_table(index="block", columns="projection", values=metric, aggfunc="first").reindex(columns=projections)


def projection_means(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Configurations down, projections across: the mean over blocks."""
    table = df.pivot_table(index="config", columns="projection", values=metric, aggfunc="mean")
    return table.reindex(index=configs_in(df), columns=projection_order(df))


def most_sensitive(df: pd.DataFrame, metric: str, n: int = TOP_N) -> pd.DataFrame:
    return df.dropna(subset=[metric]).nlargest(n, metric)[["block", "projection", metric]]


def _cell(v: Any) -> str:
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{float(v):.4f}"


def _markdown(table: pd.DataFrame, index_name: str) -> list[str]:
    cols = [str(c) for c in table.columns]
    lines = ["| " + index_name + " | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for idx, row in table.iterrows():
        lines.append(f"| {idx} | " + " | ".join(_cell(v) for v in row.tolist()) + " |")
    return lines


def write_tables(model_key: str, df: pd.DataFrame, out_dir: Path | None = None) -> tuple[Path, Path]:
    """`<model>.csv` (every layer row) and `<model>.md` (the tables) under results/sensitivity/."""
    out_dir = out_dir or paths.results_dir() / SENSITIVITY_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"{model_key}.csv"
    df.to_csv(csv, index=False)
    metric = metric_for(df)
    projections = projection_order(df)
    n_cfg = df["config"].nunique()
    lines = [
        f"# {model_key}: which layers tolerate fewer bits",
        "",
        f"Generated by `ptq sensitivity` from the `layers.json` of {n_cfg} export(s) of `{df['model'].iloc[0]}` — do not edit by hand.",
        "",
        METRIC_TEXT[metric],
        "",
        "## Average over blocks, per projection",
        "",
        *_markdown(projection_means(df, metric), "configuration"),
        "",
    ]
    for cfg in configs_in(df):
        d = df[df["config"] == cfg]
        lines += [f"## {cfg}", "", *_markdown(block_table(d, metric, projections), "block"), ""]
        top = most_sensitive(d, metric)
        if not top.empty:
            named = ", ".join(f"block {int(r['block'])} `{r['projection']}` {_cell(r[metric])}" for _, r in top.iterrows())
            lines += [f"Most sensitive layers: {named}.", ""]
    md = out_dir / f"{model_key}.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv, md


def plot_sensitivity(model_key: str, df: pd.DataFrame, out: Path) -> Path:
    """One heatmap per configuration: blocks across, projections down, log colour scale
    shared by every panel."""
    import matplotlib.pyplot as plt

    from . import plots as P

    P._style()
    metric = metric_for(df)
    projections = projection_order(df)
    configs = configs_in(df)
    blocks = sorted(int(b) for b in df["block"].unique())
    values = df[metric].dropna().astype(float)
    positive = values[values > 0]
    vmin = float(positive.min()) if not positive.empty else 1e-6
    vmax = float(values.max()) if not values.empty else 1.0
    span = math.log10(vmax) - math.log10(vmin) if vmax > vmin else 1.0
    cols = min(len(configs), PANEL_COLUMNS)
    rows = math.ceil(len(configs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=((0.27 * len(blocks) + 2.4) * cols, (0.4 * len(projections) + 1.5) * rows + 0.9),
                             squeeze=False)
    for ax, cfg in zip(axes.flat, configs, strict=False):
        d = df[df["config"] == cfg]
        table = block_table(d, metric, projections)
        ax.set_xlim(0, len(blocks))
        ax.set_ylim(0, len(projections))
        ax.invert_yaxis()
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for bi, block in enumerate(blocks):
            for pi, proj in enumerate(projections):
                if block not in table.index or proj not in table.columns:
                    continue
                v = table.at[block, proj]
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    continue
                t = (math.log10(max(float(v), vmin)) - math.log10(vmin)) / span
                colour = P.SEQ[min(len(P.SEQ) - 1, max(0, round(t * (len(P.SEQ) - 1))))]
                ax.add_patch(plt.Rectangle((bi + 0.04, pi + 0.04), 0.92, 0.92, facecolor=colour, edgecolor="none"))
                if len(blocks) <= MAX_BLOCKS_WITH_VALUES:
                    ink = "#ffffff" if P._luminance(colour) < 0.45 else P.INK["primary"]
                    ax.text(bi + 0.5, pi + 0.5, f"{float(v):.3f}", ha="center", va="center", fontsize=7, color=ink)
        step = max(1, len(blocks) // 15)
        ax.set_xticks([i + 0.5 for i in range(0, len(blocks), step)])
        ax.set_xticklabels([str(b) for b in blocks[::step]], fontsize=8)
        ax.set_yticks([i + 0.5 for i in range(len(projections))])
        ax.set_yticklabels(projections, fontsize=8)
        ax.set_xlabel("block")
        ax.set_title(cfg, loc="left")
    for ax in list(axes.flat)[len(configs):]:
        ax.set_visible(False)
    kind = "output" if metric == METRIC_OUTPUT else "weight"
    fig.suptitle(f"{model_key} — per-layer sensitivity ({kind} error, log colour scale)", x=0.01, ha="left", fontsize=12)
    fig.text(0.01, 0.006, CAPTION, ha="left", va="bottom", fontsize=8.2, color=P.INK["secondary"], wrap=True)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def run(layers_jsons: Iterable[Path], out_dir: Path | None = None, plots_dir: Path | None = None) -> dict[str, Any]:
    """Tables and heatmap for the exports given; returns what was written and the summary."""
    model_key, df = profile(layers_jsons)
    csv, md = write_tables(model_key, df, out_dir)
    png = plot_sensitivity(model_key, df, (plots_dir or paths.plots_dir()) / f"{model_key}_{PLOT_SUFFIX}.png")
    metric = metric_for(df)
    return {
        "model_key": model_key, "metric": metric, "csv": csv, "md": md, "png": png,
        "means": projection_means(df, metric), "n_configs": df["config"].nunique(), "n_layers": len(df),
    }
