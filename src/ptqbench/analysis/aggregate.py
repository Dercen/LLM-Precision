"""results/raw/runs/*.json -> results/tables/{results.csv,summary.md} and one plain-language
page per model in results/by-model/. DESIGN.md 9.5.

Joins each row onto references/literature.yaml on (model, dataset_key, algo, bits,
group_size), and additionally on sym / act_order / calibration whenever the
literature row states them, so a row is never paired with a number produced under a
different calibration. Partial rows are excluded; failed and skipped rows are kept
in the CSV (they are information) but never joined or summarised.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .. import explain, paths

ROW_COLUMNS = [
    "run_id", "quant_key", "model", "loaded_from", "dataset", "algo", "bits", "group_size", "sym",
    "act_order", "true_sequential", "calib", "dtype", "eval_mode", "ppl", "nll_sum",
    "n_tokens", "n_windows", "partial", "paper_comparable", "quant_seconds",
    "eval_seconds", "peak_vram_gb", "peak_ram_gb", "status", "reason", "device_name",
    "hostname", "git_sha", "finished_at",
]


def load_rows(runs_dir: Path | None = None) -> list[dict[str, Any]]:
    runs_dir = runs_dir or paths.runs_dir()
    rows: list[dict[str, Any]] = []
    for path in sorted(runs_dir.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as fh:
                row = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue  # a half-written tmp never has the .json suffix; this is defensive
        row.setdefault("status", "ok")
        rows.append(row)
    return rows


def dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per run_id: the newest finished wins (server and laptop rows merge here)."""
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        rid = row.get("run_id")
        if not rid:
            continue
        prev = best.get(rid)
        if prev is None or str(row.get("finished_at", "")) >= str(prev.get("finished_at", "")):
            best[rid] = row
    return list(best.values())


def load_literature(path: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = path or paths.references_dir() / "literature.yaml"
    with open(path, encoding="utf-8") as fh:
        lit = yaml.safe_load(fh)
    return lit["rows"], lit.get("calibrations", {})


def _calib_matches(row_calib: Any, lit_calib_key: str | None, calibs: dict[str, Any]) -> bool:
    if lit_calib_key is None:
        return True
    want = calibs.get(lit_calib_key)
    if not want or not isinstance(row_calib, dict):
        return False
    return all(row_calib.get(k) == want[k] for k in ("dataset", "nsamples", "seqlen") if k in want)


def _norm_gs(value: Any) -> int:
    if value is None:
        return -1
    return int(value)


def match_literature(row: dict[str, Any], lit_rows: list[dict[str, Any]], calibs: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for lit in lit_rows:
        if lit["model"] != row.get("model") or lit["dataset_key"] != row.get("dataset"):
            continue
        if lit["algo"] != row.get("algo") or int(lit["bits"]) != int(row.get("bits", -1)):
            continue
        if _norm_gs(lit.get("group_size")) != _norm_gs(row.get("group_size")):
            continue
        if "sym" in lit and bool(lit["sym"]) != bool(row.get("sym")):
            continue
        if "act_order" in lit and bool(lit["act_order"]) != bool(row.get("act_order")):
            continue
        if not _calib_matches(row.get("calib"), lit.get("calib"), calibs):
            continue
        out.append(lit)
    return out


def build_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    lit_rows, calibs = load_literature()
    records = []
    fp16: dict[tuple[str, str], float] = {
        (r["model"], r["dataset"]): r["ppl"]
        for r in rows
        if r.get("algo") == "fp" and r.get("status") == "ok" and not r.get("partial")
    }
    for row in rows:
        rec = {col: row.get(col) for col in ROW_COLUMNS}
        rec["group_size"] = _norm_gs(row.get("group_size"))
        rec["calib"] = json.dumps(row["calib"], sort_keys=True) if isinstance(row.get("calib"), dict) else None
        rec["paper_ppl"] = None
        rec["paper_source"] = None
        rec["protocol_uncertain"] = None
        rec["delta_vs_paper"] = None
        rec["delta_vs_paper_pct"] = None
        rec["delta_vs_fp16"] = None
        if row.get("status") == "ok" and not row.get("partial"):
            matches = match_literature(row, lit_rows, calibs)
            # Prefer a row whose protocol is not uncertain; then the first listed.
            matches.sort(key=lambda m: bool(m.get("protocol_uncertain")))
            if matches:
                m = matches[0]
                rec["paper_ppl"] = float(m["ppl"])
                rec["paper_source"] = f"{m['source']}:table{m['table']}"
                rec["protocol_uncertain"] = bool(m.get("protocol_uncertain", False))
                rec["delta_vs_paper"] = row["ppl"] - float(m["ppl"])
                rec["delta_vs_paper_pct"] = (row["ppl"] - float(m["ppl"])) / float(m["ppl"]) * 100
                if len(matches) > 1:
                    rec["paper_alternates"] = ";".join(f"{x['source']}={x['ppl']}" for x in matches[1:])
            base = fp16.get((row["model"], row["dataset"]))
            if base is not None and row.get("algo") != "fp":
                rec["delta_vs_fp16"] = row["ppl"] - base
        records.append(rec)
    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df = df.sort_values(["model", "dataset", "algo", "bits", "group_size"], ascending=[True, True, True, False, True])
    return df


def write_csv(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or paths.tables_dir() / "results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _fmt(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    if isinstance(x, float):
        return f"{x:,.2f}" if x >= 100 else f"{x:.4f}"
    return str(x)


def write_summary(df: pd.DataFrame, path: Path | None = None) -> Path:
    """One markdown table per (model, dataset): algorithms down, bit widths across."""
    path = path or paths.tables_dir() / "summary.md"
    ok = df[(df["status"] == "ok") & (~df["partial"].astype(bool))] if not df.empty else df
    lines = ["# PTQ Bench summary", ""]
    if ok.empty:
        lines.append("_no complete rows yet_")
    else:
        n_paper = int(ok["paper_ppl"].notna().sum())
        lines.append(f"{len(ok)} complete rows, {n_paper} joined to a published number.")
        lines.append("")
        for (model, dataset), sub in ok.groupby(["model", "dataset"], sort=True):
            fp = sub[sub["algo"] == "fp"]["ppl"]
            head = f"## {model} · {dataset}"
            if not fp.empty:
                head += f" · fp16 {fp.iloc[0]:.4f}"
            lines.append(head)
            lines.append("")
            bits_cols = sorted({int(b) for b in sub["bits"] if int(b) != 16}, reverse=True)
            lines.append("| algo | gs | " + " | ".join(f"{b}-bit" for b in bits_cols) + " |")
            lines.append("|---|---|" + "---|" * len(bits_cols))
            for (algo, gs), cell in sub[sub["algo"] != "fp"].groupby(["algo", "group_size"], sort=True):
                vals = []
                for b in bits_cols:
                    hit = cell[cell["bits"] == b]
                    if hit.empty:
                        vals.append("—")
                    else:
                        r = hit.iloc[0]
                        s = _fmt(float(r["ppl"]))
                        if pd.notna(r.get("paper_ppl")):
                            s += f" (paper {_fmt(float(r['paper_ppl']))}, {r['delta_vs_paper_pct']:+.2f}%)"
                        vals.append(s)
                lines.append(f"| {algo} | {int(gs)} | " + " | ".join(vals) + " |")
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# Plain-language labels for the by-model pages live in ptqbench.explain, shared with the
# wizard so both surfaces say the same thing. summary.md keeps the raw keys.
ALGO_NAMES = explain.ALGO_NAMES
DATASET_NAMES = explain.DATASET_NAMES
DATASET_ORDER = ["wikitext2", "c4_new", "ptb_new", "c4", "ptb"]


def model_page_key(model: str) -> str:
    """File stem shared with the plots: 'facebook/opt-125m' -> 'opt-125m'."""
    return model.split("/")[-1].lower()


def _size_order(model: str) -> tuple[float, str]:
    """Sort key: parameter count parsed from the name ('opt-1.3b' -> 1.3e9), then name."""
    m = re.search(r"(\d+(?:\.\d+)?)([mb])(?![a-z])", model_page_key(model))
    if not m:
        return (float("inf"), model)
    return (float(m.group(1)) * (1e6 if m.group(2) == "m" else 1e9), model)


def _grouping(gs: int) -> str:
    return "per row" if gs == -1 else f"groups of {gs}"


def _worse(ppl: float, base: float | None) -> str:
    """How much worse than fp16, as a percentage below 2x and a multiple above."""
    if base is None:
        return ""
    ratio = ppl / base
    if ratio < 2:
        return f"{(ratio - 1) * 100:+.1f}%"
    return f"{ratio:,.0f}× worse" if ratio >= 10 else f"{ratio:.1f}× worse"


def _datasets_in(sub: pd.DataFrame) -> list[str]:
    present = set(sub["dataset"])
    return [d for d in DATASET_ORDER if d in present] + sorted(present - set(DATASET_ORDER))


def _model_page(model: str, rows: pd.DataFrame, failed: pd.DataFrame) -> str:
    key = model_page_key(model)
    datasets = _datasets_in(rows)
    fp16 = {d: float(rows[(rows["dataset"] == d) & (rows["algo"] == "fp")]["ppl"].iloc[0])
            for d in datasets if not rows[(rows["dataset"] == d) & (rows["algo"] == "fp")].empty}
    lines = [
        f"# {key}",
        "",
        f"Model: `{model}` · generated by `ptq aggregate` from `results/raw/runs/` — do not edit by hand.",
        "",
        "**How to read this page.** Perplexity measures how surprised the model is by real text:",
        "lower is better. The *full precision* number is the unmodified model (16-bit weights).",
        "Every other number is the same model after its weights were squeezed to fewer bits;",
        "the percentage says how much worse it got. Under ~5% is barely noticeable, over 2× is",
        "badly damaged. See `results/README.md` for the method and grouping terms.",
        "",
        "## Full precision (the baseline)",
        "",
        "| dataset | perplexity |",
        "|---|---|",
    ]
    for d in datasets:
        lines.append(f"| {DATASET_NAMES.get(d, d)} | {_fmt(fp16[d]) if d in fp16 else '—'} |")
    lines.append("")

    head = datasets[0]
    base = fp16.get(head)
    quant = rows[rows["algo"] != "fp"]
    best = quant[quant["dataset"] == head]
    if not best.empty:
        lines += [f"## Best result at each bit width, on {DATASET_NAMES.get(head, head)}", "",
                  "| bits | best method | grouping | perplexity | vs full precision |", "|---|---|---|---|---|"]
        for bits in sorted({int(b) for b in best["bits"]}, reverse=True):
            r = best[best["bits"] == bits].sort_values("ppl").iloc[0]
            lines.append(
                f"| {bits} | {ALGO_NAMES.get(r['algo'], r['algo'])} | {_grouping(int(r['group_size']))} "
                f"| {_fmt(float(r['ppl']))} | {_worse(float(r['ppl']), base)} |"
            )
        lines.append("")

    lines += ["## Charts", ""]
    for suffix, what in (("perplexity", "Perplexity by bit width"), ("delta_vs_fp16", "Increase over full precision")):
        if (paths.plots_dir() / f"{key}_{suffix}.png").is_file():
            lines += [f"![{what}](../plots/{key}_{suffix}.png)", ""]
    if lines[-2] == "## Charts":
        lines += ["_No charts yet: run `ptq plot`._", ""]

    for d in datasets:
        sub = quant[quant["dataset"] == d]
        if sub.empty:
            continue
        bits_cols = sorted({int(b) for b in sub["bits"]}, reverse=True)
        lines += [f"## Every result: {DATASET_NAMES.get(d, d)}", "",
                  f"Full precision: {_fmt(fp16[d]) if d in fp16 else '—'}", "",
                  "| method | grouping | " + " | ".join(f"{b}-bit" for b in bits_cols) + " |",
                  "|---|---|" + "---|" * len(bits_cols)]
        for (algo, gs), cell in sub.groupby(["algo", "group_size"], sort=True):
            vals = []
            for b in bits_cols:
                hit = cell[cell["bits"] == b]
                if hit.empty:
                    vals.append("—")
                    continue
                ppl = float(hit.iloc[0]["ppl"])
                vals.append(f"{_fmt(ppl)} ({_worse(ppl, fp16.get(d))})" if d in fp16 else _fmt(ppl))
            lines.append(f"| {ALGO_NAMES.get(algo, algo)} | {_grouping(int(gs))} | " + " | ".join(vals) + " |")
        lines.append("")

    paper = rows[rows["paper_ppl"].notna()]
    if not paper.empty:
        lines += ["## Checked against published papers", "",
                  "Our number next to the one a paper reports for the same setup. Within about 1% means",
                  "this pipeline reproduces the paper.", "",
                  "| dataset | method | bits | grouping | ours | published | difference | source |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, r in paper.iterrows():
            method = "full precision" if r["algo"] == "fp" else ALGO_NAMES.get(r["algo"], r["algo"])
            lines.append(
                f"| {r['dataset']} | {method} | {int(r['bits'])} | {_grouping(int(r['group_size']))} "
                f"| {_fmt(float(r['ppl']))} | {_fmt(float(r['paper_ppl']))} | {r['delta_vs_paper_pct']:+.2f}% "
                f"| {r['paper_source']} |"
            )
        lines.append("")

    if not failed.empty:
        lines += ["## Runs that did not produce a number", "",
                  f"{len(failed)} configuration(s) were skipped or failed. Reasons:", ""]
        for reason, n in failed["reason"].fillna("unknown").value_counts().items():
            lines.append(f"- `{reason}` ({n}): {explain.explain_reason(reason)}")
        lines.append("")
    return "\n".join(lines)


def write_by_model(df: pd.DataFrame, out_dir: Path | None = None) -> list[Path]:
    """One page per model plus an index: results/by-model/<model>.md and README.md."""
    out_dir = out_dir or paths.by_model_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return []
    ok = df[(df["status"] == "ok") & (~df["partial"].astype(bool))]
    written: list[Path] = []
    index = [
        "# Results by model",
        "",
        "Generated by `ptq aggregate` — do not edit by hand. One page per model, smallest first.",
        "Columns below are WikiText-2 perplexity (lower is better); the percentage is how much",
        "worse than the full-precision model.",
        "",
        "| model | full precision | best 4-bit | best 3-bit |",
        "|---|---|---|---|",
    ]
    for model in sorted(ok["model"].unique(), key=_size_order):
        sub = ok[ok["model"] == model]
        key = model_page_key(model)
        failed = df[(df["model"] == model) & (df["status"] != "ok")]
        path = out_dir / f"{key}.md"
        path.write_text(_model_page(model, sub, failed) + "\n", encoding="utf-8")
        written.append(path)
        wt = sub[sub["dataset"] == "wikitext2"]
        fp = wt[wt["algo"] == "fp"]["ppl"]
        base = float(fp.iloc[0]) if not fp.empty else None
        cells = []
        for bits in (4, 3):
            hit = wt[(wt["algo"] != "fp") & (wt["bits"] == bits)].sort_values("ppl")
            if hit.empty:
                cells.append("—")
            else:
                r = hit.iloc[0]
                cells.append(f"{_fmt(float(r['ppl']))} ({ALGO_NAMES.get(r['algo'], r['algo'])}, {_worse(float(r['ppl']), base)})")
        index.append(f"| [{key}]({key}.md) | {_fmt(base) if base is not None else '—'} | " + " | ".join(cells) + " |")
    (out_dir / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    return written


def aggregate(runs_dir: Path | None = None, *, refresh_timing: bool = True) -> tuple[Path, Path, pd.DataFrame]:
    rows = dedupe(load_rows(runs_dir))
    df = build_table(rows)
    if refresh_timing:
        from . import timing

        timing.refresh(rows)
    csv, md = write_csv(df), write_summary(df)
    write_by_model(df)
    return csv, md, df
