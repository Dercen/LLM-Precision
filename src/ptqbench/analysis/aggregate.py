"""results/runs/*.json -> results/results.csv and results/summary.md. PLAN.md 9.5.

Joins each row onto references/literature.yaml on (model, dataset_key, algo, bits,
group_size), and additionally on sym / act_order / calibration whenever the
literature row states them, so a row is never paired with a number produced under a
different calibration. Partial rows are excluded; failed and skipped rows are kept
in the CSV (they are information) but never joined or summarised.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .. import paths

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
    path = path or paths.results_dir() / "results.csv"
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
    path = path or paths.results_dir() / "summary.md"
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def aggregate(runs_dir: Path | None = None, *, refresh_timing: bool = True) -> tuple[Path, Path, pd.DataFrame]:
    rows = dedupe(load_rows(runs_dir))
    df = build_table(rows)
    if refresh_timing:
        from . import timing

        timing.refresh(rows)
    return write_csv(df), write_summary(df), df
