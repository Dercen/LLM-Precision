"""`ptq wizard` (also plain `ptq`): pick a model, datasets, a method and a precision from
menus, confirm, run, and read the result next to the published number.

The prompts only fill in `Answers`; `build_runs()` turns answers into the same RunSpecs
`ptq run` executes, so a wizard row is indistinguishable from a matrix row -- same ids,
same schema, same provenance. Everything below the prompt layer is unit-testable.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config as C
from . import paths

ALGOS: list[tuple[str, str]] = [
    ("fp", "fp16 baseline — no quantization, the reference number"),
    ("rtn", "RTN — round-to-nearest, data-free, seconds"),
    ("gptq", "GPTQ — error-compensating, needs calibration data, minutes"),
    ("awq_lite", "AWQ-lite — activation-aware scaling + clipping, needs calibration data"),
    ("hqq", "HQQ — data-free with optimised zero-points"),
]
DATASETS: list[tuple[str, str]] = [
    ("wikitext2", "WikiText-2 test — the primary column in every paper"),
    ("c4_new", "C4 validation — paper column (GPTQ --new-eval variant)"),
    ("ptb_new", "Penn Treebank test — paper column (GPTQ --new-eval variant)"),
    ("c4", "C4, random-window variant (opt-in, not the paper column)"),
    ("ptb", "PTB validation variant (opt-in, not the paper column)"),
]
BITS = [8, 4, 3, 2]
GROUP_SIZES: list[tuple[int, str]] = [(-1, "per-row (the GPTQ paper's OPT setting)"), (128, "g128 (the Llama papers' setting)"), (64, "g64")]


@dataclass
class Answers:
    model_key: str
    datasets: list[str] = field(default_factory=lambda: ["wikitext2"])
    algo: str = "fp"
    bits: int = 16
    group_size: int = -1
    act_order: bool | None = None  # None -> the family default (Llama: on)
    calib_dataset: str = "c4"
    seed: int = 0
    quick: bool = False  # 20 windows, marked partial
    eval_mode: str = "auto"


def model_choices() -> list[dict[str, Any]]:
    """Every configs/models entry with whether its weights are already on disk."""
    out = []
    for path in sorted((paths.configs_dir() / "models").glob("*.yaml")):
        spec = C.load_model(path.stem)
        cached = any((paths.hf_home()).glob(f"models--{spec.repo.replace('/', '--')}/snapshots/*/config.json"))
        if not cached and spec.mirror:
            cached = any((paths.hf_home()).glob(f"models--{spec.mirror.replace('/', '--')}/snapshots/*/config.json"))
        out.append({"key": spec.key, "repo": spec.repo, "gated": spec.gated, "cached": cached, "notes": spec.notes or ""})
    return out


def build_runs(a: Answers) -> list[C.RunSpec]:
    model = C.load_model(a.model_key)
    family = C._family_of(model)
    act_order = a.act_order if a.act_order is not None else (family == "llama")
    quant = C.QuantSpec(algo=a.algo, bits=16 if a.algo == "fp" else a.bits,
                        group_size=-1 if a.algo == "fp" else a.group_size, act_order=act_order)
    calib = C.CalibSpec(dataset=a.calib_dataset, seed=a.seed) if quant.is_data_dependent() else None
    ev = C.EvalSpec(max_windows=20 if a.quick else None, eval_mode=a.eval_mode)  # type: ignore[arg-type]
    return [C.RunSpec(model=model, quant=quant, calib=calib, dataset=ds, eval=ev, dtype=model.dtype) for ds in a.datasets]


def paper_number(row: dict[str, Any]) -> tuple[float | None, str | None]:
    from .analysis import aggregate as A

    lit_rows, calibs = A.load_literature()
    matches = A.match_literature(row, lit_rows, calibs)
    matches.sort(key=lambda m: bool(m.get("protocol_uncertain")))
    if not matches:
        return None, None
    m = matches[0]
    return float(m["ppl"]), f"{m['source']} table {m['table']}"


def execute_answers(a: Answers, *, device_spec: str = "auto", write: bool = True, log=print) -> list[dict[str, Any]]:
    """Run every dataset for the chosen configuration through the shared executor."""
    from . import device as D
    from .runner import execute as X

    device = D.resolve(device_spec)
    runs = build_runs(a)
    prep = X.prepare(runs[0], device)
    rows = []
    for run in runs:
        row = X.evaluate(prep, run, progress=sys.stderr.isatty())
        if write:
            row["_written_to"] = X.write_row(row)
        rows.append(row)
    return rows


# ---- the prompt layer ---------------------------------------------------------------


def _ask() -> Answers | None:
    import questionary as q

    models = model_choices()
    m_choices = [q.Choice(f"{m['key']:14s} {m['repo']:34s} {'cached' if m['cached'] else 'download'}{'  gated→mirror' if m['gated'] else ''}", value=m["key"]) for m in models]
    model_key = q.select("Model", choices=m_choices, use_shortcuts=False).ask()
    if model_key is None:
        return None
    datasets = q.checkbox("Datasets (space to toggle, enter to confirm)",
                          choices=[q.Choice(f"{k:10s} {d}", value=k, checked=(k == "wikitext2")) for k, d in DATASETS]).ask()
    if not datasets:
        return None
    algo = q.select("Method", choices=[q.Choice(f"{k:9s} {d}", value=k) for k, d in ALGOS]).ask()
    if algo is None:
        return None
    a = Answers(model_key=model_key, datasets=datasets, algo=algo)
    if algo != "fp":
        a.bits = q.select("Bits", choices=[q.Choice(f"{b}-bit", value=b) for b in BITS], default=None).ask()
        a.group_size = q.select("Group size", choices=[q.Choice(d, value=g) for g, d in GROUP_SIZES]).ask()
        if algo == "gptq":
            fam = C._family_of(C.load_model(model_key))
            a.act_order = q.confirm("GPTQ act-order? (recommended for Llama; the OPT paper used off)", default=(fam == "llama")).ask()
        if algo in ("gptq", "awq_lite"):
            a.calib_dataset = q.select("Calibration set", choices=[
                q.Choice("c4 — 128 x 2048, the GPTQ paper's", value="c4"),
                q.Choice("pile_val — 128 x 512, the AWQ paper's", value="pile_val"),
                q.Choice("wikitext2 — quick, no download", value="wikitext2")]).ask()
    a.quick = q.confirm("Quick preview? (20 windows, marked partial, not paper-comparable)", default=False).ask()
    return a


def _summary(a: Answers) -> str:
    runs = build_runs(a)
    q = runs[0].quant
    tag = "fp16" if q.algo == "fp" else f"{q.algo} {q.bits}-bit {'per-row' if q.group_size == -1 else f'g{q.group_size}'}{' act-order' if q.algo == 'gptq' and q.act_order else ''}"
    calib = f", calibration {runs[0].calib.dataset}" if runs[0].calib else ""
    return f"{a.model_key}: {tag}{calib} on {', '.join(a.datasets)}{' (quick preview)' if a.quick else ''}"


def _print_results(rows: list[dict[str, Any]]) -> None:
    from rich.console import Console
    from rich.table import Table

    table = Table(title=f"{rows[0]['model']} · {rows[0]['algo']}", show_lines=False)
    for col in ("dataset", "perplexity", "published", "Δ", "windows", "mode", "time", "peak VRAM"):
        table.add_column(col, justify="right" if col not in ("dataset", "mode") else "left")
    for r in rows:
        paper, src = paper_number(r)
        delta = f"{(r['ppl'] - paper) / paper * 100:+.2f}%" if paper else "—"
        table.add_row(r["dataset"], f"{r['ppl']:.4f}", f"{paper:.2f} ({src})" if paper else "—", delta,
                      f"{r['n_windows']}{' partial' if r['partial'] else ''}", r["eval_mode"],
                      f"{r['eval_seconds'] + float(r.get('quant_seconds') or 0):.0f}s", f"{r.get('peak_vram_gb') or 0:.2f} GB")
    Console().print(table)
    if any(r.get("_written_to") for r in rows):
        Console().print(f"[dim]rows written to {Path(rows[0]['_written_to']).parent}; `ptq aggregate && ptq plot` to refresh the tables and charts[/dim]")


def run_wizard(device_spec: str = "auto") -> int:
    import questionary as q
    from rich.console import Console

    console = Console()
    console.print("[bold]ptq wizard[/bold] — pick a model, datasets, a method and a precision. Ctrl-C to quit.\n")
    while True:
        a = _ask()
        if a is None:
            return 0
        console.print(f"\n[bold]Plan:[/bold] {_summary(a)}")
        if not q.confirm("Run it?", default=True).ask():
            return 0
        try:
            rows = execute_answers(a, device_spec=device_spec, log=console.print)
        except KeyboardInterrupt:
            console.print("[yellow]interrupted[/yellow]")
            return 130
        _print_results(rows)
        if not q.confirm("Run another?", default=True).ask():
            return 0
