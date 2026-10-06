"""`ptq wizard` (also plain `ptq`): pick a model, the text to test on, a method and a
precision from menus, confirm, run, and read the result next to the published number
with a one-sentence verdict.

The prompts only fill in `Answers`; `build_runs()` turns answers into the same RunSpecs
`ptq run` executes, so a wizard row is indistinguishable from a matrix row -- same ids,
same schema, same provenance. Everything below the prompt layer is unit-testable. The
menu text lives in `ptqbench.explain`, shared with the by-model pages.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config as C
from . import explain as E
from . import paths

BITS = [b for b, _ in E.BITS_MENU]


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


def recommended_group_size(model_key: str) -> int:
    """Groups of 128 unless the model's config lists another size (SmolLM2: 64, since 128
    does not divide its hidden size of 576)."""
    spec = C.load_model(model_key)
    return spec.extra_group_sizes[0] if spec.extra_group_sizes else E.RECOMMENDED_GROUP_SIZE


DEMO_MODEL = "opt-125m"


def demo_answers() -> list[Answers]:
    """The two-minute path: the unmodified model, then 4-bit plain rounding, both as
    quick previews on WikiText-2, so the second run can be read against the first."""
    return [
        Answers(model_key=DEMO_MODEL, datasets=["wikitext2"], algo="fp", quick=True),
        Answers(model_key=DEMO_MODEL, datasets=["wikitext2"], algo="rtn", bits=4,
                group_size=recommended_group_size(DEMO_MODEL), quick=True),
    ]


def _minutes(seconds: float) -> str:
    if seconds < 60:
        return "under a minute"
    if seconds < 90 * 60:
        return f"about {max(1, round(seconds / 60))} minute{'s' if round(seconds / 60) != 1 else ''}"
    return f"about {seconds / 3600:.1f} hours"


def estimate(a: Answers, device_spec: str = "auto") -> str:
    """'about 3 minutes; downloads once: opt-125m (0.25 GB)', from results/raw/timing.json."""
    from . import device as D
    from .models import loader as ml
    from .runner import execute as X
    from .runner import matrix as M

    device = D.resolve(device_spec)
    runs = [X.resolve_run(r, device) for r in build_runs(a)]
    seconds, _notes = M.estimate_seconds([(runs[0].quant_key, runs)], include_done=True)
    if device.type == "cpu":
        # The timing data is from a GPU. Measured on opt-125m: the CPU takes 50x longer per
        # window (1.16 s vs 0.023 s) while loading stays a few seconds, so only the part
        # beyond the estimator's fixed 20 s of load and overhead is scaled.
        seconds = 20.0 + (seconds - 20.0) * 50
    text = _minutes(seconds)
    cached = next((m["cached"] for m in model_choices() if m["key"] == a.model_key), False)
    if not cached:
        try:
            size = ml.estimate_model_bytes(runs[0].model.repo, ml.parse_dtype(runs[0].dtype.replace("torch.", "")) or __import__("torch").float16)
            text += f"; downloads once: {a.model_key} ({size / 1024**3:.2f} GB) and the test text"
        except Exception:  # noqa: BLE001 - offline or unknown model: still say a download is coming
            text += f"; downloads once: {a.model_key} and the test text"
    return text


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
    q = runs[0].quant
    what = "loading" if q.algo == "fp" else f"loading, then quantizing with {E.method_name(q.algo)} at {q.bits} bits"
    log(f"[dim]Step 1:[/dim] {what}: {a.model_key} on {device}")
    prep = X.prepare(runs[0], device)
    rows = []
    for i, run in enumerate(runs, start=2):
        passages = "20 passages (quick preview)" if a.quick else "every passage"
        log(f"[dim]Step {i}:[/dim] measuring on {E.dataset_name(run.dataset)}, {passages}")
        row = X.evaluate(prep, run, progress=sys.stderr.isatty())
        if write:
            row["_written_to"] = X.write_row(row)
        rows.append(row)
    return rows


# ---- the prompt layer ---------------------------------------------------------------


def _model_label(m: dict[str, Any]) -> str:
    state = "already downloaded" if m["cached"] else "will download"
    gated = "  (gated: an open mirror is used)" if m["gated"] else ""
    return f"{m['key']:14s} {m['repo']:34s} {state}{gated}"


def _ask() -> Answers | None:
    import questionary as q

    models = model_choices()
    model_key = q.select(
        "Which model? (arrow keys, enter)",
        choices=[q.Choice(_model_label(m), value=m["key"]) for m in models], use_shortcuts=False,
    ).ask()
    if model_key is None:
        return None
    datasets = q.checkbox(
        "Which text should it be tested on? (space to toggle, enter to confirm)",
        choices=[q.Choice(f"{E.dataset_name(k):34s} {blurb}", value=k, checked=(k == "wikitext2")) for k, blurb in E.DATASET_MENU],
    ).ask()
    if not datasets:
        return None
    algo = q.select(
        "Which method?",
        choices=[q.Choice(f"{label:22s} {blurb}", value=k) for k, label, blurb in E.METHOD_MENU],
    ).ask()
    if algo is None:
        return None
    a = Answers(model_key=model_key, datasets=datasets, algo=algo)
    if algo != "fp":
        bit_choices = [q.Choice(f"{b}-bit   {blurb}", value=b) for b, blurb in E.BITS_MENU]
        a.bits = q.select("How many bits per weight?", choices=bit_choices,
                          default=next(c for c in bit_choices if c.value == E.RECOMMENDED_BITS)).ask()
        a.group_size = recommended_group_size(model_key)
        a.act_order = None  # the family default: on for Llama-style models
        a.calib_dataset = "c4"
        advanced = q.confirm(
            "Advanced settings? (grouping, GPTQ act-order, calibration text; the defaults are the recommended ones)",
            default=False,
        ).ask()
        if advanced:
            g_choices = [q.Choice(blurb, value=g) for g, blurb in E.GROUPING_MENU]
            a.group_size = q.select("Grouping: how many weights share one scale factor?", choices=g_choices,
                                    default=next((c for c in g_choices if c.value == a.group_size), None)).ask()
            if algo == "gptq":
                fam = C._family_of(C.load_model(model_key))
                a.act_order = q.confirm(
                    "GPTQ act-order: round the most important columns first? (recommended for Llama-style models; the OPT paper had it off)",
                    default=(fam == "llama"),
                ).ask()
            if algo in ("gptq", "awq_lite"):
                a.calib_dataset = q.select(
                    "Calibration text: the sample passages the method studies before rounding",
                    choices=[q.Choice(f"{k:10s} {blurb}", value=k) for k, blurb in E.CALIB_MENU],
                ).ask()
    a.quick = q.confirm(
        "Quick preview? Reads 20 passages instead of all of them: faster, but approximate and not compared with the papers",
        default=False,
    ).ask()
    return a


def _summary(a: Answers) -> str:
    runs = build_runs(a)
    q = runs[0].quant
    if q.algo == "fp":
        setting = "full precision (no quantization)"
    else:
        setting = f"{E.method_name(q.algo)}, {q.bits} bits, {E.grouping(q.group_size)}"
        if q.algo == "gptq" and q.act_order:
            setting += ", act-order"
        if runs[0].calib:
            setting += f", calibration text {runs[0].calib.dataset}"
    texts = ", ".join(E.dataset_name(d) for d in a.datasets)
    return f"{a.model_key}: {setting}; tested on {texts}{', quick preview' if a.quick else ''}"


def _print_results(rows: list[dict[str, Any]]) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"{rows[0]['model']} · {E.method_name(rows[0]['algo'])}", show_lines=False)
    for col in ("text", "perplexity (lower is better)", "paper's number", "vs paper", "passages", "ran", "time", "GPU memory"):
        table.add_column(col, justify="left" if col in ("text", "ran") else "right")
    for r in rows:
        paper, src = paper_number(r)
        delta = f"{(r['ppl'] - paper) / paper * 100:+.2f}%" if paper else "—"
        table.add_row(E.dataset_name(r["dataset"]), f"{r['ppl']:.4f}", f"{paper:.2f} ({src})" if paper else "—", delta,
                      f"{r['n_windows']}{' (preview)' if r['partial'] else ''}", r["eval_mode"],
                      f"{r['eval_seconds'] + float(r.get('quant_seconds') or 0):.0f}s", f"{r.get('peak_vram_gb') or 0:.2f} GB")
    console.print(table)
    for r in rows:
        baseline = E.baseline_ppl(r["model"], r["dataset"], allow_partial=bool(r["partial"]))
        console.print(f"[bold]What this means:[/bold] {E.verdict(r, baseline)}")
    if any(r.get("_written_to") for r in rows):
        console.print(f"[dim]saved to {Path(rows[0]['_written_to']).parent}; `ptq aggregate && ptq plot` refreshes the tables and charts[/dim]")


def run_wizard(device_spec: str = "auto") -> int:
    import questionary as q
    from rich.console import Console

    from . import device as D
    from .runner import execute as X

    console = Console()
    if not sys.stdin.isatty():
        console.print("ptq wizard needs an interactive terminal: run it from a terminal window, "
                      "or use `ptq eval ...` for a one-line run.")
        return 2
    console.print("[bold]ptq wizard[/bold] — measure how much a model loses when its weights are squeezed to fewer bits. Ctrl-C to quit.\n")
    device = D.resolve(device_spec)
    while True:
        mode = q.select(
            "What would you like to do?",
            choices=[
                q.Choice("Show me something in two minutes  (opt-125m: the unmodified model, then 4-bit plain rounding; quick previews)", value="demo"),
                q.Choice("Choose the model, text, method and bits myself", value="custom"),
                q.Choice("Quit", value="quit"),
            ],
        ).ask()
        if mode in (None, "quit"):
            return 0
        plan = demo_answers() if mode == "demo" else [_ask()]
        if plan[0] is None:
            continue
        console.print()
        for a in plan:
            console.print(f"[bold]Plan:[/bold] {_summary(a)}")
            console.print(f"       [dim]estimated time: {estimate(a, device_spec)}[/dim]")
        ok, reason = X.check_available(build_runs(plan[-1])[0], device)
        if not ok:
            console.print(f"[yellow]This cannot run:[/yellow] {E.explain_reason(reason)}\n")
            continue
        if not q.confirm("Run it?", default=True).ask():
            continue
        failed = False
        for a in plan:
            try:
                rows = execute_answers(a, device_spec=device_spec, log=console.print)
            except KeyboardInterrupt:
                console.print("[yellow]interrupted[/yellow]")
                return 130
            except Exception as exc:  # noqa: BLE001 - every failure gets a sentence, not a traceback
                console.print(f"[red]The run failed.[/red] {E.explain_reason(f'{type(exc).__name__}: {exc}')}")
                failed = True
                break
            _print_results(rows)
        if failed:
            if not q.confirm("Try another?", default=True).ask():
                return 1
            continue
        if not q.confirm("Run another?", default=True).ask():
            return 0
