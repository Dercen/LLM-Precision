"""`ptq doctor`: is this machine ready, in plain words, with a fix for every red line.

`ptq env-check` stays the exact, provenance-grade report (CI fails on drift). This is
the newcomer's version: a short checklist, each line green, yellow or red, and what to
do about anything that is not green. Every check is wrapped so one surprise never
hides the others.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import paths

OK, INFO, WARN, FAIL = "ok", "info", "warn", "fail"
_SYMBOL = {OK: ("✔", "green"), INFO: ("·", "dim"), WARN: ("!", "yellow"), FAIL: ("✘", "red")}


@dataclass
class Check:
    status: str
    title: str
    detail: str
    fix: str = ""


def _gb(n: float) -> str:
    return f"{n / 1024**3:.1f} GB"


# ---- the checks ---------------------------------------------------------------------------


def check_environment() -> Check:
    root = paths.repo_root()
    if Path(sys.prefix).resolve().is_relative_to(root):
        return Check(OK, "Environment", f"the project's own Python ({Path(sys.prefix).name})")
    return Check(FAIL, "Environment", f"running {sys.prefix}, not the project's .venv",
                 "run ./install.sh once, then start everything with ./ptq")


def check_gpu() -> Check:
    import torch

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        from . import device as D

        return Check(OK, "GPU", f"{props.name}, {_gb(total)} ({_gb(free)} free), driver {D._driver_version()}, torch {torch.__version__}")
    if "+cu" in torch.__version__:
        return Check(FAIL, "GPU", f"torch {torch.__version__} is built for CUDA but no GPU is usable",
                     "run ./install.sh again (it picks the build for your driver), or update the NVIDIA driver")
    return Check(WARN, "GPU", f"none; torch {torch.__version__} runs on the CPU",
                 "fine for opt-125m and opt-350m (minutes); anything bigger takes hours. A machine with an NVIDIA GPU runs the full matrix")


def check_numerics() -> Check:
    from . import device as D

    D.lock_numerics()
    if D.tf32_is_live():
        return Check(FAIL, "GPU arithmetic", "TF32 is still on, so numbers would drift from the papers",
                     "this is a torch-version problem: report it with the output of ./ptq env-check")
    return Check(OK, "GPU arithmetic", "pinned (TF32 off) so results match the papers to the 4th decimal")


def check_disk() -> Check:
    target = paths.cache_dir()
    probe = target
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    where = f"{_gb(free)} free where models are stored ({target})"
    if free < 5 * 1024**3:
        return Check(FAIL, "Disk space", where, "free some space, or point PTQ_CACHE_DIR and HF_HOME at a bigger drive (see scripts/env.sh)")
    if free < 20 * 1024**3:
        return Check(WARN, "Disk space", where, "enough for the small models; a 7B model needs 15 GB on its own")
    return Check(OK, "Disk space", where)


def check_ram() -> Check:
    import psutil

    total = psutil.virtual_memory().total
    if total < 16 * 1024**3:
        return Check(WARN, "Memory", f"{_gb(total)} of RAM", "models up to 2.7B are fine; 7B models need about 16 GB to stream")
    return Check(OK, "Memory", f"{_gb(total)} of RAM")


def _internet_ok(timeout: float = 4.0) -> bool:
    import urllib.request

    try:
        req = urllib.request.Request("https://huggingface.co", method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:  # noqa: BLE001 - any failure means "treat as offline"
        return False


def check_internet() -> Check:
    if _internet_ok():
        return Check(OK, "Internet", "huggingface.co reachable (models and text download on first use)")
    return Check(WARN, "Internet", "huggingface.co not reachable: only already-downloaded models and text can run",
                 "connect, or run `./ptq prefetch configs/experiments/<name>.yaml` while online")


def check_models() -> Check:
    from .wizard import model_choices

    models = model_choices()
    cached = [m["key"] for m in models if m["cached"]]
    if cached:
        return Check(OK, "Models downloaded", f"{', '.join(cached)} ({len(cached)} of {len(models)})")
    return Check(INFO, "Models downloaded", "none yet", "the first run fetches opt-125m (250 MB) by itself")


def check_test_text() -> Check:
    hub = paths.hf_home() / "hub"
    have = sorted(p.name.split("--", 1)[1].replace("--", "/") for p in hub.glob("datasets--*")) if hub.is_dir() else []
    if have:
        return Check(OK, "Test text downloaded", ", ".join(have))
    return Check(INFO, "Test text downloaded", "none yet", "WikiText-2 (6 MB) downloads on the first run")


def check_addons() -> Check:
    import importlib.util

    have_hqq = importlib.util.find_spec("hqq") is not None
    have_mlir = importlib.util.find_spec("torch_mlir") is not None
    parts = [f"HQQ method {'available' if have_hqq else 'missing'}",
             f"torch-mlir export {'available' if have_mlir else 'not installed (optional)'}"]
    if not have_hqq:
        return Check(WARN, "Add-ons", "; ".join(parts), "./install.sh installs HQQ; `./install.sh --with-mlir` adds the export")
    return Check(OK if have_mlir else INFO, "Add-ons", "; ".join(parts), "" if have_mlir else "`./install.sh --with-mlir` adds the export add-on")


def check_results() -> Check:
    runs = paths.runs_dir()
    n = sum(1 for _ in runs.glob("*.json")) if runs.is_dir() else 0
    return Check(INFO, "Results so far", f"{n} measurements in results/raw/runs" if n else "none yet; `./ptq` makes the first one")


CHECKS: list[Callable[[], Check]] = [
    check_environment, check_gpu, check_numerics, check_disk, check_ram,
    check_internet, check_models, check_test_text, check_addons, check_results,
]


def run_checks(*, probe_internet: bool = True) -> list[Check]:
    out: list[Check] = []
    for fn in CHECKS:
        if fn is check_internet and not probe_internet:
            out.append(Check(INFO, "Internet", "not checked (--offline)"))
            continue
        try:
            out.append(fn())
        except Exception as exc:  # noqa: BLE001 - a broken check is reported, never fatal
            out.append(Check(WARN, fn.__name__.removeprefix("check_").replace("_", " ").capitalize(),
                             f"could not be checked ({type(exc).__name__}: {str(exc)[:80]})"))
    return out


def print_report(checks: list[Check], *, console=None) -> int:
    """Print the checklist; return 1 if anything is red, else 0."""
    from rich.console import Console

    console = console or Console()
    console.print("[bold]ptq doctor[/bold]\n")
    for c in checks:
        sym, colour = _SYMBOL[c.status]
        console.print(f"[{colour}]{sym}[/{colour}] [bold]{c.title}:[/bold] {c.detail}")
        if c.fix:
            console.print(f"    [dim]→[/dim] {c.fix}")
    fails = sum(1 for c in checks if c.status == FAIL)
    warns = sum(1 for c in checks if c.status == WARN)
    console.print()
    if fails:
        console.print(f"[red]{fails} problem(s) to fix first.[/red]")
        return 1
    if warns:
        console.print(f"[yellow]Ready, with {warns} note(s) above.[/yellow] ./ptq opens the menu.")
    else:
        console.print("[green]All good.[/green] ./ptq opens the menu.")
    return 0
