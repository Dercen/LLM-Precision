"""Sole owner of filesystem locations. Every path in ptqbench comes from here."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """The repository root, found by walking up from this file to the pyproject.toml."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("could not locate repo root (no pyproject.toml above ptqbench/paths.py)")


def _env_dir(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw).expanduser() if raw else default


def _scratch_root() -> Path:
    scratch = os.environ.get("SCRATCH")
    return Path(scratch).expanduser() if scratch else Path.home() / "ml"


def hf_home() -> Path:
    return _env_dir("HF_HOME", _scratch_root() / "hf")


def cache_dir() -> Path:
    return _env_dir("PTQ_CACHE_DIR", _scratch_root() / "ptq-cache")


def calib_cache_dir() -> Path:
    return cache_dir() / "calib"


def quant_cache_dir() -> Path:
    return cache_dir() / "quant"


def tokenized_cache_dir() -> Path:
    return cache_dir() / "tokenized"


def dataset_cache_dir() -> Path:
    return cache_dir() / "datasets"


def results_dir() -> Path:
    return repo_root() / "results"


def runs_dir() -> Path:
    return results_dir() / "runs"


def logs_dir() -> Path:
    return results_dir() / "logs"


def plots_dir() -> Path:
    return results_dir() / "plots"


def timing_file() -> Path:
    return results_dir() / "timing.json"


def configs_dir() -> Path:
    return repo_root() / "configs"


def references_dir() -> Path:
    return repo_root() / "references"


def ensure_dirs() -> None:
    """Create every directory ptqbench writes into."""
    for d in (
        hf_home(),
        cache_dir(),
        calib_cache_dir(),
        quant_cache_dir(),
        tokenized_cache_dir(),
        dataset_cache_dir(),
        runs_dir(),
        logs_dir(),
        plots_dir(),
    ):
        d.mkdir(parents=True, exist_ok=True)


def describe() -> dict[str, str]:
    """Every resolved location, for `ptq env-check` and provenance."""
    return {
        "repo_root": str(repo_root()),
        "hf_home": str(hf_home()),
        "ptq_cache_dir": str(cache_dir()),
        "calib_cache": str(calib_cache_dir()),
        "quant_cache": str(quant_cache_dir()),
        "results_dir": str(results_dir()),
    }
