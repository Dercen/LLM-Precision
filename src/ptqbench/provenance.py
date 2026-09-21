"""Everything that must be recorded alongside a perplexity number to make it reproducible."""

from __future__ import annotations

import platform
import socket
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from importlib import metadata
from typing import Any

from . import paths

# Packages whose version can move a perplexity number.
TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "tokenizers",
    "datasets",
    "accelerate",
    "safetensors",
    "huggingface-hub",
    "numpy",
    "sentencepiece",
    "hqq",
    "bitsandbytes",
    "gptqmodel",
    "torchao",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@lru_cache(maxsize=1)
def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(paths.repo_root()), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode == 0:
            sha = out.stdout.strip()
            return f"{sha}{'-dirty' if _git_dirty() else ''}"
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        pass
    return "unknown"


def _git_dirty() -> bool:
    out = subprocess.run(
        ["git", "-C", str(paths.repo_root()), "status", "--porcelain"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return bool(out.stdout.strip())


@lru_cache(maxsize=1)
def package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


@lru_cache(maxsize=1)
def platform_info() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": _cpu_count(),
    }


def _cpu_count() -> int | None:
    try:
        import psutil

        return psutil.cpu_count(logical=False)
    except Exception:  # pragma: no cover
        import os

        return os.cpu_count()


def hf_model_revision(repo_id: str) -> str:
    """The resolved commit sha of a Hub repo, so a row names exact weights."""
    try:
        from huggingface_hub import model_info

        return model_info(repo_id).sha or "unknown"
    except Exception:  # offline, gated, or hub error -- never fatal
        return "unknown"


def block() -> dict[str, Any]:
    """The provenance fields shared by every result row."""
    return {
        "git_sha": git_sha(),
        "versions": package_versions(),
        **platform_info(),
        "generated_at": utc_now(),
    }
