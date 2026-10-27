"""Atomic file writes shared by every export writer.

Each writer lands its bytes in `<name>.tmp` next to the target and renames over it, so a
kill mid-export never leaves a half-written `export.json`, `weights.bin` or golden
tensor behind (the convention `runner/execute.write_row` set for result rows).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

#: Suffix of the temporary file an atomic write goes through.
TMP_SUFFIX = ".tmp"


def write_json(path: Path, payload: Any) -> Path:
    """`json.dumps(indent=2)` of `payload`, atomically; non-JSON values via `str()`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + TMP_SUFFIX)
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def write_bytes(path: Path, data: bytes | memoryview) -> Path:
    """Raw bytes, atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + TMP_SUFFIX)
    tmp.write_bytes(data)
    tmp.replace(path)
    return path


def sha256_of(data: bytes | memoryview) -> str:
    """Hex digest used to fingerprint every binary file the export writes."""
    return hashlib.sha256(data).hexdigest()
