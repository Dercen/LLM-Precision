"""Atomic file writes shared by every export writer.

Each writer lands its bytes in `<name>.tmp` next to the target and renames over it, so a
kill mid-export never leaves a half-written `export.json`, `weights.bin` or golden
tensor behind (the convention `runner/execute.write_row` set for result rows).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

#: Suffix of the temporary file an atomic write goes through.
TMP_SUFFIX = ".tmp"
#: Byte order of every binary file the export writes: little-endian, which x86 and
#: RISC-V both use natively.
BYTE_ORDER = "little"
#: bfloat16 has no numpy dtype; its 16-bit patterns are stored as this and the index
#: names the real dtype so a reader knows to reinterpret them.
BF16_STORAGE = np.uint16


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def numpy_little_endian(tensor: torch.Tensor) -> tuple[np.ndarray, str]:
    """A contiguous little-endian numpy view of `tensor`, and the torch dtype name to record."""
    t = tensor.detach().to("cpu").contiguous()
    arr = t.view(torch.int16).numpy().view(BF16_STORAGE) if t.dtype == torch.bfloat16 else t.numpy()
    if arr.dtype.byteorder == ">" or (arr.dtype.byteorder == "=" and sys.byteorder == "big"):
        arr = arr.byteswap().view(arr.dtype.newbyteorder("<"))
    return np.ascontiguousarray(arr), dtype_name(t.dtype)


def tensor_from_bytes(raw: bytes | memoryview, storage_dtype: str, dtype: str, shape: list[int]) -> torch.Tensor:
    """The inverse of `numpy_little_endian`: raw little-endian bytes back to a tensor."""
    arr = np.frombuffer(raw, dtype=np.dtype(storage_dtype).newbyteorder("<")).reshape(shape)
    t = torch.from_numpy(arr.copy())
    if dtype == "bfloat16":
        t = t.view(torch.int16).view(torch.bfloat16)
    return t


def write_json(path: Path, payload: Any) -> Path:
    """`json.dumps(indent=2)` of `payload`, atomically; non-JSON values via `str()`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + TMP_SUFFIX)
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
