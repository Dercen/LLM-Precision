"""Which algorithms exist, what they need, and why a row is skipped. PLAN.md 6, 9.

An unavailable backend never aborts a matrix: the runner writes `status=skipped`
rows with the reason, which `--rerun-incomplete` retries once the backend exists.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Backend:
    name: str
    requires: tuple[str, ...] = ()  # importable module names
    devices: tuple[str, ...] = ("cpu", "cuda")
    data_dependent: bool = False
    bits: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)

    def check(self, device_type: str, bits: int | None = None) -> tuple[bool, str | None]:
        if device_type not in self.devices:
            return False, f"device_unsupported:{device_type}"
        if bits is not None and self.name != "fp" and bits not in self.bits:
            return False, f"bits_unsupported:{bits}"
        for mod in self.requires:
            try:
                importlib.import_module(mod)
            except ImportError as exc:
                return False, f"import_error:{mod}:{exc.__class__.__name__}"
        return True, None


BACKENDS: dict[str, Backend] = {
    "fp": Backend("fp", bits=(16,)),
    "rtn": Backend("rtn"),
    "gptq": Backend("gptq", data_dependent=True),
    "awq_lite": Backend("awq_lite", data_dependent=True),
    "hqq": Backend("hqq", requires=("hqq.core.quantize",), bits=(2, 3, 4, 8)),
}


def check(algo: str, device_type: str, bits: int | None = None) -> tuple[bool, str | None]:
    backend = BACKENDS.get(algo)
    if backend is None:
        return False, f"unknown_algo:{algo}"
    return backend.check(device_type, bits)
