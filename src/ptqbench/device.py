"""Device resolution, dtype policy, and the numerical-determinism lockdown of PLAN.md 5a.

TF32 matmuls carry ~10 mantissa bits and move perplexity in the third decimal --
larger than the +/-0.02 tolerance the reference numbers in PLAN.md 13 are stated to.
`lock_numerics()` must run before any model is built, and the flags it reports are
recorded in every result row.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch

# Model-family dtype policy (PLAN.md 2). CPU always gets fp32.
_BF16_FAMILIES = ("llama-3", "llama3", "meta-llama-3")


@dataclass(frozen=True)
class NumericsState:
    """What lock_numerics() actually set, for provenance and `ptq env-check`."""

    tf32: bool
    fp16_reduced_reduction: bool
    cudnn_benchmark: bool
    deterministic: bool
    float32_matmul_precision: str
    api: str = field(default="")
    notes: tuple[str, ...] = field(default_factory=tuple)
    knobs: dict[str, str] = field(default_factory=dict)

    def as_row_fields(self) -> dict[str, Any]:
        return {
            "tf32": self.tf32,
            "deterministic": self.deterministic,
            "float32_matmul_precision": self.float32_matmul_precision,
            "fp16_reduced_precision_reduction": self.fp16_reduced_reduction,
            "fp32_precision_knobs": dict(self.knobs),
        }


# torch >= 2.9 replaced the `allow_tf32` booleans with a string `fp32_precision`
# hierarchy: a specific knob (cudnn.conv) falls back to its backend (cudnn), which
# falls back to the global (torch.backends). Reading `allow_tf32` AFTER the new API
# has been touched raises when the sub-knobs disagree, so the two APIs are never
# mixed -- the new one is used exclusively wherever it exists.
_FP32_KNOBS = (
    ("global", lambda: torch.backends),
    ("cuda.matmul", lambda: torch.backends.cuda.matmul),
    ("cudnn.conv", lambda: torch.backends.cudnn.conv),
    ("cudnn.rnn", lambda: torch.backends.cudnn.rnn),
)


def _has_new_fp32_api() -> bool:
    return hasattr(torch.backends, "fp32_precision")


def _set_fp32_precision(enable_tf32: bool) -> tuple[str, list[str]]:
    """Pin fp32 matmul/conv precision across every knob this torch exposes.

    On torch 2.14 `cudnn.conv` and `cudnn.rnn` ship as 'tf32' and `cuda.matmul`
    defers to an unset global, so leaving the defaults alone means TF32 is live
    for part of the stack. Everything is pinned explicitly instead.
    """
    notes: list[str] = []
    used: list[str] = []

    if _has_new_fp32_api():
        value = "tf32" if enable_tf32 else "ieee"
        for name, getter in _FP32_KNOBS:
            try:
                target = getter()
            except AttributeError:  # pragma: no cover - torch-version dependent
                continue
            if not hasattr(target, "fp32_precision"):
                continue
            try:
                target.fp32_precision = value
                used.append(f"{name}={value}")
            except (RuntimeError, ValueError, TypeError) as exc:
                # cudnn.rnn rejects 'ieee' on some builds; the global still covers it.
                notes.append(f"{name}.fp32_precision rejected {value!r}: {exc}")
        return "fp32_precision(" + ", ".join(used) + ")", notes

    # Legacy torch (< 2.9): the boolean API is the only one available.
    for mod, name in ((torch.backends.cuda.matmul, "cuda.matmul"), (torch.backends.cudnn, "cudnn")):
        if hasattr(mod, "allow_tf32"):
            mod.allow_tf32 = enable_tf32
            used.append(f"{name}.allow_tf32={enable_tf32}")
    return "allow_tf32(" + ", ".join(used) + ")", notes


def read_fp32_precision() -> dict[str, str]:
    """Current precision of every knob, read through whichever API is in play."""
    if _has_new_fp32_api():
        out: dict[str, str] = {}
        for name, getter in _FP32_KNOBS:
            try:
                out[name] = str(getter().fp32_precision)
            except (AttributeError, RuntimeError):  # pragma: no cover
                continue
        return out
    return {"cuda.matmul.allow_tf32": str(torch.backends.cuda.matmul.allow_tf32)}


def tf32_is_live() -> bool:
    """True if any knob would still use TF32. `ptq env-check` fails on this."""
    return any(v == "tf32" for v in read_fp32_precision().values())


def lock_numerics(*, deterministic: bool = False, allow_tf32: bool = False) -> NumericsState:
    """Pin every knob that would otherwise make perplexity hardware-dependent.

    Also initialises the CUDA context. The streamed tier never puts a whole model on
    the device, so without this the first CUDA call can be a memory-stats query, which
    raises "Invalid device argument" on an uninitialised context (found by the M5
    opt-6.7b gate, 2026-09-22; every earlier streamed run had started from a model
    that had already been resident).
    """
    notes: list[str] = []
    if torch.cuda.is_available():
        torch.cuda.init()

    api, api_notes = _set_fp32_precision(allow_tf32)
    notes.extend(api_notes)

    # fp16 GEMMs must accumulate in fp32, or a long reduction drifts.
    if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    # Autotuning picks different kernels run to run; perplexity must not depend on that.
    torch.backends.cudnn.benchmark = False

    precision = "highest" if not allow_tf32 else "high"
    torch.set_float32_matmul_precision(precision)

    if deterministic:
        # cuBLAS needs this set before its first handle is created to be deterministic.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True

    return NumericsState(
        tf32=tf32_is_live(),
        fp16_reduced_reduction=False,
        cudnn_benchmark=False,
        deterministic=deterministic,
        float32_matmul_precision=precision,
        api=api,
        notes=tuple(notes),
        knobs=read_fp32_precision(),
    )


def resolve(spec: str = "auto") -> torch.device:
    """'auto' -> cuda:0 when available, else cpu. 'cuda', 'cuda:N' and 'cpu' pass through."""
    if spec == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {spec!r} but torch.cuda.is_available() is False")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    return device


def dtype_for(model_id: str, device: torch.device) -> torch.dtype:
    """PLAN.md 2: fp16 for OPT and Llama-2, bf16 for Llama-3, fp32 on CPU."""
    if device.type == "cpu":
        return torch.float32
    lowered = model_id.lower()
    if any(tag in lowered for tag in _BF16_FAMILIES):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float16


def describe(device: torch.device) -> dict[str, Any]:
    """Device facts for `ptq env-check` and the provenance block of every row."""
    info: dict[str, Any] = {
        "device": str(device),
        "device_type": device.type,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        free, total = torch.cuda.mem_get_info(idx)
        info.update(
            {
                "device_name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "vram_total_gb": round(total / 1024**3, 2),
                "vram_free_gb": round(free / 1024**3, 2),
                "multi_processor_count": props.multi_processor_count,
                "driver_version": _driver_version(),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    return info


def _driver_version() -> str:
    try:  # torch exposes it on recent CUDA builds; nvidia-smi is the fallback
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0].strip()
    except (OSError, ValueError, subprocess.SubprocessError):  # pragma: no cover
        pass
    return "unknown"


# Headroom for the CUDA context, activations and fragmentation, on top of weights.
VRAM_RESERVE_GB = 0.8


def reset_peak_memory(device: torch.device) -> None:
    """reset_peak_memory_stats that is safe before any tensor has touched the device."""
    if device.type != "cuda":
        return
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)


def free_vram_bytes(device: torch.device) -> int:
    """Instantaneous free VRAM, as the driver reports it."""
    if device.type != "cuda":
        return 0
    return torch.cuda.mem_get_info(device.index or 0)[0]


def reclaimable_vram_bytes(device: torch.device) -> int:
    """Blocks torch's caching allocator holds but is not using -- free on demand."""
    if device.type != "cuda":
        return 0
    idx = device.index or 0
    return max(0, torch.cuda.memory_reserved(idx) - torch.cuda.memory_allocated(idx))


def vram_available_bytes(device: torch.device) -> int:
    """What a new allocation could actually get right now.

    `mem_get_info` alone under-reports, because torch's caching allocator keeps freed
    blocks reserved. Ignoring that makes a resident/streamed decision depend on what
    ran before it -- so cached blocks are counted as available.
    """
    if device.type != "cuda":
        return 0
    return free_vram_bytes(device) + reclaimable_vram_bytes(device)


def vram_capacity_bytes(device: torch.device, *, reserve_gb: float = VRAM_RESERVE_GB) -> int:
    """What the card could hold on an empty machine: a property of the hardware.

    Used for planning (`--dry-run`, tiering in PLAN.md 2a), where the answer must not
    depend on what happens to be loaded at the moment the question is asked.
    """
    if device.type != "cuda":
        return 0
    total = torch.cuda.mem_get_info(device.index or 0)[1]
    return max(0, int(total - reserve_gb * 1024**3))


def should_stream(
    model_bytes: int,
    device: torch.device,
    *,
    factor: float = 1.3,
    basis: str = "available",
) -> bool:
    """PLAN.md 2a: stream when the VRAM budget is below 1.3 x model bytes.

    basis="available" is the live load decision; basis="capacity" asks the
    hardware-only question and is what planning and tests use.
    """
    if device.type != "cuda":
        return False
    if basis == "capacity":
        budget = vram_capacity_bytes(device)
    elif basis == "available":
        budget = vram_available_bytes(device)
    else:
        raise ValueError(f"basis must be 'available' or 'capacity', got {basis!r}")
    return budget < factor * model_bytes
