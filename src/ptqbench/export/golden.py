"""Golden test vectors: for every exported function, a fixed realistic input, the output
PyTorch produces for it, and the tolerance a consumer should compare with.

Purpose: a run of the same function somewhere else -- an SST simulation of the vault
hardware, a C harness, another compiler -- can be checked against the real model.

Layout, under `<out_dir>/golden/`:

    index.json                       one entry per function: tensors, source, tolerance
    <function>/<tensor>.npy          numpy format, dtype and shape included
    <function>/<tensor>.bin          the same bytes raw, little-endian, row-major, no header

Inputs are never random noise. The whole model and the stages take real token ids from
a seeded window of the benchmark's own wikitext2 stream (`realistic_window`); a slice
takes the hidden states its block actually received on that window (`export/slices.py`).

The tolerance is derived, not guessed. The function is run a second time in float64
on the *same* weights -- the fp32 dequantized values, which float64 holds exactly -- so
the distance between the two outputs is purely the fp32 rounding of the reference
itself (`measure_against_fp64`). A consumer's own fp32 implementation accumulates
rounding of the same size in a different order, so the recommended absolute tolerance
is that measured distance times `TOLERANCE_SAFETY_FACTOR`, with a floor of the same
factor in last-place units of the largest output value (for outputs the reference
computes exactly). Both measured numbers, the factor and the resulting tolerances are
written to the index so the derivation can be audited or redone.
"""

from __future__ import annotations

import copy
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..data import datasets as ds
from . import files as F
from .quantized import QuantizedLinear

#: Directory under the export holding the vectors, and the index file inside it.
GOLDEN_DIR = "golden"
INDEX_NAME = "index.json"
#: Bumped when the index schema changes.
GOLDEN_FORMAT_VERSION = 1

#: Per-tensor file suffixes.
NPY_SUFFIX = ".npy"
BIN_SUFFIX = ".bin"
#: Byte order of every `.bin`: little-endian, which x86 and RISC-V both use natively.
BYTE_ORDER = "little"

#: Default source of real token ids and the seed that picks the window.
DEFAULT_GOLDEN_DATASET = "wikitext2"
DEFAULT_GOLDEN_SEED = 0
#: The window length the benchmark's cached token streams are cut into. Golden inputs
#: are prefixes of one such window, so reusing it means no new tokenisation and the
#: same text the perplexity rows were measured on.
GOLDEN_SOURCE_SEQLEN = 2048

#: Multiplier between the reference's measured fp32-vs-fp64 distance and the tolerance a
#: consumer should use. 2x covers the consumer's own, independent fp32 rounding (two
#: fp32 evaluations can each sit on the opposite side of the exact value); the other 4x
#: covers a different accumulation order, fused multiply-adds and a different `exp` /
#: `rsqrt` implementation in a bare-metal libm, each worth a few last-place units.
TOLERANCE_SAFETY_FACTOR = 8.0
#: Floor on the absolute tolerance, in units of the largest output magnitude: one fp32
#: last-place unit (2**-23) times the safety factor. Without it an output the reference
#: happens to compute exactly (a copied cache slot, a zero) would get a tolerance of 0.
FP32_EPSILON = 2.0**-23

#: Roles a tensor plays in a golden case.
ROLE_INPUT = "input"
ROLE_OUTPUT = "output"


# ---- tensors on disk --------------------------------------------------------------------


def _numpy_little_endian(tensor: torch.Tensor) -> tuple[np.ndarray, str]:
    """A contiguous little-endian numpy view of `tensor`, and the dtype name recorded.

    bfloat16 has no numpy dtype: its 16-bit patterns are stored as `uint16` and the index
    says `bfloat16` so a reader knows to reinterpret them.
    """
    t = tensor.detach().to("cpu").contiguous()
    dtype_name = str(t.dtype).replace("torch.", "")
    arr = t.view(torch.int16).numpy().view(np.uint16) if t.dtype == torch.bfloat16 else t.numpy()
    if arr.dtype.byteorder == ">" or (arr.dtype.byteorder == "=" and sys.byteorder == "big"):
        arr = arr.byteswap().view(arr.dtype.newbyteorder("<"))
    return np.ascontiguousarray(arr), dtype_name


def write_tensor(directory: Path, name: str, tensor: torch.Tensor, *, role: str) -> dict[str, Any]:
    """Write `<directory>/<name>.npy` and `.bin` and return the index entry for both."""
    directory = Path(directory)
    arr, dtype_name = _numpy_little_endian(tensor)
    raw = arr.tobytes()
    npy_path = directory / f"{name}{NPY_SUFFIX}"
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = npy_path.with_name(npy_path.name + F.TMP_SUFFIX)
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    tmp.replace(npy_path)
    bin_path = F.write_bytes(directory / f"{name}{BIN_SUFFIX}", raw)
    return {
        "name": name,
        "role": role,
        "shape": list(tensor.shape),
        "dtype": dtype_name,
        "storage_dtype": str(arr.dtype),
        "byte_length": len(raw),
        "npy": npy_path.name,
        "bin": bin_path.name,
        "sha256": F.sha256_of(raw),
    }


def read_tensor(directory: Path, entry: dict[str, Any]) -> torch.Tensor:
    """The tensor an index entry describes, from its `.bin` (the header-less file)."""
    raw = (Path(directory) / entry["bin"]).read_bytes()
    arr = np.frombuffer(raw, dtype=np.dtype(entry["storage_dtype"]).newbyteorder("<")).reshape(entry["shape"])
    t = torch.from_numpy(arr.copy())
    if entry["dtype"] == "bfloat16":
        t = t.view(torch.int16).view(torch.bfloat16)
    return t


# ---- realistic inputs -------------------------------------------------------------------


def realistic_window(tokenizer, dataset: str, seed: int, length: int) -> tuple[torch.Tensor, dict[str, Any]]:
    """`length` real token ids: a prefix of one seeded window of the benchmark's stream.

    The window index is drawn from `random.Random(seed)` over the stream's windows, so
    the same seed, tokenizer and dataset always give the same text. Returns the ids as
    `[1, length]` int64 and a description for the index.
    """
    stream = ds.build(dataset, tokenizer, seqlen=GOLDEN_SOURCE_SEQLEN)
    if length > stream.seqlen:
        raise ValueError(f"a golden input of {length} tokens exceeds the {stream.seqlen}-token source windows")
    index = random.Random(seed).randrange(stream.n_windows)
    ids = stream.window(index)[:, :length].to(torch.int64).contiguous()
    source = {
        "kind": "token_ids",
        "dataset": dataset,
        "split": stream.split,
        "join": stream.join,
        "seed": seed,
        "window_index": index,
        "n_windows": stream.n_windows,
        "source_seqlen": stream.seqlen,
        "length": length,
        "first_token_hash": stream.first_token_hash(),
    }
    return ids, source


# ---- the float64 reference and the tolerance ---------------------------------------------


def _set_submodule(root: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, attr = name.rpartition(".")
    setattr(root.get_submodule(parent_name) if parent_name else root, attr, new)


def fp64_reference(fn: nn.Module) -> nn.Module:
    """A float64 copy of `fn` that computes with the *same* weights.

    Every QuantizedLinear becomes a plain Linear holding its dequantized weight -- the
    fp32 (or fp16) values the evaluation ran with, cast to float64 exactly -- and
    everything else is cast to float64. So the only thing that differs from `fn` is the
    precision of the arithmetic on activations, which is what the tolerance measures.
    """
    ref = copy.deepcopy(fn)
    for name, mod in list(ref.named_modules()):
        if isinstance(mod, QuantizedLinear):
            lin = nn.Linear(mod.in_features, mod.out_features, bias=mod.bias is not None)
            lin.weight = nn.Parameter(mod.dequantize().to(torch.float64), requires_grad=False)
            if mod.bias is not None:
                lin.bias = nn.Parameter(mod.bias.detach().to(torch.float64), requires_grad=False)
            _set_submodule(ref, name, lin)
    return ref.to(torch.float64).eval()


def flat_outputs(out: Any) -> list[torch.Tensor]:
    """Every tensor a forward returned, in order: a tensor, a tuple, or a ModelOutput
    (whose `None` fields torch.export drops the same way)."""
    if torch.is_tensor(out):
        return [out]
    if hasattr(out, "to_tuple"):
        out = out.to_tuple()
    if isinstance(out, (tuple, list)):
        return [t for item in out for t in flat_outputs(item)]
    return []


def cast_float_inputs(inputs: list[torch.Tensor], dtype: torch.dtype) -> list[torch.Tensor]:
    """Float inputs to `dtype`; integer inputs (token ids, positions) untouched."""
    return [t.to(dtype) if t.is_floating_point() else t for t in inputs]


@torch.no_grad()
def run(fn: nn.Module, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    return [t.detach() for t in flat_outputs(fn(*inputs))]


def tolerance_from(max_abs_err: float, max_abs_value: float, rel_fro_err: float) -> dict[str, float]:
    """The recommended tolerances for one output from its measured numbers."""
    floor = TOLERANCE_SAFETY_FACTOR * FP32_EPSILON * max_abs_value
    return {
        "atol": max(TOLERANCE_SAFETY_FACTOR * max_abs_err, floor),
        "rel_fro_tol": TOLERANCE_SAFETY_FACTOR * rel_fro_err,
    }


@torch.no_grad()
def measure_against_fp64(
    fn: nn.Module, inputs: list[torch.Tensor], outputs: list[torch.Tensor], names: list[str]
) -> list[dict[str, Any]]:
    """Per output: the fp32 reference's distance from the float64 evaluation, and the
    tolerances derived from it (see the module docstring)."""
    ref = fp64_reference(fn)
    exact = run(ref, cast_float_inputs(inputs, torch.float64))
    del ref
    if len(exact) != len(outputs):
        raise RuntimeError(f"the float64 reference returned {len(exact)} outputs, the function {len(outputs)}")
    report = []
    for name, got, want in zip(names, outputs, exact):
        g, w = got.double(), want.double()
        diff = (g - w).abs()
        max_abs_err = float(diff.max()) if diff.numel() else 0.0
        max_abs_value = float(w.abs().max()) if w.numel() else 0.0
        w_norm = float(w.norm())
        rel_fro_err = float(diff.norm()) / w_norm if w_norm > 0 else 0.0
        entry = {
            "name": name,
            "max_abs_err_vs_fp64": max_abs_err,
            "rel_fro_err_vs_fp64": rel_fro_err,
            "max_abs_value": max_abs_value,
            **tolerance_from(max_abs_err, max_abs_value, rel_fro_err),
        }
        if g.ndim >= 2 and g.shape[-1] > 1:  # a logits-like last axis: does the top choice agree?
            entry["argmax_agreement"] = float((g.argmax(-1) == w.argmax(-1)).double().mean())
        report.append(entry)
    return report


def check_within(got: torch.Tensor, want: torch.Tensor, tolerance: dict[str, float]) -> dict[str, Any]:
    """The comparison a consumer should run: max|got - want| <= atol and the relative
    Frobenius distance <= rel_fro_tol. Returns the numbers and the verdict."""
    g, w = got.double(), want.double()
    diff = (g - w).abs()
    max_abs = float(diff.max()) if diff.numel() else 0.0
    w_norm = float(w.norm())
    rel_fro = float(diff.norm()) / w_norm if w_norm > 0 else 0.0
    return {
        "max_abs_diff": max_abs,
        "rel_fro_diff": rel_fro,
        "within_tolerance": max_abs <= tolerance["atol"] and rel_fro <= tolerance["rel_fro_tol"],
    }


# ---- cases and the index ----------------------------------------------------------------


@dataclass
class GoldenCase:
    """One exported function with its vectors: `inputs` and `outputs` are (name, tensor)
    pairs in argument/result order; `source` says where the inputs came from;
    `tolerance` is the per-output report from `measure_against_fp64` (None if skipped)."""

    function: str
    inputs: list[tuple[str, torch.Tensor]]
    outputs: list[tuple[str, torch.Tensor]]
    source: dict[str, Any]
    tolerance: list[dict[str, Any]] | None = None
    notes: list[str] = field(default_factory=list)


def build_case(
    function: str,
    fn: nn.Module,
    inputs: list[tuple[str, torch.Tensor]],
    output_names: list[str],
    *,
    source: dict[str, Any],
    measure: bool = True,
) -> GoldenCase:
    """Run `fn` on the inputs, name its outputs, and derive the tolerance."""
    tensors = [t for _, t in inputs]
    outputs = run(fn, tensors)
    if len(outputs) != len(output_names):
        raise RuntimeError(f"{function}: expected {len(output_names)} outputs {output_names}, got {len(outputs)}")
    case = GoldenCase(function, inputs, list(zip(output_names, outputs)), source)
    if measure:
        case.tolerance = measure_against_fp64(fn, tensors, outputs, output_names)
    else:
        case.notes.append("tolerance not measured: the float64 reference was skipped")
    return case


def fp64_reference_fits(model_bytes: int) -> bool:
    """Whether a float64 copy of a model of `model_bytes` fits next to it in host RAM."""
    try:
        import psutil
    except ImportError:  # pragma: no cover
        return True
    need = FP64_COPY_FACTOR * model_bytes
    return psutil.virtual_memory().available > need


#: Host RAM a float64 copy needs relative to the fp32 model: twice the bytes, plus the
#: fp32 original that stays resident, plus activations -- 3x is the guard.
FP64_COPY_FACTOR = 3


def tolerance_policy() -> dict[str, Any]:
    """The constants and the rule, written into every index so the derivation is explicit."""
    return {
        "safety_factor": TOLERANCE_SAFETY_FACTOR,
        "fp32_epsilon": FP32_EPSILON,
        "rule": (
            "atol = max(safety_factor * max_abs_err_vs_fp64, safety_factor * fp32_epsilon * max_abs_value); "
            "rel_fro_tol = safety_factor * rel_fro_err_vs_fp64; a consumer passes when "
            "max|got - golden| <= atol and ||got - golden|| / ||golden|| <= rel_fro_tol"
        ),
        "measurement": (
            "max_abs_err_vs_fp64 and rel_fro_err_vs_fp64 are the distance between the reference's own "
            "output and the same function evaluated in float64 on the same (bitwise) dequantized weights"
        ),
    }


def write_golden(out_dir: Path, cases: list[GoldenCase]) -> tuple[Path, dict[str, Any]]:
    """Write every case's tensors and the index; returns the index path and contents."""
    root = Path(out_dir) / GOLDEN_DIR
    functions: dict[str, Any] = {}
    for case in cases:
        directory = root / case.function
        entries_in = [write_tensor(directory, n, t, role=ROLE_INPUT) for n, t in case.inputs]
        entries_out = [write_tensor(directory, n, t, role=ROLE_OUTPUT) for n, t in case.outputs]
        functions[case.function] = {
            "directory": case.function,
            "inputs": entries_in,
            "outputs": entries_out,
            "source": case.source,
            "tolerance": case.tolerance,
            "notes": case.notes,
        }
    index = {
        "format_version": GOLDEN_FORMAT_VERSION,
        "byte_order": BYTE_ORDER,
        "layout": "row-major (C order), no header, dtype and shape as listed",
        "tolerance_policy": tolerance_policy(),
        "functions": functions,
    }
    return F.write_json(root / INDEX_NAME, index), index


def summary(index: dict[str, Any]) -> dict[str, Any]:
    """The numbers worth repeating in `export.json`: per function, the worst output."""
    out: dict[str, Any] = {}
    for name, entry in index["functions"].items():
        tol = entry.get("tolerance")
        if not tol:
            out[name] = None
            continue
        worst = max(tol, key=lambda e: e["max_abs_err_vs_fp64"])
        out[name] = {
            "worst_output": worst["name"],
            "max_abs_err_vs_fp64": worst["max_abs_err_vs_fp64"],
            "atol": worst["atol"],
            "rel_fro_tol": max(e["rel_fro_tol"] for e in tol),
        }
    return out


def is_finite_report(report: list[dict[str, Any]]) -> bool:
    return all(math.isfinite(e["max_abs_err_vs_fp64"]) for e in report)
