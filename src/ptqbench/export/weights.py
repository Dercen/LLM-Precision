"""Weights outside the IR: a graph-only `.mlir` whose weights are function arguments,
plus `weights.bin` (every tensor, densely packed, aligned) and `weights.json` (what
each one is and where it sits). `ptq export-mlir --weights external`.

How the IR stops embedding weights
----------------------------------
torch-mlir's FX importer asks a hook (`FxImporterHooks.resolve_literal`) how to
materialise every parameter, buffer and lifted constant the graph uses. The embedded
export answers with a dense resource; `mlir.LiftingHooks` answers with a new argument
appended to the function's entry block instead. Nothing else changes: the graph is the
same ops in the same order, tied tensors (lm_head = embed_tokens) are deduplicated by
storage so they become one argument, and every argument keeps the tensor's module
path. The alternative -- wrapping the model in `torch.func.functional_call` so the
weights become user inputs -- works too, but adds a wrapper layer, loses the
name-to-placeholder mapping unless rebuilt by hand, and gives a tied weight two
arguments. Arguments appear in the order the graph first uses each tensor (its
execution order); `weights.json` records that order per function.

Packing (`pack`, `unpack`)
--------------------------
Codes narrower than a byte are packed row by row as a little-endian bit stream:

* Row `r` of a `(rows, cols)` code matrix becomes `ceil(cols * bits / 8)` bytes
  (`row_stride_bytes`); rows are stored one after another with no gap between them.
* Within a row, code `j` occupies stream bits `[j*bits, (j+1)*bits)`, least significant
  bit first; stream bit `k` lives in bit `k % 8` of byte `k // 8`.
* So at 4 bits, code 0 is the low nibble of byte 0 and code 1 the high nibble; at 3
  bits, codes straddle byte boundaries (code 2 is bits 6..8: the top two bits of byte 0
  and the lowest bit of byte 1); at 2 bits, four codes per byte. Odd widths need no
  special case, only the formula. The trailing bits of the last byte of a row are zero.
* A C consumer reads code `j` of a row as
  `(row_bits >> (j * bits)) & ((1 << bits) - 1)` over the row as a little-endian integer,
  or for 4 bits simply `byte[j/2] & 0xF` / `byte[j/2] >> 4`.
* 8-bit codes are one byte per code, unpacked (`packed: false`); their layout is the
  same formula with `bits = 8`.
* Group boundaries stay byte-aligned for every group size that is a multiple of 8
  (64 x bits / 8 = 8 x bits bytes), so a group can be addressed without bit arithmetic.

Alignment
---------
Every tensor's byte offset in `weights.bin` is a multiple of `WEIGHT_ALIGNMENT_BYTES`,
padded with zero bytes. Scales, zero-points, biases and unquantized float tensors are
stored in exactly the dtype the function argument has, so a consumer can hand the bytes
to the function as they are; only `codes` need `unpack` first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from . import files as F
from .mlir import LiftedArgument, _storage_key
from .quantized import QuantizedLinear

#: File names next to the graph-only `.mlir`.
WEIGHTS_BIN = "weights.bin"
WEIGHTS_JSON = "weights.json"
#: Bumped when the index schema or the packing rule changes.
WEIGHTS_FORMAT_VERSION = 1

#: Alignment of every tensor's byte offset in `weights.bin`. 64 bytes is one cache
#: line on the cores in question and a multiple of every RISC-V vector element width,
#: so a tensor's first vector load never straddles a line. Named here, written into
#: `weights.json` as `alignment`, and the only place to change it.
WEIGHT_ALIGNMENT_BYTES = 64

#: Modes of `--weights`.
WEIGHT_MODES = ("embedded", "external")

#: The bit order of the packed code stream, written into `weights.json` for readers.
PACK_BIT_ORDER = "lsb-first"
BITS_PER_BYTE = 8
#: Code widths the pipeline produces; below 8 they are packed, at 8 stored as bytes.
MIN_BITS, MAX_BITS = 2, 8

#: Roles a tensor in `weights.bin` plays. The five quantized leaves come from
#: `QuantizedLinear`'s buffers; everything else is an unquantized `float` tensor
#: (embeddings, norms, lm_head, rotary tables, captured hidden-state constants) or an
#: `int` one (captured position ids).
ROLE_CODES = "codes"
ROLE_SCALE = "scale"
ROLE_ZERO = "zero"
ROLE_G_IDX = "g_idx"
ROLE_BIAS = "bias"
ROLE_FLOAT = "float"
ROLE_INT = "int"
#: QuantizedLinear attribute -> role.
QUANTIZED_LEAVES = {"weight_q": ROLE_CODES, "scale": ROLE_SCALE, "zero": ROLE_ZERO, "g_idx": ROLE_G_IDX, "bias": ROLE_BIAS}


# ---- packing ----------------------------------------------------------------------------


def _check_bits(bits: int) -> None:
    if not MIN_BITS <= bits <= MAX_BITS:
        raise ValueError(f"bits must be in [{MIN_BITS}, {MAX_BITS}], got {bits}")


def row_stride_bytes(cols: int, bits: int) -> int:
    """Bytes one row of `cols` codes of `bits` bits occupies: `ceil(cols * bits / 8)`."""
    return math.ceil(cols * bits / BITS_PER_BYTE)


def pack(codes: np.ndarray | torch.Tensor, bits: int) -> np.ndarray:
    """`(rows, cols)` codes in `[0, 2**bits)` -> `(rows, row_stride_bytes)` uint8, LSB-first."""
    _check_bits(bits)
    codes = np.ascontiguousarray(codes.cpu().numpy() if torch.is_tensor(codes) else codes)
    if codes.ndim != 2:
        raise ValueError(f"codes must be 2-D (rows, cols), got shape {codes.shape}")
    if codes.size and (codes.min() < 0 or codes.max() >= 2**bits):
        raise ValueError(f"codes must lie in [0, {2**bits - 1}] for {bits} bits")
    codes = codes.astype(np.uint8)
    if bits == BITS_PER_BYTE:
        return codes.copy()
    rows, cols = codes.shape
    # (rows, cols, 8) bit planes, bit 0 first; keep the `bits` meaningful ones per code,
    # lay them end to end along the row, and pack 8 at a time with bit 0 as the LSB.
    planes = np.unpackbits(codes[:, :, None], axis=2, bitorder="little")[:, :, :bits]
    stream = planes.reshape(rows, cols * bits)
    return np.packbits(stream, axis=1, bitorder="little")


def unpack(packed: np.ndarray | bytes, bits: int, cols: int) -> np.ndarray:
    """The inverse of `pack`: `(rows, row_stride_bytes)` bytes -> `(rows, cols)` uint8 codes."""
    _check_bits(bits)
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    if packed.ndim != 2 or packed.shape[1] != row_stride_bytes(cols, bits):
        raise ValueError(f"packed rows must be (rows, {row_stride_bytes(cols, bits)}) for {cols} x {bits}-bit codes, got {packed.shape}")
    if bits == BITS_PER_BYTE:
        return packed[:, :cols].copy()
    rows = packed.shape[0]
    stream = np.unpackbits(packed, axis=1, bitorder="little")[:, : cols * bits]
    planes = np.zeros((rows, cols, BITS_PER_BYTE), dtype=np.uint8)
    planes[:, :, :bits] = stream.reshape(rows, cols, bits)
    return np.packbits(planes, axis=2, bitorder="little")[:, :, 0]


def packing_rule() -> dict[str, Any]:
    """The packing contract, written into `weights.json` so a reader never has to guess."""
    return {
        "bit_order": PACK_BIT_ORDER,
        "rule": (
            "rows of the (rows, cols) code matrix are stored one after another, each as "
            "ceil(cols * bits / 8) bytes; within a row, code j occupies stream bits "
            "[j*bits, (j+1)*bits) least-significant bit first, and stream bit k is bit k % 8 of "
            "byte k // 8; trailing bits of a row's last byte are zero; 8-bit codes are one byte "
            "each and marked packed: false"
        ),
        "example_4bit": "code 0 is the low nibble of byte 0, code 1 the high nibble",
        "example_3bit": "codes [1, 2, 3] -> bytes [0xD1, 0x00]",
        "c_read": "code_j = (row_as_little_endian_integer >> (j * bits)) & ((1 << bits) - 1)",
    }


# ---- the file pair ----------------------------------------------------------------------


def _pad_to(n: int, alignment: int) -> int:
    return (n + alignment - 1) // alignment * alignment


@dataclass
class WeightEntry:
    """One tensor in `weights.bin`; `as_dict` is the `weights.json` entry."""

    name: str
    role: str
    shape: list[int]
    dtype: str  # the function argument's dtype (the tensor's own)
    storage_dtype: str  # numpy dtype of the bytes: uint8 for packed codes, uint16 for bf16
    bits: int | None
    group_size: int | None
    scheme: str | None
    offset: int
    length: int
    packed: bool
    row_stride_bytes: int | None
    sha256: str
    module: str | None = None  # the QuantizedLinear this leaf belongs to
    aliases: list[str] = field(default_factory=list)  # other model names of the same storage
    quant: dict[str, Any] | None = None  # sym, zero_point_domain, quant_min/max of the module

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class WeightsWriter:
    """Collects the lifted arguments of one or more functions and writes the file pair.

    Tensors are named after the *model* (`model.layers.3.mlp.gate_proj.weight_q`), not
    after whichever wrapper was exported, by matching storage against the model's own
    parameters and buffers; a tensor the model does not own (a captured rotary table in
    a slice, say) keeps the name the importer saw. Each function records its argument
    list, so three stage functions can share one `weights.bin`.
    """

    def __init__(self, model: nn.Module, modules: dict[str, QuantizedLinear], *, name_prefix: str = ""):
        self._modules = modules
        self._names_by_key: dict[tuple, list[str]] = {}
        for name, t in list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False)):
            self._names_by_key.setdefault(_storage_key(t), []).append(name_prefix + name)
        self._entries: dict[str, WeightEntry] = {}
        self._blobs: dict[str, bytes] = {}
        self._functions: dict[str, dict[str, Any]] = {}
        self._offset = 0

    # -- naming and roles

    def canonical_name(self, arg: LiftedArgument) -> tuple[str, list[str]]:
        names = self._names_by_key.get(_storage_key(arg.tensor)) if arg.tensor is not None else None
        if not names:
            return arg.name, []
        return names[0], names[1:]

    def _describe(self, name: str, tensor: torch.Tensor) -> tuple[str, str | None, dict[str, Any] | None]:
        module_name, _, leaf = name.rpartition(".")
        mod = self._modules.get(module_name)
        if mod is not None and leaf in QUANTIZED_LEAVES:
            d = mod.describe()
            quant = {k: d[k] for k in ("sym", "zero_point_domain", "quant_min", "quant_max", "n_groups", "algo")}
            return QUANTIZED_LEAVES[leaf], module_name, quant
        return (ROLE_FLOAT if tensor.is_floating_point() else ROLE_INT), None, None

    # -- adding

    def add_function(self, func_name: str, n_user_inputs: int, lifted: list[LiftedArgument]) -> None:
        arguments = []
        for arg in lifted:
            name, aliases = self.canonical_name(arg)
            if name not in self._entries:
                self._add_tensor(name, arg.tensor, aliases)
            arguments.append({"index": arg.index, "name": name})
        self._functions[func_name] = {"n_user_inputs": n_user_inputs, "arguments": arguments}

    def _add_tensor(self, name: str, tensor: torch.Tensor, aliases: list[str]) -> None:
        role, module_name, quant = self._describe(name, tensor)
        mod = self._modules.get(module_name) if module_name else None
        bits = group_size = scheme = stride = None
        packed = False
        if role == ROLE_CODES:
            bits, group_size, scheme = mod.bits, mod.group_size, mod.scheme
            if bits < BITS_PER_BYTE:
                arr = pack(tensor, bits)
                packed = True
            else:
                arr = tensor.detach().cpu().contiguous().numpy()
            stride = int(arr.shape[1])
            storage_dtype = str(arr.dtype)
        else:
            if mod is not None:
                bits, group_size, scheme = mod.bits, mod.group_size, mod.scheme
            arr, _ = F.numpy_little_endian(tensor)
            storage_dtype = str(arr.dtype)
        raw = arr.tobytes()
        offset = _pad_to(self._offset, WEIGHT_ALIGNMENT_BYTES)
        self._entries[name] = WeightEntry(
            name=name, role=role, shape=list(tensor.shape), dtype=F.dtype_name(tensor.dtype),
            storage_dtype=storage_dtype, bits=bits, group_size=group_size, scheme=scheme,
            offset=offset, length=len(raw), packed=packed, row_stride_bytes=stride,
            sha256=F.sha256_of(raw), module=module_name, aliases=aliases, quant=quant,
        )
        self._blobs[name] = raw
        self._offset = offset + len(raw)

    # -- writing

    def write(self, out_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
        """`weights.bin` and `weights.json`; returns both paths and the index."""
        out_dir = Path(out_dir)
        blob = bytearray(self._offset)
        for name, entry in self._entries.items():
            blob[entry.offset : entry.offset + entry.length] = self._blobs[name]
        bin_path = F.write_bytes(out_dir / WEIGHTS_BIN, bytes(blob))
        index = {
            "format_version": WEIGHTS_FORMAT_VERSION,
            "file": WEIGHTS_BIN,
            "byte_order": F.BYTE_ORDER,
            "alignment": WEIGHT_ALIGNMENT_BYTES,
            "packing": packing_rule(),
            "total_bytes": self._offset,
            "n_tensors": len(self._entries),
            "roles": {role: sum(1 for e in self._entries.values() if e.role == role) for role in
                      (ROLE_CODES, ROLE_SCALE, ROLE_ZERO, ROLE_G_IDX, ROLE_BIAS, ROLE_FLOAT, ROLE_INT)},
            "tensors": [e.as_dict() for e in self._entries.values()],
            "functions": self._functions,
        }
        json_path = F.write_json(out_dir / WEIGHTS_JSON, index)
        return bin_path, json_path, index

    @property
    def n_tensors(self) -> int:
        return len(self._entries)


# ---- reading back -----------------------------------------------------------------------


def read_weights(out_dir: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Every tensor of `weights.bin`, unpacked, in the dtype and shape the function wants."""
    out_dir = Path(out_dir)
    index = F.read_json(out_dir / WEIGHTS_JSON)
    blob = (out_dir / index["file"]).read_bytes()
    if len(blob) != index["total_bytes"]:
        raise ValueError(f"{index['file']} is {len(blob)} bytes, the index says {index['total_bytes']}")
    tensors: dict[str, torch.Tensor] = {}
    for entry in index["tensors"]:
        raw = blob[entry["offset"] : entry["offset"] + entry["length"]]
        if F.sha256_of(raw) != entry["sha256"]:
            raise ValueError(f"{entry['name']}: bytes do not match the recorded sha256")
        if entry["role"] == ROLE_CODES and entry["packed"]:
            rows, cols = entry["shape"]
            packed = np.frombuffer(raw, dtype=np.uint8).reshape(rows, entry["row_stride_bytes"])
            tensors[entry["name"]] = torch.from_numpy(unpack(packed, entry["bits"], cols))
        else:
            tensors[entry["name"]] = F.tensor_from_bytes(raw, entry["storage_dtype"], entry["dtype"], entry["shape"])
    return tensors, index


def load_into(model: nn.Module, tensors: dict[str, torch.Tensor], index: dict[str, Any]) -> list[str]:
    """Copy every tensor of `read_weights` into `model` by name (and alias). Returns
    the names that were loaded; raises if a name is not a parameter or buffer."""
    owned = dict(model.named_parameters(remove_duplicate=False))
    owned.update(dict(model.named_buffers(remove_duplicate=False)))
    aliases = {e["name"]: e.get("aliases", []) for e in index["tensors"]}
    loaded = []
    with torch.no_grad():
        for name, value in tensors.items():
            targets = [name, *aliases.get(name, [])]
            for target in targets:
                if target not in owned:
                    raise KeyError(f"{target}: the model has no parameter or buffer of that name")
                owned[target].copy_(value.to(owned[target].dtype))
                loaded.append(target)
    return loaded
