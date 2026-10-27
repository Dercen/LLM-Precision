"""MLIR through torch-mlir's FX importer: the raw torch dialect, and its lowering to
linalg-on-tensors.

`torch_mlir` is an optional extra (`uv sync --extra mlir`); everything here imports it
lazily so the nn.Module and JSON halves of the export work without it.

Two forms of the same function can be written (`FORMS`):

* `raw`: the torch dialect exactly as torch.export and torch-mlir's decomposition table
  produce it. Dequantization is still visible as `quantized_decomposed.dequantize_*` (or
  explicit ATen) ops, so this is the only form that still says "this weight is 4-bit".
* `linalg`: torch-mlir's own backend pipeline applied to the raw form: linalg, arith,
  tensor and math ops on static tensors. Dequantization becomes plain integer-to-float
  arithmetic (`arith.extui`, `arith.sitofp`, `arith.subf`, `arith.mulf`) inside
  `linalg.generic` bodies; the quantization structure survives only in `layers.json`
  and, with `--weights external`, `weights.json`.

Weights reach the IR in one of two ways, chosen per import:

* embedded (`RawLiteralHooks`): each distinct tensor once, as a dense resource built
  from its raw buffer. The importer's default literal path goes through
  `tensor.tolist()` -- a Python list per weight, ~1.5 s and ~1 GB of transient objects
  per 40M parameters -- which is why the hook hands it the buffer directly.
* lifted (`LiftingHooks`): the same hook answers with a new function argument instead
  of a literal, so the graph is identical but takes its weights as inputs. The
  argument order is the order the graph first uses each tensor (execution order), and
  every argument keeps the tensor's module path, which `export/weights.py` writes next
  to the packed bytes.

Both go through `FxImporter.import_program`, not the older `import_frozen_program`: the
frozen path binds buffers from `state_dict` only, so a non-persistent buffer such as
Llama's rotary `inv_freq` raises "Could not find state mapping for buffer" and no
Llama-family model could be exported before this module switched (2026-10-26). The
only differences in the raw text for OPT are one function attribute
(`torch.assume_strict_symbolic_shapes`) and a tied `lm_head` weight referenced once
instead of embedded twice.
"""

from __future__ import annotations

import functools
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

#: The dequantization ops a raw export may contain, with the trailing space so a count of
#: `dequantize_per_channel ` does not also match `dequantize_per_channel_group `.
DEQUANT_OPS = (
    "torch.quantized_decomposed.dequantize_per_channel ",
    "torch.quantized_decomposed.dequantize_per_channel_group ",
)

#: The forms of a function that can be written; `OUTPUT_TYPES` is what the CLI accepts.
FORMS = ("raw", "linalg")
OUTPUT_TYPES = ("raw", "linalg", "both")

#: File-name suffixes per form. A function `f` lands in `f.mlir` (raw) and
#: `f.linalg.mlir` (lowered); `--bytecode` adds the same names with `BYTECODE_SUFFIX`.
RAW_SUFFIX = ".mlir"
LINALG_SUFFIX = ".linalg.mlir"
BYTECODE_SUFFIX = ".mlirbc"

#: Op-name prefix that proves a lowered module really is linalg-on-tensors.
LINALG_PREFIX = "linalg."
#: Op-name prefix that must not survive the lowering: a `torch.` op means the backend
#: pipeline left a torch-dialect op behind, and the lowered form is not usable.
TORCH_PREFIX = "torch."
#: torch-mlir's private dialect (scatter, scan, attention). It is legal linalg-pipeline
#: output, but neither a stock `mlir-opt` -- the tool the vault compiler drives -- nor
#: this wheel's Python parser (`parse` below; it ships no tm_tensor binding) can read
#: it; only `torch-mlir-opt` can. Every occurrence is counted into `export.json` and
#: noted. A whole-model graph has one `tm_tensor.scan`: transformers derives position
#: ids (OPT) or checks them for packed sequences (Llama) with a `cumsum`;
#: `--explicit-positions` removes it.
TM_TENSOR_PREFIX = "tm_tensor."

#: Op-name prefixes `export.json` counts in the raw form: the quantized ops, anything
#: that fell back to an opaque `torch.operator`, and the matmul-shaped ops.
RAW_OP_PREFIXES = (
    "torch.quantized_decomposed", "torch.operator", "torch.aten.index_select",
    "torch.aten.mm", "torch.aten.linear", "torch.aten.bmm",
)

#: torch-mlir's pipeline from the raw torch dialect to the backend contract. The
#: option braces are mandatory (the pipeline parser rejects a bare name) and are left
#: empty: no backend-legal ops, no extra shape library. Identical to what
#: `torch_mlir.fx.export_and_import(output_type="linalg-on-tensors")` runs.
_TORCH_BACKEND_PIPELINE = (
    "builtin.module(func.func(torch-match-quantized-custom-ops), "
    "torchdynamo-export-to-torch-backend-pipeline{ extra-library=})"
)


def available() -> bool:
    try:
        import torch_mlir  # noqa: F401
    except ImportError:
        return False
    return True


def version() -> str | None:
    try:
        from importlib import metadata

        return metadata.version("torch-mlir")
    except Exception:  # noqa: BLE001 - absent or unversioned both mean "unknown"
        return None


def _require():
    try:
        import torch_mlir  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "torch-mlir is not installed: `uv sync --extra mlir` (adds the torch-mlir dev wheel "
            "and ml_dtypes), or pass --no-mlir to skip the .mlir file"
        ) from exc


def align_legacy_tf32_flags() -> None:
    """Make the legacy `allow_tf32` booleans agree with the per-knob precision settings.

    `device.lock_numerics` pins the per-knob `fp32_precision` API only, and PyTorch's
    legacy `torch.backends.cudnn.allow_tf32` getter then raises because the boolean
    it still holds (True by default) disagrees with the conv/rnn knobs. Nothing in
    the evaluation path reads that getter; `torch.export` does, while capturing
    global state. The setters write what the knobs already say, so TF32 stays off.
    No hasattr(): reading the legacy property is exactly what raises.
    """
    from .. import device as D

    live = D.tf32_is_live()
    for mod in (torch.backends.cudnn, torch.backends.cuda.matmul):
        try:
            mod.allow_tf32 = live
        except AttributeError:  # pragma: no cover - a build without the legacy flag
            pass


# ---- how weights reach the IR: embedded literals or lifted arguments --------------------


@dataclass(frozen=True)
class LiftedArgument:
    """One tensor that `LiftingHooks` turned into a function argument.

    `index` is its position in the function's argument list, after the user inputs;
    `name` is the tensor's module path (its state_dict key, e.g.
    `model.layers.0.mlp.gate_proj.weight_q`) or `constant_<n>` for a tensor the forward
    created and torch.export lifted; `shape` and `dtype` are the tensor's own.
    """

    index: int
    name: str
    shape: tuple[int, ...]
    dtype: str
    #: The tensor itself (a reference, not a copy), so the weight writer can store its
    #: bytes and match it against the model's own parameter names by storage.
    tensor: Any = field(default=None, repr=False, compare=False)


def _storage_key(t: torch.Tensor) -> tuple:
    """Identity of a tensor's bytes: tied weights (lm_head = embed_tokens) share one."""
    return (t.data_ptr(), tuple(t.shape), t.dtype, str(t.device))


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


@functools.cache
def _hook_classes():
    """The two hook classes, built lazily because they subclass torch-mlir's base."""
    import numpy as np
    from torch_mlir.extras.fx_importer import FxImporterHooks, create_mlir_tensor_type
    from torch_mlir.ir import DenseResourceElementsAttr, Location, Operation

    try:
        import ml_dtypes
    except ImportError:
        ml_dtypes = None

    class DedupHooks(FxImporterHooks):
        """Shared bookkeeping: each distinct storage is materialised once, and counted."""

        def __init__(self):
            self._by_key: dict[tuple, Any] = {}
            self.n_literals = 0  # uses resolved through the hook
            self.n_bytes = 0  # bytes of distinct tensors that went through it

        @staticmethod
        def _skip(literal: Any) -> bool:
            # Non-tensors and splats (0 or 1 element) take the importer's own path, which
            # inlines them as `dense<...>` constants; bools were already turned into uint8.
            return not isinstance(literal, torch.Tensor) or literal.numel() <= 1

    class RawLiteralHooks(DedupHooks):
        """Embed each distinct tensor once, as a dense resource built from its raw bytes."""

        def resolve_literal(self, gni, literal, info):
            if self._skip(literal):
                return None
            key = _storage_key(literal)
            attr = self._by_key.get(key)
            if attr is None:
                t = literal.detach().to("cpu").contiguous()
                if t.dtype == torch.bfloat16:
                    if ml_dtypes is None:
                        return None  # let the importer raise its own, clearer error
                    arr = t.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
                else:
                    arr = t.numpy()
                shape = "_".join(str(d) for d in literal.shape)
                name = f"torch_tensor_{shape}_{literal.dtype}"
                tensor_type = create_mlir_tensor_type(literal)
                arr = np.ascontiguousarray(arr)
                attr = DenseResourceElementsAttr.get_from_buffer(arr, name, tensor_type)
                self._by_key[key] = attr
                self.n_bytes += arr.nbytes
            self.n_literals += 1
            vtensor_type = gni._cc.tensor_to_vtensor_type(literal)
            return Operation.create(
                name="torch.vtensor.literal", results=[vtensor_type], attributes={"value": attr}
            ).result

    class LiftingHooks(DedupHooks):
        """Turn each distinct tensor into a new argument of the function being built.

        `import_program` creates the function with the user inputs only and binds every
        parameter, buffer and lifted constant lazily, at its first use in the graph, so
        appending a block argument here puts the weights after the user inputs in
        execution order. The function's type attribute still lists the user inputs only
        until `_adopt_appended_arguments` rewrites it after the import.
        """

        def __init__(self, names_by_key: dict[tuple, str]):
            super().__init__()
            self.arguments: list[LiftedArgument] = []
            self._names_by_key = names_by_key  # constants carry no InputInfo; see register

        def resolve_literal(self, gni, literal, info):
            if self._skip(literal):
                return None
            key = _storage_key(literal)
            value = self._by_key.get(key)
            if value is None:
                if info is not None:
                    name = str(info.input_spec.target)
                else:
                    name = self._names_by_key.get(key, f"constant_{len(self.arguments)}")
                block = gni._b
                value = block.add_argument(
                    gni._cc.tensor_to_vtensor_type(literal), Location.unknown(gni._c)
                )
                self._by_key[key] = value
                self.arguments.append(LiftedArgument(
                    index=len(block.arguments) - 1, name=name,
                    shape=tuple(literal.shape), dtype=_dtype_name(literal.dtype), tensor=literal,
                ))
                self.n_bytes += literal.numel() * literal.element_size()
            self.n_literals += 1
            return value

    return RawLiteralHooks, LiftingHooks


def _raw_literal_hooks():
    return _hook_classes()[0]()


def _lifting_hooks(prog: torch.export.ExportedProgram):
    """LiftingHooks that can name the program's lifted tensor constants by storage."""
    names = {
        _storage_key(t): name for name, t in prog.constants.items() if isinstance(t, torch.Tensor)
    }
    return _hook_classes()[1](names)


def _adopt_appended_arguments(func_op, context) -> None:
    """Rewrite `function_type` so it lists the block arguments the hooks appended."""
    from torch_mlir import ir

    entry = func_op.regions[0].blocks[0]
    with context:
        ftype = ir.FunctionType(ir.TypeAttr(func_op.attributes["function_type"]).value)
        new_type = ir.FunctionType.get([a.type for a in entry.arguments], list(ftype.results))
        func_op.attributes["function_type"] = ir.TypeAttr.get(new_type)


# ---- export + import --------------------------------------------------------------------


@dataclass
class ImportedFunction:
    """One `func.func` in its own MLIR module, plus what the import recorded about it."""

    module: Any  # torch_mlir.ir.Module
    func_name: str
    n_user_inputs: int
    #: The weights that became arguments, in argument order; empty when embedded.
    lifted: list[LiftedArgument] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def export_program(
    fn: nn.Module, example_args: tuple, *, dynamic_shapes: Any = None
) -> torch.export.ExportedProgram:
    """`torch.export` plus torch-mlir's decomposition table: the two steps
    `torch_mlir.fx.export_and_import` performs before importing, done here so one
    ExportedProgram can be both imported and run (`prog.module()`) for checks."""
    _require()
    from torch_mlir.extras.fx_decomp_util import get_decomposition_table

    align_legacy_tf32_flags()
    with torch.no_grad():
        prog = torch.export.export(fn, tuple(example_args), dynamic_shapes=dynamic_shapes, strict=False)
        return prog.run_decompositions(get_decomposition_table())


def import_program(
    prog: torch.export.ExportedProgram,
    *,
    func_name: str = "main",
    lift_weights: bool = False,
    import_symbolic_shape_expressions: bool = False,
) -> ImportedFunction:
    """A decomposed ExportedProgram as one `func.func` in the raw torch dialect."""
    _require()
    from torch_mlir import ir
    from torch_mlir.dialects import torch as torch_d
    from torch_mlir.extras.fx_importer import FxImporter

    context = ir.Context()
    torch_d.register_dialect(context)
    hooks = _lifting_hooks(prog) if lift_weights else _raw_literal_hooks()
    importer = FxImporter(context=context, hooks=hooks)
    func_op = importer.import_program(
        prog, func_name=func_name, import_symbolic_shape_expressions=import_symbolic_shape_expressions
    )
    n_user_inputs = len(prog.graph_signature.user_inputs)
    lifted: list[LiftedArgument] = []
    if lift_weights:
        _adopt_appended_arguments(func_op, context)
        lifted = list(hooks.arguments)
    stats = {
        "literals": hooks.n_literals,
        "distinct_literal_bytes": hooks.n_bytes,
        "weights": "lifted" if lift_weights else "embedded",
        "torch_mlir_version": version(),
    }
    return ImportedFunction(importer.module, func_name, n_user_inputs, lifted, stats)


def import_function(
    fn: nn.Module,
    example_args: tuple,
    *,
    func_name: str = "main",
    lift_weights: bool = False,
    dynamic_shapes: Any = None,
    import_symbolic_shape_expressions: bool = False,
) -> ImportedFunction:
    """`fn(*example_args)` as a raw torch-dialect function: export, decompose, import."""
    prog = export_program(fn, example_args, dynamic_shapes=dynamic_shapes)
    return import_program(
        prog, func_name=func_name, lift_weights=lift_weights,
        import_symbolic_shape_expressions=import_symbolic_shape_expressions,
    )


#: The smallest sequence length a `--dynamic-seqlen` export admits: torch.export
#: rejects a symbolic dimension whose range includes 0 and 1 (it specialises them).
DYNAMIC_SEQLEN_MIN = 2


def seqlen_dynamic_shapes(example_input_ids: torch.Tensor, max_seqlen: int | None) -> tuple:
    """`dynamic_shapes` for `model(input_ids)` with a symbolic sequence length, bounded
    by the model's position table (or the example length when it has none)."""
    from torch.export import Dim

    upper = max_seqlen or int(example_input_ids.shape[1])
    return ({1: Dim("seqlen", min=DYNAMIC_SEQLEN_MIN, max=max(upper, DYNAMIC_SEQLEN_MIN))},)


def export_torch_dialect(
    model: nn.Module,
    example_input_ids: torch.Tensor,
    *,
    dynamic_seqlen: bool = False,
    max_seqlen: int | None = None,
    func_name: str = "main",
    lift_weights: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """`model(input_ids)` in the raw torch dialect: the whole-model entry point.

    Returns the MLIR module and a small stats dict (the pre-2026-10-26 interface, kept
    for callers that only need `main(%input_ids)`); `import_function` is the general
    form that slices and stages use.
    """
    dynamic_shapes = seqlen_dynamic_shapes(example_input_ids, max_seqlen) if dynamic_seqlen else None
    imported = import_function(
        model, (example_input_ids,), func_name=func_name, lift_weights=lift_weights,
        dynamic_shapes=dynamic_shapes, import_symbolic_shape_expressions=dynamic_seqlen,
    )
    stats = {**imported.stats, "dynamic_seqlen": dynamic_seqlen}
    if lift_weights:
        stats["lifted_arguments"] = [
            {k: v for k, v in asdict(a).items() if k != "tensor"} for a in imported.lifted
        ]
    return imported.module, stats


# ---- lowering and checks ----------------------------------------------------------------


def lower_to_linalg(module: Any) -> None:
    """In place: raw torch dialect -> torch backend contract -> linalg-on-tensors.

    The two public pipeline runners `torch_mlir.fx.export_and_import` itself calls for
    `output_type="linalg-on-tensors"`. In place because the module may hold hundreds of
    MB of dense resources; write the raw form first if both are wanted.
    """
    _require()
    from torch_mlir.compiler_utils import (
        OutputType,
        lower_mlir_module,
        run_pipeline_with_repro_report,
    )

    run_pipeline_with_repro_report(module, _TORCH_BACKEND_PIPELINE, "Lowering TorchFX IR -> Torch Backend IR")
    lower_mlir_module(False, OutputType.LINALG_ON_TENSORS, module)


def count_ops(module: Any, prefixes: tuple[str, ...] | None = None) -> dict[str, int]:
    """Occurrences of every op name under the module, optionally only those with a prefix."""
    counts: dict[str, int] = {}

    def walk(op):
        counts[op.name] = counts.get(op.name, 0) + 1
        for region in op.regions:
            for block in region.blocks:
                for inner in block.operations:
                    walk(inner.operation)

    walk(module.operation)
    if prefixes is not None:
        counts = {k: v for k, v in counts.items() if k.startswith(prefixes)}
    return dict(sorted(counts.items()))


@dataclass
class LoweredReport:
    """What `check_lowered` found: the linalg op counts and any tm_tensor ops."""

    linalg_ops: dict[str, int]
    tm_tensor_ops: dict[str, int]


def check_lowered(module: Any) -> LoweredReport:
    """Prove a lowered module is what a consumer expects, or raise.

    Three conditions: the MLIR verifier accepts it, at least one `linalg.*` op is
    present, and no `torch.*` op survived. `tm_tensor.*` ops do not fail the check (they
    are legitimate pipeline output) but are reported so the caller can note them.
    """
    if not module.operation.verify():
        raise ValueError("the lowered module does not verify")
    counts = count_ops(module)
    linalg = {k: v for k, v in counts.items() if k.startswith(LINALG_PREFIX)}
    if not linalg:
        raise ValueError("the lowered module contains no linalg ops")
    leftovers = {k: v for k, v in counts.items() if k.startswith(TORCH_PREFIX)}
    if leftovers:
        raise ValueError(f"torch-dialect ops survived the lowering: {leftovers}")
    tm_tensor = {k: v for k, v in counts.items() if k.startswith(TM_TENSOR_PREFIX)}
    return LoweredReport(linalg_ops=linalg, tm_tensor_ops=tm_tensor)


# ---- files ------------------------------------------------------------------------------


def write_module(module: Any, path: Path, *, bytecode: bool = False) -> Path:
    """Write text MLIR (`.mlir`) or bytecode (`.mlirbc`), streamed to the file, atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        if bytecode:
            module.operation.write_bytecode(fh)
        else:
            module.operation.print(file=fh, binary=True)
    tmp.replace(path)
    return path


def write_forms(
    imported: ImportedFunction,
    out_dir: Path,
    stem: str,
    *,
    output_type: str = "raw",
    bytecode: bool = False,
) -> tuple[dict[str, str | None], dict[str, Any]]:
    """Write the requested forms of one function and describe them for `export.json`.

    Returns `(files, stats)`: `files` maps `mlir`, `mlir_bytecode`, `mlir_linalg` and
    `mlir_linalg_bytecode` to file names (None when not written); `stats` carries the
    import stats plus `forms`, the raw op counts (`ops`), the lowered op counts
    (`linalg_ops`, `tm_tensor_ops`) and byte sizes. Lowering is in place, so the raw
    form is written and counted first.
    """
    if output_type not in OUTPUT_TYPES:
        raise ValueError(f"output_type must be one of {OUTPUT_TYPES}, got {output_type!r}")
    out_dir = Path(out_dir)
    module = imported.module
    files: dict[str, str | None] = {
        "mlir": None, "mlir_bytecode": None, "mlir_linalg": None, "mlir_linalg_bytecode": None,
    }
    stats: dict[str, Any] = {**imported.stats, "function": imported.func_name, "forms": []}
    stats["ops"] = count_ops(module, RAW_OP_PREFIXES)
    if output_type in ("raw", "both"):
        path = write_module(module, out_dir / f"{stem}{RAW_SUFFIX}")
        files["mlir"] = path.name
        stats["mlir_bytes"] = path.stat().st_size
        if bytecode:
            files["mlir_bytecode"] = write_module(module, out_dir / f"{stem}{BYTECODE_SUFFIX}", bytecode=True).name
        stats["forms"].append("raw")
    if output_type in ("linalg", "both"):
        lower_to_linalg(module)
        report = check_lowered(module)
        stats["linalg_ops"] = report.linalg_ops
        stats["tm_tensor_ops"] = report.tm_tensor_ops
        path = write_module(module, out_dir / f"{stem}{LINALG_SUFFIX}")
        files["mlir_linalg"] = path.name
        stats["linalg_bytes"] = path.stat().st_size
        if bytecode:
            files["mlir_linalg_bytecode"] = write_module(
                module, out_dir / f"{stem}{LINALG_SUFFIX.replace('.mlir', BYTECODE_SUFFIX)}", bytecode=True
            ).name
        stats["forms"].append("linalg")
    return files, stats


def count_dequant_ops(text: str) -> dict[str, int]:
    counts = {op.strip(): text.count(op) for op in DEQUANT_OPS}
    counts["torch.operator"] = text.count("torch.operator")
    return counts


def parse(path: Path) -> Any:
    """Re-parse a written .mlir/.mlirbc into a verified module (raises if it does not parse).

    Every dialect the wheel ships is loaded, so lowered files (linalg, arith, tensor,
    math, cf) parse as well as raw torch-dialect ones. A lowered file that kept
    `tm_tensor` ops does not: the wheel has no Python binding for that dialect.
    """
    _require()
    from torch_mlir import ir
    from torch_mlir.dialects import torch as torch_d

    context = ir.Context()
    torch_d.register_dialect(context)
    context.load_all_available_dialects()
    data = Path(path).read_bytes()
    module = ir.Module.parse(data, context)
    if not module.operation.verify():
        raise ValueError(f"{path} does not verify")
    return module
