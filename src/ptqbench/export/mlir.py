"""torch-dialect MLIR through torch-mlir's FX importer.

`torch_mlir` is an optional extra (`uv sync --extra mlir`); everything here imports it
lazily so the nn.Module and JSON halves of the export work without it.

The importer's default literal path goes through `tensor.tolist()` -- a Python list per
weight, ~1.5 s and ~1 GB of transient objects per 40M parameters -- so `RawLiteralHooks`
hands it the tensor's buffer directly. Literals are real tensors during
`import_frozen_program`, which is what makes that safe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

DEQUANT_OPS = (
    "torch.quantized_decomposed.dequantize_per_channel ",
    "torch.quantized_decomposed.dequantize_per_channel_group ",
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


def _raw_literal_hooks():
    import numpy as np
    from torch_mlir.extras.fx_importer import FxImporterHooks, create_mlir_tensor_type
    from torch_mlir.ir import DenseResourceElementsAttr, Operation

    try:
        import ml_dtypes
    except ImportError:
        ml_dtypes = None

    class RawLiteralHooks(FxImporterHooks):
        """Embed each distinct tensor once, as a dense resource built from its raw bytes."""

        def __init__(self):
            self._attrs: dict[tuple, Any] = {}
            self.n_literals = 0
            self.n_bytes = 0

        def resolve_literal(self, gni, literal, info):
            if not isinstance(literal, torch.Tensor) or literal.dtype == torch.bool:
                return None  # bools are converted by the importer itself
            key = (literal.data_ptr(), tuple(literal.shape), literal.dtype, str(literal.device))
            attr = self._attrs.get(key)
            if attr is None:
                t = literal.detach().to("cpu").contiguous()
                if t.dtype == torch.bfloat16:
                    if ml_dtypes is None:
                        return None  # let the importer raise its own, clearer error
                    arr = t.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
                else:
                    arr = t.numpy()
                if arr.size <= 1:
                    return None  # splats: the importer's DenseElementsAttr path
                shape = "_".join(str(d) for d in literal.shape)
                name = f"torch_tensor_{shape}_{literal.dtype}"
                tensor_type = create_mlir_tensor_type(literal)
                arr = np.ascontiguousarray(arr)
                attr = DenseResourceElementsAttr.get_from_buffer(arr, name, tensor_type)
                self._attrs[key] = attr
                self.n_bytes += arr.nbytes
            self.n_literals += 1
            vtensor_type = gni._cc.tensor_to_vtensor_type(literal)
            return Operation.create(
                name="torch.vtensor.literal", results=[vtensor_type], attributes={"value": attr}
            ).result

    return RawLiteralHooks()


def export_torch_dialect(
    model: nn.Module,
    example_input_ids: torch.Tensor,
    *,
    dynamic_seqlen: bool = False,
    max_seqlen: int | None = None,
    func_name: str = "main",
) -> tuple[Any, dict[str, Any]]:
    """`torch_mlir.fx.export_and_import` of `model(input_ids)` in the raw torch dialect.

    Raw (not the backend contract) keeps the dequantization ops as the module wrote
    them; `torch-mlir-opt --torchdynamo-export-to-torch-backend-pipeline` is the next
    step for a consumer. Returns the MLIR module and a small stats dict.
    """
    _require()
    from torch_mlir import fx, ir
    from torch_mlir.dialects import torch as torch_d
    from torch_mlir.extras.fx_importer import FxImporter

    align_legacy_tf32_flags()
    hooks = _raw_literal_hooks()
    context = ir.Context()
    torch_d.register_dialect(context)
    importer = FxImporter(context=context, hooks=hooks)

    dynamic_shapes = None
    if dynamic_seqlen:
        from torch.export import Dim

        upper = max_seqlen or int(example_input_ids.shape[1])
        dynamic_shapes = ({1: Dim("seqlen", min=2, max=max(upper, 2))},)

    with torch.no_grad():
        module = fx.export_and_import(
            model,
            example_input_ids,
            output_type="raw",
            fx_importer=importer,
            dynamic_shapes=dynamic_shapes,
            import_symbolic_shape_expressions=dynamic_seqlen,
            func_name=func_name,
        )
    stats = {
        "literals": hooks.n_literals,
        "distinct_literal_bytes": hooks.n_bytes,
        "dynamic_seqlen": dynamic_seqlen,
        "torch_mlir_version": version(),
    }
    return module, stats


def write_module(module: Any, path: Path, *, bytecode: bool = False) -> Path:
    """Write text MLIR (`.mlir`) or bytecode (`.mlirbc`), streamed to the file."""
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


def count_dequant_ops(text: str) -> dict[str, int]:
    counts = {op.strip(): text.count(op) for op in DEQUANT_OPS}
    counts["torch.operator"] = text.count("torch.operator")
    return counts


def parse(path: Path) -> Any:
    """Re-parse a written .mlir/.mlirbc into a verified module (raises if it does not parse)."""
    _require()
    from torch_mlir import ir
    from torch_mlir.dialects import torch as torch_d

    context = ir.Context()
    torch_d.register_dialect(context)
    data = Path(path).read_bytes()
    module = ir.Module.parse(data, context)
    if not module.operation.verify():
        raise ValueError(f"{path} does not verify")
    return module
