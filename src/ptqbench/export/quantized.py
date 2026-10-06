"""Weight-only quantized Linear layers whose dequantization stays explicit. DESIGN.md 6.

The pipeline fake-quantizes: every target `nn.Linear` ends up holding the *dequantized*
weight `scale * (q - zero)` in the model dtype, and the integer codes `q` and the grid
are gone. `QuantizedLinear` is the same layer with the two halves kept apart -- integer
codes plus scale/zero buffers, and the dequantization performed in `forward` -- built
from the `QuantRecord` a quantizer files when run with `record=`. Its dequantized
weight is bitwise the one the evaluation ran with (asserted in tests/test_export_mlir.py),
so a perplexity row and an export share one set of numbers.

How the pipeline's representation maps onto torch-mlir (measured on torch-mlir
20260930.892, torch 2.14.0; the probe is in the module history):

* Per-row grids (`group_size=-1`) are per-output-channel affine quantization with an
  integer zero-point: `quantized_decomposed.dequantize_per_channel(axis=0)`, which
  torch-mlir imports as a first-class torch-dialect op and lowers to linalg.
* Grouped grids (`group_size=128/64`) have a (rows, n_groups) scale, which the ATen
  quantized-tensor types (`_make_per_channel_quantized_tensor`) cannot hold -- they
  take one axis and a 1-D scale. PyTorch's `quantized_decomposed.dequantize_per_channel_group`
  does hold it, and torch-mlir imports and lowers that op too.
* 2..7-bit codes have no PyTorch dtype; they live in a uint8 container with
  `quant_min=0, quant_max=2**bits-1` carried on the op, which is how PT2E/ExecuTorch
  express sub-byte weights. The container is the only thing torch-mlir types see.
* GPTQ with `act_order` and a group size assigns scattered input columns to each group.
  No `quantized_decomposed` op takes a per-column group index, so those layers (and
  hqq, whose zero-point is a float) are written as explicit ATen ops instead:
  `index_select` of the scale/zero per column, `sub`, `mul`. torch-mlir lowers those
  as plain elementwise ops; they are correct but not recognised as quantization.
* Activations are never quantized by the pipeline, so every export is dequantize-then-
  float-matmul. torch-mlir's `FuseQuantizedOps` (integer matmul) needs both operands
  quantized and does not apply.
* bf16 models need the `ml_dtypes` package for torch-mlir to embed their literals.
"""

from __future__ import annotations

import torch
import torch.ao.quantization.fx._decomposed  # registers torch.ops.quantized_decomposed
from torch import nn
from torch.nn import functional as F

from ..models import families
from ..quantizers.fakequant import QuantRecord

REPRESENTATIONS = ("decomposed", "aten")


class QuantizedLinear(nn.Module):
    """`F.linear(x, dequantize(codes), bias)` with the dequantization left in the graph.

    `representation="decomposed"` uses `torch.ops.quantized_decomposed.dequantize_per_channel`
    (per row) or `dequantize_per_channel_group` (contiguous groups) when the record fits
    them, and falls back to explicit ATen ops otherwise (`scheme` says which was used).
    `representation="aten"` always writes the explicit ops.
    """

    def __init__(
        self,
        record: QuantRecord,
        *,
        bias: torch.Tensor | None,
        weight_dtype: torch.dtype,
        representation: str = "decomposed",
    ):
        super().__init__()
        if representation not in REPRESENTATIONS:
            raise ValueError(f"representation must be one of {REPRESENTATIONS}, got {representation!r}")
        self.out_features, self.in_features = record.rows, record.cols
        self.bits = record.bits
        self.group_size = record.group_size
        self.sym = record.sym
        self.algo = record.algo
        self.quant_min, self.quant_max = 0, record.maxq
        self.weight_dtype = weight_dtype
        self.compute_dtype = record.compute_dtype
        self.representation = representation
        self.zero_point_domain = "int" if record.zero_is_integer else "float"

        if record.n_groups == 1:
            self.scheme = "per_row"
        elif record.contiguous_groups:
            self.scheme = "per_group"
        else:
            self.scheme = "per_group_permuted"
        decomposed = (
            representation == "decomposed"
            and self.zero_point_domain == "int"
            and self.scheme != "per_group_permuted"
            and record.compute_dtype == torch.float32
        )
        self.dequant_op = (
            "quantized_decomposed.dequantize_per_channel" if decomposed and self.scheme == "per_row"
            else "quantized_decomposed.dequantize_per_channel_group" if decomposed
            else "aten"
        )

        self.register_buffer("weight_q", record.codes.contiguous())
        self.register_buffer("scale", record.scale.to(record.compute_dtype).contiguous())
        zero = record.zero.to(torch.int32 if decomposed else record.compute_dtype)
        self.register_buffer("zero", zero.contiguous())
        self.g_idx: torch.Tensor | None
        if record.g_idx is not None:
            self.register_buffer("g_idx", record.g_idx.to(torch.int64).contiguous())
        else:
            self.g_idx = None
        self.bias = None if bias is None else nn.Parameter(bias.detach().clone(), requires_grad=False)

    @property
    def n_groups(self) -> int:
        return self.scale.shape[1]

    def dequantize(self) -> torch.Tensor:
        """The (out_features, in_features) weight in `weight_dtype`."""
        rows, cols = self.out_features, self.in_features
        if self.dequant_op == "quantized_decomposed.dequantize_per_channel":
            return torch.ops.quantized_decomposed.dequantize_per_channel(
                self.weight_q, self.scale.view(rows), self.zero.view(rows), 0,
                self.quant_min, self.quant_max, self.weight_q.dtype, out_dtype=self.weight_dtype,
            )
        if self.dequant_op == "quantized_decomposed.dequantize_per_channel_group":
            return torch.ops.quantized_decomposed.dequantize_per_channel_group(
                self.weight_q, self.scale, self.zero, self.quant_min, self.quant_max,
                self.weight_q.dtype, cols // self.n_groups, self.weight_dtype,
            )
        # Explicit ATen: the same `scale * (q - zero)` fakequant._apply computes.
        q = self.weight_q.to(self.compute_dtype)
        if self.g_idx is not None:
            scale = self.scale.index_select(1, self.g_idx)
            zero = self.zero.index_select(1, self.g_idx)
            w = scale * (q - zero)
        elif self.n_groups == 1:
            w = self.scale * (q - self.zero)
        else:
            gs = cols // self.n_groups
            w = self.scale.unsqueeze(2) * (q.view(rows, self.n_groups, gs) - self.zero.unsqueeze(2))
            w = w.view(rows, cols)
        return w.to(self.weight_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.dequantize(), self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, bits={self.bits}, "
            f"group_size={self.group_size}, scheme={self.scheme}, dequant={self.dequant_op}, "
            f"bias={self.bias is not None}"
        )

    def describe(self) -> dict:
        return {
            "algo": self.algo,
            "bits": self.bits,
            "group_size": self.group_size,
            "sym": self.sym,
            "scheme": self.scheme,
            "dequant_op": self.dequant_op,
            "storage_dtype": str(self.weight_q.dtype).replace("torch.", ""),
            "quant_min": self.quant_min,
            "quant_max": self.quant_max,
            "zero_point_domain": self.zero_point_domain,
            "compute_dtype": str(self.compute_dtype).replace("torch.", ""),
            "weight_dtype": str(self.weight_dtype).replace("torch.", ""),
            "shape": [self.out_features, self.in_features],
            "n_groups": self.n_groups,
        }


def swap_in_quantized(
    model: nn.Module,
    records: dict[str, QuantRecord],
    *,
    representation: str = "decomposed",
    family: families.Family | None = None,
) -> dict[str, QuantizedLinear]:
    """Replace every recorded target Linear of `model` with a QuantizedLinear, in place.

    Returns the new modules by qualified name. The model keeps everything else
    (embeddings, norms, lm_head, AWQ-folded predecessors) as the quantizer left it.
    """
    targets = families.target_modules(model, family)
    missing = sorted(set(records) - set(targets))
    if missing:
        raise KeyError(f"records name modules the model does not have: {missing[:5]}")
    swapped: dict[str, QuantizedLinear] = {}
    for name, linear in targets.items():
        rec = records.get(name)
        if rec is None:
            continue
        if tuple(rec.codes.shape) != tuple(linear.weight.shape):
            raise ValueError(f"{name}: record shape {tuple(rec.codes.shape)} != weight {tuple(linear.weight.shape)}")
        new = QuantizedLinear(
            rec, bias=linear.bias, weight_dtype=linear.weight.dtype, representation=representation
        ).to(linear.weight.device)
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, new)
        swapped[name] = new
    return swapped
