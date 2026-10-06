"""Export stage: a quantized model from the pipeline as a torch.nn.Module with explicit
dequantization, and as torch-dialect MLIR via torch-mlir. See `quantized.py` for what maps
onto torch-mlir's quantized-op support and what does not."""

from .quantized import QuantizedLinear, swap_in_quantized
from .stage import ExportOptions, ExportResult, export_quantized

__all__ = ["ExportOptions", "ExportResult", "QuantizedLinear", "export_quantized", "swap_in_quantized"]
