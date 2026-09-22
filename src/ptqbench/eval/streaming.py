"""Block streaming: the one primitive behind GPTQ and --stream-blocks. PLAN.md 4, 5c.

Walk the decoder blocks in order, one block on the accelerator at a time, pushing a
set of hidden states through each. GPTQ pushes every calibration window and pauses
per block to quantize; streamed evaluation pushes a batch of evaluation windows and
finishes with the post-block modules and the chunked lm_head. Building it once means
every GPTQ run on a small model exercises the machinery the 7B evaluations rely on.

Two facts about transformers 5.17 shape this file, both measured rather than assumed
(`OPTDecoderLayer` and `LlamaDecoderLayer`, 2026-09-21):

* A decoder layer returns a bare Tensor, not a tuple. The reference's `layer(...)[0]`
  would silently index the batch dimension.
* The model hands each layer `hidden_states` positionally and everything else as
  kwargs: `attention_mask` (None under SDPA), `position_ids`, and for Llama the
  rotary `position_embeddings` tuple. Replaying hidden states alone breaks Llama,
  so the Catcher records `*args` and `**kwargs` verbatim and every block gets them.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Self

import torch
from torch import nn

from .. import device as D
from ..models import families


class _CaughtInputs(Exception):
    """Raised by the Catcher to abort the forward pass once block 0 has been reached."""


@dataclass
class CapturedInputs:
    """Block-0 inputs for N windows, plus the kwargs every block must be replayed with."""

    hidden: torch.Tensor  # (N, seqlen, hidden) on cache_device
    args: tuple[Any, ...]  # positional args after hidden_states; expected empty
    kwargs: dict[str, Any]  # verbatim from the model; tensors stay on the compute device

    @property
    def n(self) -> int:
        return self.hidden.shape[0]


@dataclass
class _Sink:
    hidden: list[torch.Tensor] = field(default_factory=list)
    args: tuple[Any, ...] | None = None
    kwargs: dict[str, Any] | None = None


class Catcher(nn.Module):
    """Stands in for block 0: records what the model passes it, then aborts the pass."""

    def __init__(self, block: nn.Module, sink: _Sink):
        super().__init__()
        self.block = block
        self.sink = sink

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        self.sink.hidden.append(hidden_states.detach())
        if self.sink.kwargs is None:
            self.sink.args = tuple(args)
            self.sink.kwargs = dict(kwargs)
        else:
            _assert_same_shapes(self.sink.kwargs, kwargs)
        raise _CaughtInputs


def _assert_same_shapes(first: dict[str, Any], later: dict[str, Any]) -> None:
    """Every window is batch 1 at one seqlen, so the kwargs must be identical in shape."""
    for key, value in later.items():
        ref = first.get(key)
        if torch.is_tensor(value) and torch.is_tensor(ref) and value.shape != ref.shape:
            raise RuntimeError(f"block kwarg {key!r} changed shape between windows: "
                               f"{tuple(ref.shape)} -> {tuple(value.shape)}")


def _block_container(model: nn.Module, family: families.Family) -> tuple[nn.Module, str]:
    """(parent module, attribute name) of the ModuleList holding the decoder blocks."""
    *parent_path, attr = family.blocks_path.split(".")
    node: Any = model
    for part in parent_path:
        node = getattr(node, part)
    return node, attr


def move_non_block_modules(model: nn.Module, family: families.Family, device: torch.device) -> None:
    """Move embeddings, norms, rotary tables and lm_head; leave the decoder blocks put."""
    parent, attr = _block_container(model, family)
    blocks = getattr(parent, attr)
    setattr(parent, attr, nn.ModuleList())
    try:
        model.to(device)
    finally:
        setattr(parent, attr, blocks)


def resolve_cache_device(
    n_windows: int,
    seqlen: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    extra_bytes: int = 0,
    spec: str = "auto",
) -> torch.device:
    """Where the two hidden-state buffers live: on the accelerator when they fit.

    `extra_bytes` is whatever else must share the card at the same time -- the
    largest block plus, for GPTQ, its Hessians and their Cholesky temporaries.
    """
    if spec != "auto":
        return torch.device(spec)
    if device.type != "cuda":
        return torch.device("cpu")
    cache_bytes = 2 * n_windows * seqlen * hidden_size * dtype.itemsize
    budget = D.vram_available_bytes(device) - int(D.VRAM_RESERVE_GB * 1024**3)
    return device if cache_bytes + extra_bytes < budget else torch.device("cpu")


@torch.no_grad()
def capture_block_inputs(
    model: nn.Module,
    family: families.Family,
    windows: torch.Tensor,
    device: torch.device,
    cache_device: torch.device,
) -> CapturedInputs:
    """Run the model's own embedding path for each window and intercept at block 0."""
    parent, attr = _block_container(model, family)
    blocks = getattr(parent, attr)
    original = blocks[0]
    sink = _Sink()
    blocks[0] = Catcher(original, sink)
    try:
        for i in range(windows.shape[0]):
            try:
                model(input_ids=windows[i : i + 1].to(device), use_cache=False)
            except _CaughtInputs:
                pass
    finally:
        blocks[0] = original

    if not sink.hidden or sink.kwargs is None:
        raise RuntimeError("Catcher never fired; is blocks_path right for this model?")

    hidden = torch.cat([h.to(cache_device) for h in sink.hidden], dim=0)
    return CapturedInputs(hidden=hidden, args=sink.args or (), kwargs=sink.kwargs)


def _unwrap(out: Any) -> torch.Tensor:
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def forward_block(
    block: nn.Module,
    hidden_in: torch.Tensor,
    captured: CapturedInputs,
    device: torch.device,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Push every window through one block, batch 1 at a time like the reference.

    Writes into `out` (same shape and device as `hidden_in`) when given, so the two
    buffers can be swapped between blocks without reallocation.
    """
    if out is None:
        out = torch.empty_like(hidden_in)
    for i in range(hidden_in.shape[0]):
        x = hidden_in[i : i + 1].to(device, non_blocking=True)
        y = _unwrap(block(x, *captured.args, **captured.kwargs))
        out[i].copy_(y[0], non_blocking=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return out


class BlockStreamer:
    """Context manager that yields decoder blocks one at a time on the compute device.

    offload=True:  the model lives in host RAM; non-block modules move to the device
                   for the duration, each block moves in and out as it is visited.
    offload=False: the model is already resident; nothing moves.
    offload=None:  decide from where the parameters currently are.
    """

    def __init__(
        self,
        model: nn.Module,
        family: families.Family,
        device: torch.device,
        *,
        offload: bool | None = None,
    ):
        self.model = model
        self.family = family
        self.device = device
        param_device = next(model.parameters()).device
        self.offload = (param_device != device) if offload is None else offload

    def __enter__(self) -> Self:
        if self.offload:
            move_non_block_modules(self.model, self.family, self.device)
        return self

    def __exit__(self, *exc: object) -> None:
        if self.offload:
            move_non_block_modules(self.model, self.family, torch.device("cpu"))
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def blocks(self) -> Iterator[tuple[int, nn.Module]]:
        for idx, block in enumerate(self.family.block_list(self.model)):
            if self.offload:
                block.to(self.device)
            try:
                yield idx, block
            finally:
                if self.offload:
                    block.to("cpu")
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

    def post_block_modules(self) -> list[nn.Module]:
        return self.family.post_block_modules(self.model)

    @property
    def eval_mode(self) -> str:
        return "streamed" if self.offload else "resident"
