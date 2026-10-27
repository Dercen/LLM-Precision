"""One piece of a model as its own function: a decoder block, its attention, or its MLP.

`ptq export-mlir --slice block --block-index 3` exports `block_3(hidden_states)`: the
block exactly as the model runs it on a real text sample, with every other input the
block receives -- rotary cos/sin tables, position ids, the (absent) mask and cache
flags -- captured by a forward hook and frozen into the function as constants. The
captured hidden states are the realistic input of the piece and its captured output
is the golden answer, so a consumer can test one block of SmolLM2 (hidden size 576)
without running the whole model.

Pieces, through the family map (`models/families.py`):

* `block`: `blocks[i](hidden_states, **kwargs)`; one `[1, S, hidden]` tensor in and out.
* `attention`: `blocks[i].self_attn(hidden_states, **kwargs)`; the function returns only
  `attn_output`, not the attention weights. Its input is the post-norm hidden state.
* `mlp`: Llama's `blocks[i].mlp`, or OPT's `fc1 -> activation -> fc2` chain, which OPT
  runs on a flattened `[S, hidden]` tensor (the block reshapes before `fc1`), so an OPT
  MLP slice is 2-D. In both families the residual add stays outside the piece.

Capturing uses `register_forward_pre_hook(with_kwargs=True)` on the first module of
the piece and `register_forward_hook` on the last, so the OPT MLP (two modules) and the
one-module pieces share one code path. Attention receives `hidden_states` as a keyword
argument in transformers 5.17, blocks receive it positionally; both are handled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from ..models import families

#: The pieces a slice export can cut out.
SLICE_KINDS = ("block", "attention", "mlp")
#: The keyword the attention module receives its primary input under.
HIDDEN_KWARG = "hidden_states"
#: Names of the slice function's input and output tensors in the golden index.
SLICE_INPUT_NAME = "hidden_states"
SLICE_OUTPUT_NAMES = {"block": "hidden_states_out", "attention": "attn_output", "mlp": "mlp_output"}
#: Which element of the piece's return value is the function's result: attention returns
#: `(attn_output, attn_weights)`, the others a bare tensor.
OUTPUT_INDEX = {"block": None, "attention": 0, "mlp": None}


def function_name(kind: str, block_index: int) -> str:
    """`block_3`, `attention_3`, `mlp_3`: the func.func name and the file stem."""
    if kind not in SLICE_KINDS:
        raise ValueError(f"slice must be one of {SLICE_KINDS}, got {kind!r}")
    return f"{kind}_{block_index}"


@dataclass
class CapturedPiece:
    """What one forward pass through the real model handed the piece, verbatim."""

    kind: str
    block_index: int
    piece: nn.Module  # the module the exported function wraps (shares the model's weights)
    hidden: torch.Tensor  # the primary input, as received
    args: tuple[Any, ...]  # positional arguments after the primary input (expected empty)
    kwargs: dict[str, Any]  # keyword arguments, verbatim
    output: torch.Tensor  # the tensor the function returns

    @property
    def function(self) -> str:
        return function_name(self.kind, self.block_index)


def _bounds(block: nn.Module, family: families.Family, kind: str) -> tuple[nn.Module, nn.Module, nn.Module]:
    """(module whose input starts the piece, module whose output ends it, the piece)."""
    if kind == "block":
        return block, block, block
    if kind == "attention":
        attn = family.attention_module(block)
        return attn, attn, attn
    if kind == "mlp":
        first, last = family.mlp_bounds(block)
        return first, last, family.mlp_module(block)
    raise ValueError(f"slice must be one of {SLICE_KINDS}, got {kind!r}")


@torch.no_grad()
def capture_piece(
    model: nn.Module, family: families.Family, kind: str, block_index: int, input_ids: torch.Tensor
) -> CapturedPiece:
    """Run `model(input_ids)` once and record what piece `kind` of block `block_index` saw."""
    blocks = family.block_list(model)
    if not 0 <= block_index < len(blocks):
        raise IndexError(f"block index {block_index} out of range: the model has {len(blocks)} blocks")
    first, last, piece = _bounds(blocks[block_index], family, kind)
    seen: dict[str, Any] = {}

    def on_input(module, args, kwargs):
        if "args" in seen:
            raise RuntimeError(f"{kind} {block_index} ran twice in one forward pass")
        seen["args"], seen["kwargs"] = tuple(args), dict(kwargs)

    def on_output(module, args, output):
        seen["output"] = output

    handles = [first.register_forward_pre_hook(on_input, with_kwargs=True), last.register_forward_hook(on_output)]
    try:
        model(input_ids=input_ids)
    finally:
        for h in handles:
            h.remove()
    if "args" not in seen or "output" not in seen:
        raise RuntimeError(f"{kind} {block_index} never ran; is the family map right for this model?")

    args, kwargs = seen["args"], seen["kwargs"]
    if args:
        hidden, rest = args[0], tuple(args[1:])
    elif HIDDEN_KWARG in kwargs:
        hidden, rest = kwargs.pop(HIDDEN_KWARG), ()
    else:
        raise RuntimeError(f"{kind} {block_index} received no hidden states (args={len(args)}, kwargs={sorted(kwargs)})")
    output = seen["output"]
    index = OUTPUT_INDEX[kind]
    if index is not None:
        output = output[index]
    if not torch.is_tensor(output):
        raise RuntimeError(f"{kind} {block_index} returned {type(output).__name__}, not a tensor")
    return CapturedPiece(kind, block_index, piece, hidden.detach(), rest, kwargs, output.detach())


class Replay(nn.Module):
    """`piece(hidden_states, *args, **kwargs)` with every captured argument a constant.

    Captured tensors (and tuples of tensors, such as Llama's `(cos, sin)` rotary pair)
    become buffers, so they are embedded like any other weight or lifted as named
    arguments with `--weights external`. Captured Python values (`None`, `False`, ints)
    are replayed as they were. Anything else (a Cache object, say) is refused: a piece
    that depends on live state is not a pure function of its hidden states.
    """

    def __init__(self, captured: CapturedPiece):
        super().__init__()
        self.piece = captured.piece
        self.kind = captured.kind
        self.output_index = OUTPUT_INDEX[captured.kind]
        self._const_args = tuple(captured.args)
        if any(torch.is_tensor(a) for a in self._const_args):
            raise ValueError("positional tensor arguments after hidden_states are not supported")
        self._spec: dict[str, tuple[str, Any]] = {}  # kwarg -> ("buffer", name) | ("tuple", [names]) | ("value", v)
        for key, value in captured.kwargs.items():
            if torch.is_tensor(value):
                self.register_buffer(key, value.detach().clone())
                self._spec[key] = ("buffer", key)
            elif isinstance(value, tuple) and value and all(torch.is_tensor(v) for v in value):
                names = []
                for j, v in enumerate(value):
                    name = f"{key}_{j}"
                    self.register_buffer(name, v.detach().clone())
                    names.append(name)
                self._spec[key] = ("tuple", names)
            elif value is None or isinstance(value, (bool, int, float, str)):
                self._spec[key] = ("value", value)
            else:
                raise ValueError(f"captured kwarg {key!r} of type {type(value).__name__} cannot be replayed as a constant")

    def captured_kwargs(self) -> dict[str, Any]:
        """A JSON-able description of what was frozen in, for export.json."""
        out: dict[str, Any] = {}
        for key, (how, what) in self._spec.items():
            if how == "buffer":
                t = getattr(self, what)
                out[key] = {"buffer": what, "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")}
            elif how == "tuple":
                out[key] = [
                    {"buffer": n, "shape": list(getattr(self, n).shape), "dtype": str(getattr(self, n).dtype).replace("torch.", "")}
                    for n in what
                ]
            else:
                out[key] = what
        return out

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        kwargs: dict[str, Any] = {}
        for key, (how, what) in self._spec.items():
            if how == "buffer":
                kwargs[key] = getattr(self, what)
            elif how == "tuple":
                kwargs[key] = tuple(getattr(self, n) for n in what)
            else:
                kwargs[key] = what
        out = self.piece(hidden_states, *self._const_args, **kwargs)
        return out if self.output_index is None else out[self.output_index]


@dataclass
class SliceReport:
    """What a slice export records in export.json."""

    kind: str
    block_index: int
    function: str
    input_shape: list[int]
    output_shape: list[int]
    dtype: str
    captured_kwargs: dict[str, Any]
    replay_bitwise: bool
    quantized_modules: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@torch.no_grad()
def build_replay(captured: CapturedPiece) -> tuple[Replay, SliceReport]:
    """The exportable function for a captured piece, checked to reproduce the capture bitwise."""
    replay = Replay(captured).eval()
    again = replay(captured.hidden)
    bitwise = torch.equal(again, captured.output)
    if not bitwise:
        raise RuntimeError(f"{captured.function}: replaying the captured inputs did not reproduce the captured output")
    quantized = sorted(name for name, m in captured.piece.named_modules() if hasattr(m, "weight_q"))
    report = SliceReport(
        kind=captured.kind, block_index=captured.block_index, function=captured.function,
        input_shape=list(captured.hidden.shape), output_shape=list(captured.output.shape),
        dtype=str(captured.hidden.dtype).replace("torch.", ""),
        captured_kwargs=replay.captured_kwargs(), replay_bitwise=bitwise, quantized_modules=quantized,
    )
    return replay, report
