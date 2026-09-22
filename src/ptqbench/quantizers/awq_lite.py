"""AWQ-lite: activation-aware scaling and clipping, then the reference RTN grid.
PLAN.md 6. A reduced re-implementation of Lin et al.'s AWQ on `eval/streaming.py`.

Per decoder block, for each *scale group* -- a set of Linears that share an input X
and have a predecessor the scale can be folded into:

1. s_x = mean|X| per input channel, s_w = mean|W| per input channel over the group.
2. For alpha on a 20-point grid in [0, 1): s = s_x^alpha / s_w^(1-alpha), normalised
   so sqrt(max s * min s) = 1; score = sum over the group of
   ||Q(W diag(s)) (X / s) - W X||^2 under RTN on a token subsample. Keep the best s.
3. Fold: the group's W <- W diag(s) (columns) and the predecessor emits X / s -- a
   norm's weight (and bias) divided by s, or a Linear's output rows divided by s.
4. Clip search per Linear (q/k skipped, as in the reference): shrink the per-row
   (or per-group) max by a ratio in [0.5, 1] and keep the ratio with the least
   output error, per output row.

Blocks are visited in order with the *scaled, clipped, still-fp* weights of earlier
blocks feeding later ones, and every target Linear is RTN-quantized only at the end,
as the reference does. "Lite" names the two simplifications: the alpha search scores
the group's own Linear outputs rather than the enclosing attention/MLP output, and
the clip search uses the same subsample rather than the reference's separate one.

Scale groups are what makes or breaks this. A group only exists when the fold is
exact: a pre-LN norm feeding q/k/v (post-LN OPT-350m has none, so those groups are
skipped and recorded); v_proj feeding o_proj only when their shapes match (GQA
repeats v channels across heads, so a per-o-channel scale is not a per-v-channel
scale -- the reference skips it too); fc1 feeding fc2 because ReLU is positively
homogeneous; up_proj feeding down_proj because the gated product is linear in up.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from ..eval import streaming
from ..models import families
from . import fakequant as fq
from . import rtn as rtn_mod


@dataclass(frozen=True)
class AWQSpec:
    bits: int
    group_size: int = 128
    sym: bool = False
    grid: int = 20
    clip: bool = True
    clip_grid: int = 20
    clip_max_shrink: float = 0.5
    #: Tokens used to score alpha and clip candidates; more is slower, not better.
    score_tokens: int = 8192

    def as_row_fields(self) -> dict[str, Any]:
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "sym": self.sym,
            "awq_grid": self.grid,
            "awq_clip": self.clip,
        }


@dataclass
class ScaleGroup:
    name: str
    linears: dict[str, nn.Linear]  # name -> module, all sharing input X
    prev: nn.Module
    prev_kind: str  # "norm" | "linear_rows"


@dataclass
class AWQReport:
    spec: AWQSpec
    n_modules: int
    quant_seconds: float
    n_groups_scaled: int
    n_groups_skipped: int
    skipped: list[str] = field(default_factory=list)
    alphas: list[float] = field(default_factory=list)
    mean_relative_error: float = 0.0
    eval_mode: str = "resident"

    def as_row_fields(self) -> dict[str, Any]:
        return {
            "algo": "awq_lite",
            **self.spec.as_row_fields(),
            "n_quantized_modules": self.n_modules,
            "quant_seconds": round(self.quant_seconds, 3),
            "awq_groups_scaled": self.n_groups_scaled,
            "awq_groups_skipped": self.n_groups_skipped,
            "awq_skipped": self.skipped,
            "awq_mean_alpha": round(sum(self.alphas) / len(self.alphas), 4) if self.alphas else None,
            "mean_relative_error": round(self.mean_relative_error, 6),
            "quant_eval_mode": self.eval_mode,
        }


# ---- scale groups -----------------------------------------------------------------


def _get(block: nn.Module, path: str) -> nn.Module | None:
    node: Any = block
    for part in path.split("."):
        node = getattr(node, part, None)
        if node is None:
            return None
    return node


def scale_groups(block: nn.Module, family: families.Family) -> tuple[list[ScaleGroup], list[str]]:
    """The exact-fold groups of one block, plus the reasons any were skipped."""
    groups: list[ScaleGroup] = []
    skipped: list[str] = []

    if family.name == "opt":
        pre_ln = bool(getattr(block, "do_layer_norm_before", True))
        qkv = {n: _get(block, f"self_attn.{n}") for n in ("q_proj", "k_proj", "v_proj")}
        if pre_ln and _get(block, "self_attn_layer_norm") is not None:
            groups.append(ScaleGroup("qkv", qkv, _get(block, "self_attn_layer_norm"), "norm"))
        else:
            skipped.append("qkv: post-LN block has no foldable norm before attention")
        v, o = _get(block, "self_attn.v_proj"), _get(block, "self_attn.out_proj")
        if v is not None and o is not None and v.out_features == o.in_features:
            groups.append(ScaleGroup("out_proj", {"out_proj": o}, v, "linear_rows"))
        else:
            skipped.append("out_proj: v_proj/out_proj shape mismatch")
        if pre_ln and _get(block, "final_layer_norm") is not None:
            groups.append(ScaleGroup("fc1", {"fc1": _get(block, "fc1")}, _get(block, "final_layer_norm"), "norm"))
        else:
            skipped.append("fc1: post-LN block has no foldable norm before the MLP")
        act = getattr(block, "activation_fn", None)
        if _is_positively_homogeneous(act):
            groups.append(ScaleGroup("fc2", {"fc2": _get(block, "fc2")}, _get(block, "fc1"), "linear_rows"))
        else:
            skipped.append(f"fc2: activation {type(act).__name__} is not positively homogeneous")

    elif family.name == "llama":
        qkv = {n: _get(block, f"self_attn.{n}") for n in ("q_proj", "k_proj", "v_proj")}
        groups.append(ScaleGroup("qkv", qkv, _get(block, "input_layernorm"), "norm"))
        v, o = _get(block, "self_attn.v_proj"), _get(block, "self_attn.o_proj")
        if v.out_features == o.in_features:
            groups.append(ScaleGroup("o_proj", {"o_proj": o}, v, "linear_rows"))
        else:
            skipped.append("o_proj: GQA repeats v channels across heads; per-channel fold is not exact")
        gate_up = {n: _get(block, f"mlp.{n}") for n in ("gate_proj", "up_proj")}
        groups.append(ScaleGroup("gate_up", gate_up, _get(block, "post_attention_layernorm"), "norm"))
        groups.append(ScaleGroup("down_proj", {"down_proj": _get(block, "mlp.down_proj")}, _get(block, "mlp.up_proj"), "linear_rows"))
    else:
        skipped.append(f"family {family.name!r} has no scale-group map")

    return groups, skipped


def _is_positively_homogeneous(act: Any) -> bool:
    """ReLU-like: f(x / s) == f(x) / s for s > 0. GELU/SiLU are not."""
    if act is None:
        return False
    name = type(act).__name__.lower() if not callable(act) or isinstance(act, nn.Module) else getattr(act, "__name__", "").lower()
    return name in ("relu", "relu6", "leakyrelu")


# ---- the searches -----------------------------------------------------------------


@torch.no_grad()
def _score(group: ScaleGroup, X: torch.Tensor, s: torch.Tensor, spec: AWQSpec) -> float:
    """sum over the group's Linears of ||Q(W s)(X/s) - W X||^2 under the RTN grid."""
    Xs = X / s
    total = 0.0
    for lin in group.linears.values():
        W = lin.weight.detach().float()
        ref = X @ W.t()
        Wq = fq.quantize_weight(W * s, bits=spec.bits, sym=spec.sym, group_size=spec.group_size)
        total += float(((Xs @ Wq.t() - ref) ** 2).sum())
    return total


@torch.no_grad()
def search_scale(group: ScaleGroup, X: torch.Tensor, spec: AWQSpec) -> tuple[torch.Tensor, float, float]:
    """Best per-input-channel scale on the alpha grid. Returns (s, alpha, loss)."""
    s_x = X.abs().mean(dim=0).clamp_min(1e-5)
    W_all = torch.cat([lin.weight.detach().float() for lin in group.linears.values()], dim=0)
    s_w = W_all.abs().mean(dim=0).clamp_min(1e-5)

    best_s, best_alpha, best_loss = None, 0.0, float("inf")
    for i in range(spec.grid):
        alpha = i / spec.grid
        s = (s_x.pow(alpha) / s_w.pow(1 - alpha)).clamp_min(1e-4)
        s = s / (s.max() * s.min()).sqrt()
        loss = _score(group, X, s, spec)
        if loss < best_loss:
            best_s, best_alpha, best_loss = s, alpha, loss
    if best_s is None or not torch.isfinite(torch.tensor(best_loss)):
        best_s = torch.ones_like(s_x)
        best_alpha = 0.0
    return best_s, best_alpha, best_loss


@torch.no_grad()
def apply_scale(group: ScaleGroup, s: torch.Tensor) -> None:
    """Exact fold: predecessor emits X / s, the group consumes it with W diag(s)."""
    dtype = next(iter(group.linears.values())).weight.dtype
    if group.prev_kind == "norm":
        group.prev.weight.data.div_(s.to(group.prev.weight.dtype))
        if getattr(group.prev, "bias", None) is not None:
            group.prev.bias.data.div_(s.to(group.prev.bias.dtype))
    elif group.prev_kind == "linear_rows":
        group.prev.weight.data.div_(s.to(group.prev.weight.dtype).unsqueeze(1))
        if getattr(group.prev, "bias", None) is not None:
            group.prev.bias.data.div_(s.to(group.prev.bias.dtype))
    else:
        raise ValueError(group.prev_kind)
    for lin in group.linears.values():
        lin.weight.data.mul_(s.to(dtype).unsqueeze(0))


@torch.no_grad()
def search_clip(lin: nn.Linear, X: torch.Tensor, spec: AWQSpec) -> torch.Tensor:
    """Per output row (and per column group), the max-shrink ratio with least error.

    Returns the clipped weight in the Linear's dtype. Never widens the range.
    """
    W = lin.weight.detach().float()
    rows, cols = W.shape
    gs = spec.group_size if spec.group_size != -1 else cols
    if cols % gs != 0:
        raise ValueError(f"in_features {cols} not divisible by group_size {gs}")
    n_groups = cols // gs
    Wg = W.reshape(rows, n_groups, gs)
    Xg = X.reshape(X.shape[0], n_groups, gs)  # (tokens, groups, gs)
    ref = torch.einsum("tgc,rgc->rgt", Xg, Wg)  # per-row, per-group partial outputs
    max_abs = Wg.abs().amax(dim=2, keepdim=True)

    best_err = torch.full((rows, n_groups), float("inf"), device=W.device)
    best_ratio = torch.ones((rows, n_groups), device=W.device)
    for i in range(spec.clip_grid + 1):
        ratio = 1.0 - i * (1.0 - spec.clip_max_shrink) / spec.clip_grid
        bound = max_abs * ratio
        Wc = torch.clamp(Wg, -bound, bound).reshape(rows, cols)
        Wq = fq.quantize_weight(Wc, bits=spec.bits, sym=spec.sym, group_size=spec.group_size).reshape(rows, n_groups, gs)
        err = ((torch.einsum("tgc,rgc->rgt", Xg, Wq) - ref) ** 2).sum(dim=2)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_ratio = torch.where(better, torch.full_like(best_ratio, ratio), best_ratio)
    bound = max_abs * best_ratio.unsqueeze(2)
    return torch.clamp(Wg, -bound, bound).reshape(rows, cols).to(lin.weight.dtype)


# ---- the driver -------------------------------------------------------------------


def _collect_into(sink: list[torch.Tensor]):
    """A forward hook that appends the module's (tokens, in_features) input to `sink`."""

    def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: Any) -> None:
        x = inputs[0].detach()
        sink.append(x.reshape(-1, x.shape[-1]).float())

    return hook



@torch.no_grad()
def apply_awq_lite(
    model: nn.Module,
    calib_windows: torch.Tensor,
    *,
    spec: AWQSpec,
    device: torch.device,
    family: families.Family | None = None,
    cache_device: str = "auto",
    offload: bool | None = None,
) -> AWQReport:
    fam = family or families.for_model(model)
    started = time.perf_counter()

    n_windows, seqlen = calib_windows.shape
    dtype = next(model.parameters()).dtype
    blocks = fam.block_list(model)
    largest_block = max(sum(p.numel() * p.element_size() for p in b.parameters()) for b in blocks)
    cache_dev = streaming.resolve_cache_device(
        n_windows, seqlen, model.config.hidden_size, dtype, device, extra_bytes=largest_block, spec=cache_device
    )

    # Which windows score the searches: an evenly spaced subsample of the calibration set.
    tokens_per_window = seqlen
    n_score_windows = max(1, min(n_windows, spec.score_tokens // tokens_per_window))
    score_idx = torch.linspace(0, n_windows - 1, n_score_windows).round().long().tolist()

    all_skipped: list[str] = []
    alphas: list[float] = []
    n_scaled = 0

    with streaming.BlockStreamer(model, fam, device, offload=offload) as streamer:
        cap = streaming.capture_block_inputs(model, fam, calib_windows, device, cache_dev)
        inps, outs = cap.hidden, torch.empty_like(cap.hidden)

        for bi, block in streamer.blocks():
            groups, skipped = scale_groups(block, fam)
            all_skipped.extend(f"block{bi}/{s}" for s in skipped)

            # 1. Collect X for every group's first Linear on the scoring subsample.
            feats: dict[str, list[torch.Tensor]] = {g.name: [] for g in groups}
            handles = []
            for g in groups:
                first = next(iter(g.linears.values()))
                handles.append(first.register_forward_hook(_collect_into(feats[g.name])))
            sub = streaming.CapturedInputs(hidden=inps[score_idx], args=cap.args, kwargs=cap.kwargs)
            try:
                streaming.forward_block(block, sub.hidden, sub, device)
            finally:
                for h in handles:
                    h.remove()

            # 2-3. Scale search and fold, group by group (later groups see earlier folds
            #      only through the block's own forward, exactly as the reference).
            for g in groups:
                X = torch.cat(feats[g.name], dim=0)
                s, alpha, _ = search_scale(g, X, spec)
                apply_scale(g, s)
                alphas.append(alpha)
                n_scaled += 1
            feats = {}

            # 4. Clip search on the scaled weights, with fresh inputs (the folds changed X).
            if spec.clip:
                targets = families.block_targets(block, fam)
                clip_targets = {n: m for n, m in targets.items() if not n.endswith(("q_proj", "k_proj"))}
                feats2: dict[str, list[torch.Tensor]] = {n: [] for n in clip_targets}
                handles = [m.register_forward_hook(_collect_into(feats2[n])) for n, m in clip_targets.items()]
                try:
                    streaming.forward_block(block, sub.hidden, sub, device)
                finally:
                    for h in handles:
                        h.remove()
                for n, m in clip_targets.items():
                    m.weight.data.copy_(search_clip(m, torch.cat(feats2[n], dim=0), spec))
                feats2 = {}

            # Next block sees the scaled + clipped, still-fp outputs.
            streaming.forward_block(block, inps, cap, device, out=outs)
            inps, outs = outs, inps
            if device.type == "cuda":
                torch.cuda.empty_cache()

        eval_mode = streamer.eval_mode

    # 5. Quantize everything with the reference grid.
    rep = rtn_mod.apply_rtn(model, bits=spec.bits, group_size=spec.group_size, sym=spec.sym, family=fam)

    return AWQReport(
        spec=spec,
        n_modules=rep.n_modules,
        quant_seconds=time.perf_counter() - started,
        n_groups_scaled=n_scaled,
        n_groups_skipped=len(all_skipped),
        skipped=sorted({s.split("/", 1)[1] for s in all_skipped}),
        alphas=alphas,
        mean_relative_error=rep.mean_relative_error,
        eval_mode=eval_mode,
    )
