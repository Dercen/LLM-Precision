"""GPTQ. PLAN.md 6. A port of IST-DASLab/gptq's `GPTQ.fasterquant` and the
`opt_sequential` / `llama_sequential` drivers, on top of `eval/streaming.py`.

Per block: forward hooks accumulate H = 2/n * sum(x x^T) over the calibration windows
in fp32; each target Linear is then quantized column by column with OBS error feedback
against the Cholesky factor of the damped inverse Hessian; the block is re-run with its
quantized weights to produce the next block's inputs.

Faithfulness notes, all deliberate:

* The grid is `fakequant.find_params`, i.e. the reference quantizer, so `rtn` and
  `gptq` share one definition of "4-bit" and the parity test covers both.
* Per-row (group_size=-1) finds the scale once on the full weight before any update.
  Grouped, non-static finds it every `group_size` columns on the OUTER W, which
  carries error feedback from previous blocks but not from earlier columns of the
  current block. That is what the reference does; it is replicated, not fixed.
* `nsamples` counts windows, not tokens, so H is (2/nsamples) * sum over windows of
  X X^T. GPTQ is invariant to that scale; only the damping term sees it, exactly as
  in the reference.
* No `true_sequential` by default: the OPT tables in the paper were produced without
  it. With it, each family's `sequential_groups` are quantized in order, re-collecting
  H between groups as llama.py --true-sequential does.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from .. import device as D
from ..eval import streaming
from ..models import families
from . import fakequant as fq


@dataclass(frozen=True)
class GPTQSpec:
    bits: int
    group_size: int = -1
    sym: bool = False
    act_order: bool = False
    true_sequential: bool = False
    static_groups: bool = False
    percdamp: float = 0.01
    blocksize: int = 128

    def as_row_fields(self) -> dict[str, Any]:
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "sym": self.sym,
            "act_order": self.act_order,
            "true_sequential": self.true_sequential,
            "static_groups": self.static_groups,
            "percdamp": self.percdamp,
            "blocksize": self.blocksize,
        }


@dataclass
class GPTQReport:
    spec: GPTQSpec
    n_modules: int
    quant_seconds: float
    total_loss: float
    per_block_loss: list[float] = field(default_factory=list)
    eval_mode: str = "resident"
    cache_device: str = ""
    peak_vram_gb: float | None = None

    def as_row_fields(self) -> dict[str, Any]:
        return {
            "algo": "gptq",
            **self.spec.as_row_fields(),
            "n_quantized_modules": self.n_modules,
            "quant_seconds": round(self.quant_seconds, 3),
            "gptq_total_loss": self.total_loss,
            "gptq_per_block_loss": [round(x, 4) for x in self.per_block_loss],
            "quant_eval_mode": self.eval_mode,
            "quant_cache_device": self.cache_device,
            "quant_peak_vram_gb": self.peak_vram_gb,
        }


class GPTQ:
    """Hessian accumulation and the column-wise OBS quantization of one Linear."""

    def __init__(self, layer: nn.Linear):
        self.layer = layer
        weight = layer.weight
        self.rows, self.columns = weight.shape
        self.dev = weight.device
        self.H: torch.Tensor | None = torch.zeros(
            (self.columns, self.columns), device=self.dev, dtype=torch.float32
        )
        self.nsamples = 0

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor) -> None:
        if self.H is None:
            raise RuntimeError("add_batch after quantize")
        if inp.ndim == 2:
            inp = inp.unsqueeze(0)
        n_new = inp.shape[0]  # the reference counts windows, not tokens
        x = inp.reshape(-1, inp.shape[-1]).t().float()  # (in_features, tokens)
        self.H *= self.nsamples / (self.nsamples + n_new)
        self.nsamples += n_new
        x = math.sqrt(2 / self.nsamples) * x
        self.H += x.matmul(x.t())

    @torch.no_grad()
    def quantize(self, spec: GPTQSpec) -> float:
        """Quantize the layer's weight in place; return the summed OBS loss."""
        if self.H is None:
            raise RuntimeError("quantize called twice")
        weight = self.layer.weight
        W = weight.data.clone().float()
        H = self.H
        self.H = None

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        bits, gs, sym = spec.bits, spec.group_size, spec.sym
        maxq = 2**bits - 1

        grid: fq.QuantGrid | None = None
        if gs == -1:
            grid = fq.find_params(W, bits=bits, sym=sym, group_size=-1)

        static: list[fq.QuantGrid] | None = None
        if spec.static_groups:
            if gs == -1:
                raise ValueError("static_groups needs a group_size")
            static = [
                fq.find_params(W[:, i : i + gs], bits=bits, sym=sym, group_size=-1)
                for i in range(0, self.columns, gs)
            ]

        perm = invperm = None
        if spec.act_order:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = spec.percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        blocksize = spec.blocksize
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if gs != -1:
                    if static is None:
                        if (i1 + i) % gs == 0:
                            grid = fq.find_params(
                                W[:, (i1 + i) : (i1 + i + gs)], bits=bits, sym=sym, group_size=-1
                            )
                    else:
                        idx = i1 + i
                        if perm is not None:
                            idx = int(perm[idx])
                        grid = static[idx // gs]

                assert grid is not None
                q = fq._apply(w.unsqueeze(1), grid.scale, grid.zero, maxq).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            losses[:, i1:i2] = Losses1 / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if invperm is not None:
            Q = Q[:, invperm]

        weight.data.copy_(Q.to(weight.dtype))
        return float(losses.sum().item())

    def free(self) -> None:
        self.H = None


def _hessian_bytes(block: nn.Module, family: families.Family) -> int:
    """Largest single Hessian plus its Cholesky temporaries: in_features^2 fp32 x 2."""
    targets = families.block_targets(block, family)
    if not targets:
        return 0
    return 2 * 4 * max(m.in_features**2 for m in targets.values())


@torch.no_grad()
def apply_gptq(
    model: nn.Module,
    calib_windows: torch.Tensor,
    *,
    spec: GPTQSpec,
    device: torch.device,
    family: families.Family | None = None,
    cache_device: str = "auto",
    offload: bool | None = None,
    progress: bool = True,
) -> GPTQReport:
    """The sequential driver: quantize block by block, feeding each the last's outputs."""
    fam = family or families.for_model(model)
    started = time.perf_counter()
    D.reset_peak_memory(device)

    n_windows, seqlen = calib_windows.shape
    hidden_size = model.config.hidden_size
    dtype = next(model.parameters()).dtype
    blocks = fam.block_list(model)
    largest_block = max(sum(p.numel() * p.element_size() for p in b.parameters()) for b in blocks)
    largest_hessian = max(_hessian_bytes(b, fam) for b in blocks)
    cache_dev = streaming.resolve_cache_device(
        n_windows, seqlen, hidden_size, dtype, device,
        extra_bytes=largest_block + largest_hessian, spec=cache_device,
    )

    per_block_loss: list[float] = []
    n_modules = 0

    with streaming.BlockStreamer(model, fam, device, offload=offload) as streamer:
        cap = streaming.capture_block_inputs(model, fam, calib_windows, device, cache_dev)
        inps, outs = cap.hidden, torch.empty_like(cap.hidden)

        block_iter = streamer.blocks()
        if progress and _is_tty():
            try:
                from tqdm import tqdm

                block_iter = tqdm(block_iter, total=len(blocks), desc="gptq", unit="block")
            except ImportError:
                pass

        for _, block in block_iter:
            targets = families.block_targets(block, fam)
            if spec.true_sequential:
                groups = [
                    [n for n in group if n in targets] for group in fam.sequential_groups
                ]
                groups = [g for g in groups if g]
                missing = set(targets) - {n for g in groups for n in g}
                if missing:
                    raise RuntimeError(f"sequential_groups miss targets: {sorted(missing)}")
            else:
                groups = [list(targets)]

            block_loss = 0.0
            for group in groups:
                workers = {name: GPTQ(targets[name]) for name in group}
                handles = [
                    targets[name].register_forward_hook(
                        lambda _m, inp, _out, w=workers[name]: w.add_batch(inp[0].detach())
                    )
                    for name in group
                ]
                try:
                    streaming.forward_block(block, inps, cap, device, out=outs)
                finally:
                    for h in handles:
                        h.remove()
                for name in group:
                    block_loss += workers[name].quantize(spec)
                    workers[name].free()
                    n_modules += 1
                del workers

            # Re-run with quantized weights: the next block sees what it will see at eval.
            streaming.forward_block(block, inps, cap, device, out=outs)
            inps, outs = outs, inps
            per_block_loss.append(block_loss)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        eval_mode = streamer.eval_mode

    peak = None
    if device.type == "cuda":
        peak = round(torch.cuda.max_memory_allocated(device) / 1024**3, 3)

    return GPTQReport(
        spec=spec,
        n_modules=n_modules,
        quant_seconds=time.perf_counter() - started,
        total_loss=float(sum(per_block_loss)),
        per_block_loss=per_block_loss,
        eval_mode=eval_mode,
        cache_device=str(cache_dev),
        peak_vram_gb=peak,
    )


def _is_tty() -> bool:
    import sys

    return bool(getattr(sys.stderr, "isatty", lambda: False)())
