"""Perplexity, exactly as PLAN.md 5 specifies, with the chunked cross-entropy of 5b.

The reference protocol (IST-DASLab/gptq, shared by GPTQ, AWQ, SmoothQuant, OmniQuant)
takes a MEAN over the 2047 predictions of a 2048-token window and then multiplies by
2048. That quirk is preserved. Because it is a mean of a sum, it can be computed
chunk-wise: accumulate the per-token NLL sum in fp32, divide by 2047 once, multiply
by 2048. Chunking never materialises more than `ce_chunk x vocab` logits, which is
what makes a 128k-vocab model evaluable on an 8 GB card.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

DEFAULT_CE_CHUNK = 256


@dataclass
class PerplexityResult:
    ppl: float
    nll_sum: float
    n_tokens: int
    n_windows: int
    seqlen: int
    partial: bool
    eval_seconds: float
    ce_chunk: int
    eval_mode: str = "resident"
    peak_vram_gb: float | None = None
    peak_ram_gb: float | None = None
    per_window_nll: list[float] = field(default_factory=list)

    def as_row_fields(self) -> dict[str, Any]:
        return {
            "ppl": self.ppl,
            "nll_sum": self.nll_sum,
            "n_tokens": self.n_tokens,
            "n_windows": self.n_windows,
            "seqlen": self.seqlen,
            "partial": self.partial,
            "eval_seconds": round(self.eval_seconds, 3),
            "ce_chunk": self.ce_chunk,
            "eval_mode": self.eval_mode,
            "peak_vram_gb": self.peak_vram_gb,
            "peak_ram_gb": self.peak_ram_gb,
        }


def _decoder_and_head(model) -> tuple[Any, Any] | None:
    """The pre-lm_head module and the lm_head, when the model exposes them cleanly.

    Returning None means the caller must fall back to full-logits evaluation.
    """
    try:
        decoder = model.get_decoder()
        head = model.get_output_embeddings()
    except (AttributeError, NotImplementedError):
        return None
    if decoder is None or head is None:
        return None
    return decoder, head


def _window_nll_sum_chunked(model, ids: torch.Tensor, ce_chunk: int) -> torch.Tensor:
    """Sum of per-token NLL over the window's 2047 predictions, accumulated in fp32.

    Runs the decoder once, then applies lm_head in slices so the full
    (1, seqlen, vocab) logits tensor is never materialised.
    """
    parts = _decoder_and_head(model)
    if parts is None or ce_chunk <= 0:
        return _window_nll_sum_full(model, ids)

    decoder, head = parts
    out = decoder(input_ids=ids, use_cache=False)
    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

    seqlen = ids.shape[1]
    targets = ids[0, 1:]  # the 2047 predicted tokens
    total = torch.zeros((), dtype=torch.float64, device=hidden.device)

    # Position p in [0, seqlen-1) predicts token p+1.
    for start in range(0, seqlen - 1, ce_chunk):
        stop = min(start + ce_chunk, seqlen - 1)
        logits = head(hidden[:, start:stop, :]).float()
        nll = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets[start:stop],
            reduction="sum",
        )
        total += nll.double()
        del logits, nll

    return total


def _window_nll_sum_full(model, ids: torch.Tensor) -> torch.Tensor:
    """Unchunked reference path: materialise all logits, then one cross-entropy."""
    logits = model(input_ids=ids, use_cache=False).logits
    shifted = logits[:, :-1, :].float()
    nll = F.cross_entropy(
        shifted.reshape(-1, shifted.shape[-1]),
        ids[0, 1:],
        reduction="sum",
    )
    return nll.double()


@torch.no_grad()
def evaluate(
    model,
    stream,
    *,
    device: torch.device,
    max_windows: int | None = None,
    ce_chunk: int = DEFAULT_CE_CHUNK,
    eval_mode: str = "resident",
    progress: bool = True,
) -> PerplexityResult:
    """PLAN.md 5.3-5.5. Batch 1, no grad, tail window dropped by the TokenStream."""
    seqlen = stream.seqlen
    n_windows = stream.n_windows
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)
    if n_windows == 0:
        raise ValueError("no complete windows to evaluate")

    partial = max_windows is not None and max_windows < stream.n_windows or seqlen < 2048

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    peak_ram_before = _rss_gb()

    iterator: Any = range(n_windows)
    if progress and _is_tty():
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc=f"ppl[{stream.key}]", unit="win")
        except ImportError:
            pass

    nll_sum = 0.0
    per_window: list[float] = []
    started = time.perf_counter()

    for i in iterator:
        ids = stream.window(i).to(device)
        window_sum = _window_nll_sum_chunked(model, ids, ce_chunk)
        # The reference quirk: mean over seqlen-1 predictions, scaled back to seqlen.
        nll_i = float(window_sum.item()) / (seqlen - 1) * seqlen
        per_window.append(nll_i)
        nll_sum += nll_i

    elapsed = time.perf_counter() - started
    n_tokens = n_windows * seqlen
    ppl = math.exp(nll_sum / n_tokens)

    peak_vram = None
    if device.type == "cuda":
        peak_vram = round(torch.cuda.max_memory_allocated(device) / 1024**3, 3)

    return PerplexityResult(
        ppl=ppl,
        nll_sum=nll_sum,
        n_tokens=n_tokens,
        n_windows=n_windows,
        seqlen=seqlen,
        partial=bool(partial),
        eval_seconds=elapsed,
        ce_chunk=ce_chunk,
        eval_mode=eval_mode,
        peak_vram_gb=peak_vram,
        peak_ram_gb=round(max(_rss_gb(), peak_ram_before), 3),
        per_window_nll=per_window,
    )


def _is_tty() -> bool:
    """Progress bars belong on a terminal; in a pipe or log they are just noise."""
    import sys

    return bool(getattr(sys.stderr, "isatty", lambda: False)())


def _rss_gb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024**3
    except Exception:  # noqa: BLE001 - RAM reporting is diagnostic, never fatal.
        return 0.0
