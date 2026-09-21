"""PLAN.md 5b: chunked lm_head + cross-entropy must equal the full-logits path."""

from __future__ import annotations

import pytest
import torch

from ptqbench import device as D
from ptqbench.data import datasets as ds
from ptqbench.eval import perplexity as P
from ptqbench.models import loader as ml

pytestmark = pytest.mark.gpu

MODEL = "facebook/opt-125m"


@pytest.fixture(scope="module")
def loaded():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    D.lock_numerics()
    dev = D.resolve("auto")
    lm = ml.load(MODEL, device=dev)
    stream = ds.build("wikitext2", lm.tokenizer, seqlen=2048)
    return lm, stream, dev


@pytest.mark.parametrize("chunk", [128, 256, 512, 2048])
def test_chunked_matches_full_logits(loaded, chunk):
    lm, stream, dev = loaded
    ref = P.evaluate(lm.model, stream, device=dev, max_windows=8, ce_chunk=0, progress=False)
    got = P.evaluate(lm.model, stream, device=dev, max_windows=8, ce_chunk=chunk, progress=False)
    assert abs(got.ppl - ref.ppl) < 1e-4, f"chunk={chunk}: {got.ppl} vs {ref.ppl}"


def test_chunking_reduces_peak_vram(loaded):
    """The whole point of 5b: a smaller chunk must cost less VRAM."""
    lm, stream, dev = loaded
    full = P.evaluate(lm.model, stream, device=dev, max_windows=4, ce_chunk=0, progress=False)
    small = P.evaluate(lm.model, stream, device=dev, max_windows=4, ce_chunk=256, progress=False)
    assert small.peak_vram_gb < full.peak_vram_gb


def test_repeat_eval_is_bit_identical(loaded):
    """PLAN.md 5a: with TF32 off and benchmark off, a repeated eval must not drift."""
    lm, stream, dev = loaded
    a = P.evaluate(lm.model, stream, device=dev, max_windows=8, progress=False)
    b = P.evaluate(lm.model, stream, device=dev, max_windows=8, progress=False)
    assert a.nll_sum == b.nll_sum
