"""PLAN.md 5c: streamed evaluation is the same protocol as resident evaluation.

Measured 2026-09-21 on opt-125m, opt-350m (project_out, no final norm) and
SmolLM2-135M (Llama rotary kwargs): resident and streamed agree to 0.0 exactly.
The gate is 1e-4 so a future kernel change cannot turn a real discrepancy into flake.
"""

from __future__ import annotations

import pytest
import torch

from ptqbench import device as D
from ptqbench.data import datasets as ds
from ptqbench.eval import perplexity as P
from ptqbench.eval import streaming
from ptqbench.models import families
from ptqbench.models import loader as ml

pytestmark = pytest.mark.gpu

# Three topologies the family map must get right: OPT pre-LN, OPT post-LN with
# project_in/out, and Llama with rotary position_embeddings passed as kwargs.
MODELS = ["facebook/opt-125m", "facebook/opt-350m", "HuggingFaceTB/SmolLM2-135M"]


def _need_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


@pytest.fixture(scope="module", params=MODELS)
def loaded(request):
    _need_cuda()
    D.lock_numerics()
    dev = D.resolve("auto")
    lm = ml.load(request.param, device=dev)
    stream = ds.build("wikitext2", lm.tokenizer, seqlen=2048)
    return lm, stream, dev


def test_streamed_matches_resident_when_resident(loaded):
    lm, stream, dev = loaded
    ref = P.evaluate(lm.model, stream, device=dev, max_windows=8, progress=False)
    got = P.evaluate_streamed(
        lm.model, stream, device=dev, max_windows=8, window_batch=3, offload=False, progress=False
    )
    assert got.eval_mode == "resident"
    assert abs(got.ppl - ref.ppl) < 1e-4, (got.ppl, ref.ppl)


def test_streamed_matches_resident_when_offloaded(loaded):
    """The real streamed tier: weights in host RAM, one block on the card at a time."""
    lm, stream, dev = loaded
    ref = P.evaluate(lm.model, stream, device=dev, max_windows=8, progress=False)
    lm.model.cpu()
    torch.cuda.empty_cache()
    try:
        got = P.evaluate_streamed(
            lm.model, stream, device=dev, max_windows=8, window_batch=3, offload=True, progress=False
        )
    finally:
        lm.model.to(dev)
    assert got.eval_mode == "streamed"
    assert abs(got.ppl - ref.ppl) < 1e-4, (got.ppl, ref.ppl)
    assert got.peak_vram_gb < ref.peak_vram_gb, "offloading must lower peak VRAM"


def test_offload_restores_the_model_to_cpu(loaded):
    """BlockStreamer must leave a host-resident model host-resident."""
    lm, _stream, dev = loaded
    fam = families.for_model(lm.model)
    lm.model.cpu()
    try:
        with streaming.BlockStreamer(lm.model, fam, dev, offload=True) as s:
            assert s.eval_mode == "streamed"
            for _, block in s.blocks():
                assert next(block.parameters()).device == dev
                break
        assert all(p.device.type == "cpu" for p in lm.model.parameters())
    finally:
        lm.model.to(dev)


def test_catcher_records_every_window_and_the_kwargs(loaded):
    lm, stream, dev = loaded
    fam = families.for_model(lm.model)
    windows = torch.cat([stream.window(i) for i in range(3)], dim=0)
    cap = streaming.capture_block_inputs(lm.model, fam, windows, dev, dev)
    assert cap.n == 3
    assert cap.hidden.shape == (3, 2048, lm.model.config.hidden_size)
    assert cap.args == ()
    assert "position_ids" in cap.kwargs
    if fam.name == "llama":
        assert "position_embeddings" in cap.kwargs, "Llama needs rotary kwargs replayed"


def test_window_batch_size_does_not_change_the_answer(loaded):
    lm, stream, dev = loaded
    a = P.evaluate_streamed(lm.model, stream, device=dev, max_windows=7, window_batch=2, offload=False, progress=False)
    b = P.evaluate_streamed(lm.model, stream, device=dev, max_windows=7, window_batch=7, offload=False, progress=False)
    assert a.nll_sum == b.nll_sum
