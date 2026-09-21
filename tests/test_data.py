"""PLAN.md 5.6 and 5.1: window counts, tokenizer class, and token-id hashes."""

from __future__ import annotations

import pytest

from ptqbench.data import datasets as ds

# (model, expected tokenizer class) -- PLAN.md 5.1
TOKENIZER_CLASSES = {
    "facebook/opt-125m": "GPT2Tokenizer",
    "HuggingFaceTB/SmolLM2-135M": "GPT2Tokenizer",
}


@pytest.fixture(scope="module")
def opt_tokenizer():
    from transformers import AutoTokenizer

    from ptqbench import paths

    return AutoTokenizer.from_pretrained("facebook/opt-125m", cache_dir=str(paths.hf_home()))


@pytest.mark.smoke
def test_registry_lists_expected_keys():
    for key in ("wikitext2", "c4", "c4_new", "ptb", "ptb_new"):
        assert key in ds.available()


@pytest.mark.smoke
def test_opt_tokenizer_class(opt_tokenizer):
    assert type(opt_tokenizer).__name__ == TOKENIZER_CLASSES["facebook/opt-125m"]


@pytest.mark.smoke
def test_wikitext2_window_count_and_hash(opt_tokenizer):
    """~140 windows with the OPT tokenizer; the hash guards against tokenizer drift."""
    stream = ds.build("wikitext2", opt_tokenizer, seqlen=2048)
    assert stream.n_source_rows == 4358
    assert stream.n_windows == 140
    assert stream.split == "test"
    # Recorded 2026-09-21, transformers 5.17.0 / tokenizers 0.23.2.
    assert stream.first_token_hash() == "e7f12b6bf3733f32"


@pytest.mark.smoke
def test_windows_are_non_overlapping_and_exact(opt_tokenizer):
    stream = ds.build("wikitext2", opt_tokenizer, seqlen=2048)
    assert stream.ids.shape[1] == stream.n_windows * 2048
    first, second = stream.window(0), stream.window(1)
    assert first.shape == (1, 2048) and second.shape == (1, 2048)
    assert not (first == second).all()
