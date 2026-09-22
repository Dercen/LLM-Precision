"""PLAN.md 5.6 and 5.1: window counts, tokenizer class, and token-id hashes."""

from __future__ import annotations

import pytest

from ptqbench.data import datasets as ds

# (model, expected tokenizer class) -- PLAN.md 5.1
TOKENIZER_CLASSES = {
    "facebook/opt-125m": "GPT2Tokenizer",
    "HuggingFaceTB/SmolLM2-135M": "GPT2Tokenizer",
    "NousResearch/Llama-2-7b-hf": "LlamaTokenizer",
    "unsloth/Meta-Llama-3.1-8B": "TokenizersBackend",
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


@pytest.fixture(scope="module")
def llama_tokenizer():
    from transformers import AutoTokenizer

    from ptqbench import config as C
    from ptqbench import paths

    spec = C.load_model("llama-2-7b").with_mirror()
    return AutoTokenizer.from_pretrained(spec.repo, cache_dir=str(paths.hf_home()), revision=spec.revision)


@pytest.mark.smoke
def test_llama2_tokenizer_class_and_windows(llama_tokenizer):
    """PLAN.md 5.1 / 5.6 for the Llama family, measured 2026-09-22 on the pinned mirror."""
    assert type(llama_tokenizer).__name__ == TOKENIZER_CLASSES["NousResearch/Llama-2-7b-hf"]
    stream = ds.build("wikitext2", llama_tokenizer, seqlen=2048)
    assert stream.n_windows == 166
    assert stream.ids[0, 0].item() == llama_tokenizer.bos_token_id, "one BOS at the stream start"
    assert stream.ids[0, 1:].eq(llama_tokenizer.bos_token_id).sum() == 0, "and nowhere else"
    assert stream.first_token_hash() == "f3c95364766921f6"
    c4 = ds.build("c4_new", llama_tokenizer, seqlen=2048)
    assert c4.n_windows == 256 and c4.first_token_hash() == "049bab220bbdc351"


@pytest.mark.smoke
def test_llama31_tokenizer_class_and_windows():
    """Llama-3.1 on its pinned mirror: 128k vocab, BOS once, and c4_new is NOT 256 here."""
    from transformers import AutoTokenizer

    from ptqbench import config as C
    from ptqbench import paths

    spec = C.load_model("llama-3.1-8b").with_mirror()
    tok = AutoTokenizer.from_pretrained(spec.repo, cache_dir=str(paths.hf_home()), revision=spec.revision)
    assert type(tok).__name__ == TOKENIZER_CLASSES["unsloth/Meta-Llama-3.1-8B"] == spec.tokenizer_class
    assert len(tok) == 128256
    stream = ds.build("wikitext2", tok, seqlen=2048)
    assert stream.n_windows == 141 and stream.first_token_hash() == "30b8e336afa8a93e"
    assert stream.ids[0, 0].item() == tok.bos_token_id and stream.ids[0, 1:].eq(tok.bos_token_id).sum() == 0
    assert ds.build("c4_new", tok, seqlen=2048).n_windows == 252, "first-1100-docs protocol; 256 only for OPT-verbose tokenizers"


@pytest.mark.smoke
def test_model_bytes_estimate_counts_an_untied_lm_head():
    import torch

    from ptqbench import config as C
    from ptqbench.models import loader as ml

    est = ml.estimate_model_bytes(C.load_model("llama-3.1-8b").with_mirror().repo, torch.bfloat16)
    assert abs(est / 1024**3 - 16.06 * 1000**3 / 1024**3) / (16.06 * 1000**3 / 1024**3) < 0.01
    tied = ml.estimate_model_bytes("facebook/opt-125m", torch.float16)
    # Positional embeddings, biases and norms are not estimated: "good to a few percent".
    assert abs(tied - 125_000_000 * 2) / (125_000_000 * 2) < 0.02
