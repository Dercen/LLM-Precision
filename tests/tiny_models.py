"""Tiny randomly initialised models for the export tests: one OPT and one Llama, each
two blocks of hidden size 64, plus a 256-word tokenizer. Nothing downloads, everything
runs on the CPU in seconds, and the assertions built on them are about faithfulness
(bitwise weights, exact replays, verified IR), never about accuracy.

Hidden size 64 means group size 16 gives four groups per row, enough to tell a
per-group grid from a per-row one; the Llama has two key/value heads for four query
heads so grouped-query attention (the SmolLM2 shape) is exercised too.
"""

from __future__ import annotations

import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    OPTConfig,
    OPTForCausalLM,
    PreTrainedTokenizerFast,
)

#: Vocabulary of the tiny models and the tokenizer; ids 0..2 are the special tokens.
VOCAB = 256
#: Hidden size shared by both tiny models.
HIDDEN = 64
#: Feed-forward width shared by both tiny models.
FFN = 128
#: Decoder blocks per tiny model.
N_BLOCKS = 2
#: Query heads; the Llama uses half as many key/value heads (grouped-query attention).
N_HEADS = 4
N_KV_HEADS = 2
#: Position table length, which also bounds `--dynamic-seqlen` and the stage cache.
MAX_POSITIONS = 64
#: The seed every tiny model is built from, so two calls give identical weights.
MODEL_SEED = 0
#: A group size that divides both widths and gives several groups per row.
GROUP_SIZE = 16


def tiny_opt() -> OPTForCausalLM:
    torch.manual_seed(MODEL_SEED)
    cfg = OPTConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, ffn_dim=FFN, num_hidden_layers=N_BLOCKS,
        num_attention_heads=N_HEADS, max_position_embeddings=MAX_POSITIONS, word_embed_proj_dim=HIDDEN,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    model = OPTForCausalLM(cfg).eval()
    model.config.use_cache = False  # as models/loader.py sets it for evaluation
    return model


def tiny_llama() -> LlamaForCausalLM:
    torch.manual_seed(MODEL_SEED)
    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=FFN, num_hidden_layers=N_BLOCKS,
        num_attention_heads=N_HEADS, num_key_value_heads=N_KV_HEADS, max_position_embeddings=MAX_POSITIONS,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    model = LlamaForCausalLM(cfg).eval()
    model.config.use_cache = False
    return model


TINY_MODELS = {"opt": tiny_opt, "llama": tiny_llama}


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    """A whitespace word-level tokenizer over `w3 .. w255`, so real text can be faked
    with a sentence of such words and every token id is a valid tiny-model input."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2, **{f"w{i}": i for i in range(3, VOCAB)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", bos_token="</s>", eos_token="</s>")
