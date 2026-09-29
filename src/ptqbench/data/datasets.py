"""Evaluation token streams. DESIGN.md 5.2-5.3 and 7.

Each key loads its split, joins it the way the GPTQ reference does, tokenizes once
(default special tokens, so Llama gets exactly one BOS at the start), and keeps
`n_windows * seqlen` tokens. Tokenized streams are cached per tokenizer.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .. import paths

C4_VAL_FILE = "en/c4-validation.00000-of-00008.json.gz"


@dataclass
class TokenStream:
    key: str
    ids: torch.Tensor  # (1, n_windows * seqlen)
    seqlen: int
    split: str
    join: str
    n_source_rows: int

    @property
    def n_windows(self) -> int:
        return self.ids.shape[1] // self.seqlen

    def window(self, i: int) -> torch.Tensor:
        if not 0 <= i < self.n_windows:
            raise IndexError(f"window {i} out of range ({self.n_windows})")
        return self.ids[:, i * self.seqlen : (i + 1) * self.seqlen]

    def first_token_hash(self) -> str:
        """Guards against tokenizer drift: a hash of the first 32 token ids."""
        head = self.ids[0, :32].tolist()
        return hashlib.sha256(json.dumps(head).encode()).hexdigest()[:16]


# ---- raw text loaders: (text or list of docs, split, join label, n_source_rows) ----

def _wikitext2():
    from datasets import load_dataset

    rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join(rows["text"]), "test", '"\\n\\n".join', len(rows)


def _ptb_rows(split: str):
    from datasets import load_dataset

    try:
        return load_dataset("ptb-text-only/ptb_text_only", revision="refs/convert/parquet", split=split)
    except Exception:  # noqa: BLE001 - DESIGN.md 7 fallback: read the parquet file directly
        url = f"hf://datasets/ptb-text-only/ptb_text_only@refs%2Fconvert%2Fparquet/penn_treebank/{split}/0000.parquet"
        return load_dataset("parquet", data_files=url, split="train")


def _ptb():
    rows = _ptb_rows("validation")
    return "\n\n".join(rows["sentence"]), "validation", '"\\n\\n".join', len(rows)


def _ptb_new():
    rows = _ptb_rows("test")
    return " ".join(rows["sentence"]), "test", '" ".join', len(rows)


def _c4_val():
    from datasets import load_dataset

    return load_dataset("allenai/c4", data_files={"validation": C4_VAL_FILE}, split="validation")


def _c4_new():
    rows = _c4_val()
    return " ".join(rows[:1100]["text"]), "validation", '" ".join', len(rows)


_LOADERS: dict[str, Callable] = {
    "wikitext2": _wikitext2,
    "ptb": _ptb,
    "ptb_new": _ptb_new,
    "c4_new": _c4_new,
    "c4": None,  # random-window draw, built separately below
}


def available() -> list[str]:
    return list(_LOADERS)


def _tokenizer_tag(tokenizer) -> str:
    name = getattr(tokenizer, "name_or_path", "") or type(tokenizer).__name__
    return hashlib.sha256(f"{name}|{len(tokenizer)}".encode()).hexdigest()[:12]


def _c4_random(tokenizer, seqlen: int, n: int = 256, seed: int = 0) -> torch.Tensor:
    """GPTQ `get_c4` validation: seeded random docs longer than seqlen, one window each."""
    rows = _c4_val()
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        doc = rows[rng.randint(0, len(rows) - 1)]["text"]
        ids = tokenizer(doc, return_tensors="pt").input_ids
        if ids.shape[1] <= seqlen:
            continue
        s = rng.randint(0, ids.shape[1] - seqlen - 1)
        out.append(ids[:, s : s + seqlen])
    return torch.cat(out, dim=1)


def needs_bos_per_window(tokenizer) -> bool:
    """Gemma degrades badly without <bos> at the start of every sequence (window 2 of
    wikitext2 at seqlen 512: perplexity 11.96 with it, 529 without, gemma-2-2b)."""
    return type(tokenizer).__name__.startswith("Gemma")


def _bos_windows(ids: torch.Tensor, bos: int, seqlen: int) -> torch.Tensor:
    """Re-cut a stream that starts with one BOS into windows that each start with BOS."""
    body = ids[:, 1:] if ids[0, 0].item() == bos else ids
    n = body.shape[1] // (seqlen - 1)
    chunks = body[:, : n * (seqlen - 1)].reshape(n, seqlen - 1)
    return torch.cat([torch.full((n, 1), bos, dtype=ids.dtype), chunks], dim=1).reshape(1, -1)


def build(key: str, tokenizer, seqlen: int = 2048) -> TokenStream:
    if key not in _LOADERS:
        raise KeyError(f"unknown dataset {key!r}; available: {available()}")
    cache = paths.tokenized_cache_dir() / f"{key}-{_tokenizer_tag(tokenizer)}-s{seqlen}.pt"
    if cache.is_file():
        blob = torch.load(cache, weights_only=False)
        return TokenStream(key=key, seqlen=seqlen, **blob)

    if key == "c4":
        ids, split, join, n_rows = _c4_random(tokenizer, seqlen), "validation", "random windows", len(_c4_val())
    else:
        text, split, join, n_rows = _LOADERS[key]()
        ids = tokenizer(text, return_tensors="pt").input_ids
        if key == "c4_new":
            ids = ids[:, : 256 * seqlen]
    if needs_bos_per_window(tokenizer):
        if key == "c4":  # already one window per document slice: overwrite the first token
            ids = ids.reshape(-1, seqlen).clone()
            ids[:, 0] = tokenizer.bos_token_id
            ids = ids.reshape(1, -1)
        else:
            ids = _bos_windows(ids, tokenizer.bos_token_id, seqlen)
        join += " + <bos> per window"
    n_windows = ids.shape[1] // seqlen
    ids = ids[:, : n_windows * seqlen].contiguous()

    blob = {"ids": ids, "split": split, "join": join, "n_source_rows": n_rows}
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".pt.tmp")
    torch.save(blob, tmp)
    tmp.replace(cache)
    return TokenStream(key=key, seqlen=seqlen, **blob)
