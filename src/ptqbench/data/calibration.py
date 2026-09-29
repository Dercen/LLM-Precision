"""Calibration windows for GPTQ and AWQ-lite. DESIGN.md 7.

c4 (default): the GPTQ reference loop over the first C4 train shard -- seeded random
document, redrawn until longer than seqlen, one random window. wikitext2: the same
loop over one long stream of the train split (the small smoke-test option). pile_val:
AWQ's `get_calib_dataset` -- shuffle, keep short documents, concatenate, cut into
seqlen blocks. Built windows are cached per tokenizer and spec.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass

import torch

from .. import paths

C4_TRAIN_FILE = "en/c4-train.00000-of-01024.json.gz"


@dataclass
class CalibSpec:
    dataset: str = "c4"
    nsamples: int = 128
    seqlen: int = 2048
    seed: int = 0


def _tokenizer_hash(tokenizer) -> str:
    name = getattr(tokenizer, "name_or_path", "") or type(tokenizer).__name__
    return hashlib.sha256(f"{name}|{len(tokenizer)}".encode()).hexdigest()[:12]


def _c4(tokenizer, spec: CalibSpec) -> torch.Tensor:
    from datasets import load_dataset

    rows = load_dataset("allenai/c4", data_files={"train": C4_TRAIN_FILE}, split="train")
    random.seed(spec.seed)
    out = []
    while len(out) < spec.nsamples:
        i = random.randint(0, len(rows) - 1)
        ids = tokenizer(rows[i]["text"], return_tensors="pt").input_ids
        if ids.shape[1] <= spec.seqlen:
            continue
        s = random.randint(0, ids.shape[1] - spec.seqlen - 1)
        out.append(ids[:, s : s + spec.seqlen])
    return torch.cat(out, dim=0)


def _wikitext2(tokenizer, spec: CalibSpec) -> torch.Tensor:
    from datasets import load_dataset

    rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    ids = tokenizer("\n\n".join(rows["text"]), return_tensors="pt").input_ids
    random.seed(spec.seed)
    out = []
    for _ in range(spec.nsamples):
        s = random.randint(0, ids.shape[1] - spec.seqlen - 1)
        out.append(ids[:, s : s + spec.seqlen])
    return torch.cat(out, dim=0)


def _pile_val_path():
    """datasets cannot read the .zst directly (DESIGN.md 7); decompress it once."""
    out = paths.dataset_cache_dir() / "pile_val.jsonl"
    if out.is_file():
        return out
    import zstandard
    from huggingface_hub import hf_hub_download

    src = hf_hub_download("mit-han-lab/pile-val-backup", "val.jsonl.zst", repo_type="dataset",
                          cache_dir=str(paths.hf_home()))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jsonl.tmp")
    with open(src, "rb") as fin, open(tmp, "wb") as fout:
        zstandard.ZstdDecompressor().copy_stream(fin, fout)
    tmp.replace(out)
    return out


def _pile_val(tokenizer, spec: CalibSpec) -> torch.Tensor:
    from datasets import load_dataset

    rows = load_dataset("json", data_files=str(_pile_val_path()), split="train").shuffle(seed=42 + spec.seed)
    samples, n_run = [], 0
    for row in rows:
        enc = tokenizer.encode(row["text"].strip())
        if not enc or len(enc) > spec.seqlen:
            continue
        samples.append(torch.tensor([enc]))
        n_run += 1
        if n_run == spec.nsamples:
            break
    cat = torch.cat(samples, dim=1)
    n_split = cat.shape[1] // spec.seqlen
    return torch.cat([cat[:, i * spec.seqlen : (i + 1) * spec.seqlen] for i in range(n_split)], dim=0)


_BUILDERS = {"c4": _c4, "wikitext2": _wikitext2, "pile_val": _pile_val}


def build(tokenizer, spec: CalibSpec) -> torch.Tensor:
    """(nsamples, seqlen) token ids; pile_val may return fewer, as AWQ's loader does."""
    if spec.dataset not in _BUILDERS:
        raise KeyError(f"unknown calibration set {spec.dataset!r}; available: {sorted(_BUILDERS)}")
    name = f"{spec.dataset}-{_tokenizer_hash(tokenizer)}-n{spec.nsamples}-s{spec.seqlen}-seed{spec.seed}.pt"
    cache = paths.calib_cache_dir() / name
    if cache.is_file():
        return torch.load(cache)
    windows = _BUILDERS[spec.dataset](tokenizer, spec).contiguous()
    from .datasets import needs_bos_per_window

    if needs_bos_per_window(tokenizer):  # same reason as the eval windows (datasets.py)
        windows[:, 0] = tokenizer.bos_token_id
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".pt.tmp")
    torch.save(windows, tmp)
    tmp.replace(cache)
    return windows


def fingerprint(windows: torch.Tensor) -> str:
    """Identifies the exact calibration tokens a quantization saw."""
    h = hashlib.sha256(json.dumps(list(windows.shape)).encode())
    h.update(windows.to(torch.int64).cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def describe(spec: CalibSpec) -> dict:
    return asdict(spec)
