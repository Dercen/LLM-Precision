"""Glue for `ptq eval`: load, build the stream, evaluate, write one atomic JSON row."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from .. import device as D
from .. import paths, provenance
from ..data import datasets as ds
from ..models import loader as ml
from . import perplexity as ppl

PROTOCOL_VERSION = 1


def _run_id(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def write_row(row: dict[str, Any]) -> str:
    """Atomic write: tmp file then os.replace, so a kill never leaves a half row."""
    paths.ensure_dirs()
    target = paths.runs_dir() / f"{row['run_id']}.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return str(target)


def run_single_eval(
    *,
    model_id: str,
    dataset_key: str = "wikitext2",
    device_spec: str = "auto",
    seqlen: int = 2048,
    max_windows: int | None = None,
    ce_chunk: int = ppl.DEFAULT_CE_CHUNK,
    dtype_override: str | None = None,
    deterministic: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    paths.ensure_dirs()
    numerics = D.lock_numerics(deterministic=deterministic)
    device = D.resolve(device_spec)

    started = provenance.utc_now()
    loaded = ml.load(model_id, device=device, dtype=ml.parse_dtype(dtype_override))
    stream = ds.build(dataset_key, loaded.tokenizer, seqlen=seqlen)

    result = ppl.evaluate(
        loaded.model,
        stream,
        device=device,
        max_windows=max_windows,
        ce_chunk=ce_chunk,
    )

    quant_key = _run_id(
        {
            "model": model_id,
            "revision": loaded.revision,
            "algo": "fp",
            "bits": 16,
            "dtype": str(loaded.dtype),
        }
    )
    run_id = _run_id(
        {
            "quant_key": quant_key,
            "dataset": dataset_key,
            "seqlen": seqlen,
            "max_windows": max_windows,
            "protocol_version": PROTOCOL_VERSION,
        }
    )

    # PLAN.md 5.8: fp16/bf16-on-GPU rows at the paper config are directly comparable.
    paper_comparable = (
        device.type == "cuda"
        and str(loaded.dtype) in ("torch.float16", "torch.bfloat16")
        and not result.partial
        and seqlen == 2048
    )

    row: dict[str, Any] = {
        "run_id": run_id,
        "quant_key": quant_key,
        "protocol_version": PROTOCOL_VERSION,
        "model": model_id,
        "model_revision": loaded.revision,
        "tokenizer_class": loaded.tokenizer_class,
        "algo": "fp",
        "backend": "torch",
        "bits": 16,
        "group_size": None,
        "dataset": dataset_key,
        "split": stream.split,
        "join": stream.join,
        "n_source_rows": stream.n_source_rows,
        "first_token_hash": stream.first_token_hash(),
        "dtype": str(loaded.dtype),
        "device": str(device),
        "attn_implementation": loaded.attn_implementation,
        "model_bytes": loaded.model_bytes,
        **result.as_row_fields(),
        "paper_comparable": paper_comparable,
        "quant_seconds": 0.0,
        "status": "ok",
        "reason": None,
        "started_at": started,
        "finished_at": provenance.utc_now(),
        **numerics.as_row_fields(),
        **D.describe(device),
        **provenance.block(),
    }

    if write:
        row["_written_to"] = write_row(row)
    return row
