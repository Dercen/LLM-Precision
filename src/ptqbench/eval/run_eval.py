"""Glue for `ptq eval`: load, build the stream, evaluate, write one atomic JSON row."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from .. import device as D
from .. import paths, provenance
from ..data import calibration as calib_mod
from ..data import datasets as ds
from ..models import loader as ml
from ..quantizers import gptq as gptq_mod
from ..quantizers import rtn as rtn_mod
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
    algo: str = "fp",
    bits: int = 16,
    group_size: int = -1,
    sym: bool = False,
    act_order: bool = False,
    true_sequential: bool = False,
    percdamp: float = 0.01,
    calib: calib_mod.CalibSpec | None = None,
    eval_mode: str = "auto",
    window_batch: int = 32,
    write: bool = True,
) -> dict[str, Any]:
    paths.ensure_dirs()
    numerics = D.lock_numerics(deterministic=deterministic)
    device = D.resolve(device_spec)

    started = provenance.utc_now()
    # Resident: load straight onto the device. Streamed: keep the weights in host
    # RAM and let eval/streaming.py move one block at a time (PLAN.md 2a, 5c).
    model_dtype = ml.parse_dtype(dtype_override) or D.dtype_for(model_id, device)
    if eval_mode == "auto":
        est = ml.estimate_model_bytes(model_id, model_dtype)
        eval_mode = "streamed" if D.should_stream(est, device) else "resident"
    if eval_mode not in ("resident", "streamed"):
        raise ValueError(f"eval_mode must be auto|resident|streamed, got {eval_mode!r}")
    resident = eval_mode == "resident"
    loaded = ml.load(model_id, device=device, dtype=model_dtype, to_device=resident)

    quant_fields: dict[str, Any] = {
        "algo": "fp",
        "bits": 16,
        "group_size": None,
        "sym": None,
        "quant_seconds": 0.0,
    }
    calib_fields: dict[str, Any] = {"calib": None, "calib_fingerprint": None}

    if algo == "fp":
        pass
    elif algo == "rtn":
        report = rtn_mod.apply_rtn(loaded.model, bits=bits, group_size=group_size, sym=sym)
        quant_fields = report.as_row_fields()
    elif algo == "gptq":
        spec = calib or calib_mod.CalibSpec()
        windows = calib_mod.build(loaded.tokenizer, spec)
        calib_fields = {**spec.as_row_fields(), "calib_fingerprint": calib_mod.fingerprint(windows)}
        gspec = gptq_mod.GPTQSpec(
            bits=bits, group_size=group_size, sym=sym, act_order=act_order,
            true_sequential=true_sequential, percdamp=percdamp,
        )
        report = gptq_mod.apply_gptq(
            loaded.model, windows, spec=gspec, device=device, offload=not resident
        )
        quant_fields = report.as_row_fields()
    else:
        raise ValueError(f"unknown algo {algo!r}; available: fp, rtn, gptq")

    stream = ds.build(dataset_key, loaded.tokenizer, seqlen=seqlen)

    if resident:
        result = ppl.evaluate(
            loaded.model, stream, device=device, max_windows=max_windows, ce_chunk=ce_chunk
        )
    else:
        result = ppl.evaluate_streamed(
            loaded.model, stream, device=device, max_windows=max_windows,
            ce_chunk=ce_chunk, window_batch=window_batch, offload=True,
        )

    quant_key = _run_id(
        {
            "model": model_id,
            "revision": loaded.revision,
            "algo": algo,
            "bits": quant_fields["bits"],
            "group_size": quant_fields["group_size"],
            "sym": quant_fields["sym"],
            "act_order": quant_fields.get("act_order"),
            "true_sequential": quant_fields.get("true_sequential"),
            "percdamp": quant_fields.get("percdamp"),
            "calib": calib_fields["calib"],
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
        "backend": "torch",
        "dataset": dataset_key,
        "split": stream.split,
        "join": stream.join,
        "n_source_rows": stream.n_source_rows,
        "first_token_hash": stream.first_token_hash(),
        "dtype": str(loaded.dtype),
        "device": str(device),
        "attn_implementation": loaded.attn_implementation,
        "model_bytes": loaded.model_bytes,
        **quant_fields,
        **calib_fields,
        **result.as_row_fields(),
        "paper_comparable": paper_comparable,
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
