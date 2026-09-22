"""Run one quantization and one evaluation for a RunSpec. The single implementation
behind `ptq eval` and `ptq run`, so ids, row fields and error handling never diverge.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass
from typing import Any

import torch

from .. import config as C
from .. import device as D
from .. import paths, provenance, registry
from ..data import calibration as calib_mod
from ..data import datasets as ds
from ..eval import perplexity as ppl
from ..models import loader as ml
from . import quant_cache


@dataclass
class Prepared:
    """A loaded, quantized model ready to evaluate any number of datasets."""

    loaded: ml.LoadedModel
    device: torch.device
    resident: bool
    quant_fields: dict[str, Any]
    calib_fields: dict[str, Any]
    numerics: D.NumericsState


def resolve_run(run: C.RunSpec, device: torch.device) -> C.RunSpec:
    """Fill in the dtype the device policy picks, so quant_key reflects the real dtype,
    and swap a gated repo for its mirror when the weights cannot be fetched."""
    if run.model.gated and run.model.mirror and not run.model.canonical and not repo_is_fetchable(run.model.repo):
        run = run.model_copy(update={"model": run.model.with_mirror()})
    if run.dtype == "auto":
        return run.model_copy(update={"dtype": str(D.dtype_for(run.model.canonical_repo, device))})
    return run


_FETCHABLE: dict[str, bool] = {}


def repo_is_fetchable(repo: str) -> bool:
    """A real (tiny) download: `model_info` succeeds on gated repos, only files are gated."""
    if repo in _FETCHABLE:
        return _FETCHABLE[repo]
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(repo, "config.json", cache_dir=str(paths.hf_home()))
        _FETCHABLE[repo] = True
    except Exception:  # noqa: BLE001 - gated, missing, offline: all mean "use the mirror"
        _FETCHABLE[repo] = False
    return _FETCHABLE[repo]


def choose_eval_mode(run: C.RunSpec, device: torch.device) -> str:
    """PLAN.md 2a's rule, plus what the quantizer itself needs on the card.

    The eval-only rule said opt-2.7b (4.93 GB) is resident, and it is -- for evaluation.
    GPTQ adds two in_features^2 fp32 Hessian buffers (0.84 GB for its fc2) and AWQ-lite
    its scoring temporaries, which put the same model over the edge: eleven OOMs in the
    first resident_full run, 2026-09-22. Data-dependent methods count those extras.
    """
    if run.eval.eval_mode != "auto":
        return run.eval.eval_mode
    dtype = ml.parse_dtype(_short_dtype(run.dtype)) or torch.float16
    est = ml.estimate_model_bytes(run.model.repo, dtype)
    return "streamed" if D.should_stream(est + quantizer_extra_bytes(run), device) else "resident"


def quantizer_extra_bytes(run: C.RunSpec) -> int:
    """Device memory a quantizer needs beyond the weights, from config.json alone."""
    if not run.quant.is_data_dependent():
        return 0
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(run.model.repo, cache_dir=str(paths.hf_home()), revision=run.model.revision)
    hidden = getattr(cfg, "hidden_size", 0)
    ffn = getattr(cfg, "intermediate_size", None) or getattr(cfg, "ffn_dim", 0)
    widest = max(hidden, ffn)
    if run.quant.algo == "gptq":
        return 2 * 4 * widest**2  # H and its Cholesky temporaries for the widest Linear
    return 4 * 8192 * widest * 4  # awq_lite scoring temporaries (see awq_lite.apply_awq_lite)


def _short_dtype(name: str) -> str:
    return name.replace("torch.", "")


def prepare(
    run: C.RunSpec,
    device: torch.device,
    *,
    use_quant_cache: bool = True,
    cache_min_seconds: float = 60.0,
    deterministic: bool = False,
    eval_mode: str | None = None,
) -> Prepared:
    """Load once and quantize once (or restore from the quant cache)."""
    paths.ensure_dirs()
    numerics = D.lock_numerics(deterministic=deterministic)
    run = resolve_run(run, device)
    mode = eval_mode or choose_eval_mode(run, device)
    resident = mode == "resident"
    dtype = ml.parse_dtype(_short_dtype(run.dtype))
    loaded = ml.load(run.model.repo, device=device, dtype=dtype, to_device=resident, revision=run.model.revision)

    q = run.quant
    quant_fields: dict[str, Any] = {"algo": "fp", "bits": 16, "group_size": None, "sym": None, "quant_seconds": 0.0}
    calib_fields: dict[str, Any] = {"calib": None, "calib_fingerprint": None}
    cache_ok = use_quant_cache and q.is_data_dependent()

    if q.algo != "fp" and cache_ok and quant_cache.has(run.quant_key):
        started = time.perf_counter()
        meta = quant_cache.load_into(loaded.model, run.quant_key)
        quant_cache.touch(run.quant_key)
        quant_fields = {**q.model_dump(), "n_quantized_modules": None, "quant_seconds": round(time.perf_counter() - started, 3), **meta}
        quant_fields["group_size"] = q.group_size
        if run.calib:
            calib_fields = {"calib": run.calib.model_dump(), "calib_fingerprint": None}
        return Prepared(loaded, device, resident, quant_fields, calib_fields, numerics)

    if q.algo == "fp":
        pass
    elif q.algo == "rtn":
        from ..quantizers import rtn as rtn_mod

        quant_fields = rtn_mod.apply_rtn(loaded.model, bits=q.bits, group_size=q.group_size, sym=q.sym).as_row_fields()
    elif q.algo == "hqq":
        from ..quantizers import hqq_adapter

        quant_fields = hqq_adapter.apply_hqq(loaded.model, bits=q.bits, group_size=q.group_size, device=device).as_row_fields()
    elif q.algo in ("gptq", "awq_lite"):
        spec = run.calib or C.CalibSpec()
        windows = calib_mod.build(loaded.tokenizer, calib_mod.CalibSpec(**spec.model_dump()))
        calib_fields = {"calib": spec.model_dump(), "calib_fingerprint": calib_mod.fingerprint(windows)}
        if q.algo == "gptq":
            from ..quantizers import gptq as gptq_mod

            gspec = gptq_mod.GPTQSpec(
                bits=q.bits, group_size=q.group_size, sym=q.sym, act_order=q.act_order,
                true_sequential=q.true_sequential, static_groups=q.static_groups,
                percdamp=q.percdamp, blocksize=q.blocksize,
            )
            quant_fields = gptq_mod.apply_gptq(loaded.model, windows, spec=gspec, device=device, offload=not resident).as_row_fields()
        else:
            from ..quantizers import awq_lite

            aspec = awq_lite.AWQSpec(bits=q.bits, group_size=q.group_size, sym=q.sym, grid=q.awq_grid, clip=q.awq_clip)
            quant_fields = awq_lite.apply_awq_lite(loaded.model, windows, spec=aspec, device=device, offload=not resident).as_row_fields()
    else:
        raise ValueError(f"unknown algo {q.algo!r}")

    if cache_ok and q.algo != "fp" and float(quant_fields.get("quant_seconds", 0)) >= cache_min_seconds:
        quant_cache.save(loaded.model, run.quant_key, {"run": run.model_dump(), "quant_fields": quant_fields, "calib_fields": calib_fields})
        quant_fields["quant_cache_hit"] = False
    resident = _promote_to_resident(loaded, run, device, resident)
    return Prepared(loaded, device, resident, quant_fields, calib_fields, numerics)


def _promote_to_resident(loaded: ml.LoadedModel, run: C.RunSpec, device: torch.device, resident: bool) -> bool:
    """A model streamed only because its quantizer needed the room evaluates resident.

    opt-2.7b quantizes streamed (Hessians) but fits on the card for evaluation; the
    number is identical either way (PLAN.md 5c), evaluation is ~4x faster resident.
    """
    if resident or device.type != "cuda" or run.eval.eval_mode != "auto":
        return resident
    if D.should_stream(loaded.model_bytes, device):
        return False
    loaded.model.to(device)
    torch.cuda.empty_cache()
    return True


def evaluate(prep: Prepared, run: C.RunSpec, *, progress: bool = True) -> dict[str, Any]:
    """One row: build the stream, evaluate, assemble provenance. Does not write."""
    run = resolve_run(run, prep.device)
    started = provenance.utc_now()
    stream = ds.build(run.dataset, prep.loaded.tokenizer, seqlen=run.eval.seqlen)
    if prep.resident:
        result = ppl.evaluate(prep.loaded.model, stream, device=prep.device, max_windows=run.eval.max_windows, ce_chunk=run.eval.ce_chunk, progress=progress)
    else:
        result = ppl.evaluate_streamed(
            prep.loaded.model, stream, device=prep.device, max_windows=run.eval.max_windows,
            ce_chunk=run.eval.ce_chunk, window_batch=run.eval.window_batch, offload=True, progress=progress,
        )
    paper_comparable = (
        prep.device.type == "cuda"
        and str(prep.loaded.dtype) in ("torch.float16", "torch.bfloat16")
        and not result.partial
        and run.eval.seqlen == 2048
        and _calib_is_paper(run)
    )
    return {
        "run_id": run.run_id,
        "quant_key": run.quant_key,
        "protocol_version": C.PROTOCOL_VERSION,
        "model": run.model.canonical_repo,
        "model_key": run.model.key,
        "loaded_from": run.model.repo,
        "model_revision": prep.loaded.revision,
        "tokenizer_class": prep.loaded.tokenizer_class,
        "backend": "torch",
        "dataset": run.dataset,
        "split": stream.split,
        "join": stream.join,
        "n_source_rows": stream.n_source_rows,
        "first_token_hash": stream.first_token_hash(),
        "dtype": str(prep.loaded.dtype),
        "device": str(prep.device),
        "attn_implementation": prep.loaded.attn_implementation,
        "model_bytes": prep.loaded.model_bytes,
        **prep.quant_fields,
        **prep.calib_fields,
        **result.as_row_fields(),
        "paper_comparable": paper_comparable,
        "status": "ok",
        "reason": None,
        "started_at": started,
        "finished_at": provenance.utc_now(),
        **prep.numerics.as_row_fields(),
        **D.describe(prep.device),
        **provenance.block(),
    }


def _calib_is_paper(run: C.RunSpec) -> bool:
    """PLAN.md 5.8: awq_lite is comparable only under the AWQ paper's pile_val 128x512."""
    if run.quant.algo != "awq_lite":
        return True
    c = run.calib
    return bool(c and c.dataset == "pile_val" and c.nsamples == 128 and c.seqlen == 512)


def status_row(run: C.RunSpec, device: torch.device, status: str, reason: str, *, exc: BaseException | None = None) -> dict[str, Any]:
    """A skipped or failed row that still carries every id and provenance field."""
    run = resolve_run(run, device)
    row = {
        "run_id": run.run_id,
        "quant_key": run.quant_key,
        "protocol_version": C.PROTOCOL_VERSION,
        "model": run.model.canonical_repo,
        "model_key": run.model.key,
        "loaded_from": run.model.repo,
        "dataset": run.dataset,
        **run.quant.model_dump(),
        "calib": run.calib.model_dump() if run.calib else None,
        "dtype": run.dtype,
        "device": str(device),
        "ppl": None,
        "partial": False,
        "paper_comparable": False,
        "status": status,
        "reason": reason,
        "finished_at": provenance.utc_now(),
        **provenance.block(),
    }
    if exc is not None:
        log = paths.runs_dir() / f"{run.run_id}.log"
        log.write_text("".join(traceback.format_exception(exc)), encoding="utf-8")
        row["traceback_file"] = str(log)
    return row


def write_row(row: dict[str, Any]) -> str:
    """Atomic: tmp file then os.replace, so a kill never leaves a half row."""
    paths.ensure_dirs()
    target = paths.runs_dir() / f"{row['run_id']}.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return str(target)


def is_done(run: C.RunSpec, *, rerun_incomplete: bool = False) -> bool:
    path = paths.runs_dir() / f"{run.run_id}.json"
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if row.get("protocol_version") != C.PROTOCOL_VERSION:
        return False
    if row.get("status") != "ok":
        return not rerun_incomplete
    return True


def check_available(run: C.RunSpec, device: torch.device) -> tuple[bool, str | None]:
    ok, reason = registry.check(run.quant.algo, device.type, run.quant.bits)
    if not ok:
        return ok, reason
    gs = run.quant.group_size
    if gs not in (-1, None) and run.quant.algo != "fp":
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(run.model.repo, cache_dir=str(paths.hf_home()))
        dims = [getattr(cfg, "hidden_size", 0), getattr(cfg, "intermediate_size", None) or getattr(cfg, "ffn_dim", 0)]
        if any(d and d % gs != 0 for d in dims):
            return False, f"group_size_indivisible:{gs}:{dims}"
    return True, None
