"""`ptq eval` for one model/dataset pair: a thin wrapper over runner.execute so the
CLI single-shot path and the matrix share one id scheme, one row schema and one
error path. Kept for the tests and scripts that call `run_single_eval`."""

from __future__ import annotations

from typing import Any

from .. import config as C
from .. import device as D
from ..runner import execute as X


def run_single_eval(
    *,
    model_id: str,
    dataset_key: str = "wikitext2",
    device_spec: str = "auto",
    seqlen: int = 2048,
    max_windows: int | None = None,
    ce_chunk: int = 256,
    dtype_override: str | None = None,
    deterministic: bool = False,
    algo: str = "fp",
    bits: int = 16,
    group_size: int = -1,
    sym: bool = False,
    act_order: bool = False,
    true_sequential: bool = False,
    percdamp: float = 0.01,
    calib: Any = None,
    eval_mode: str = "auto",
    window_batch: int = 32,
    model_key: str | None = None,
    write: bool = True,
) -> dict[str, Any]:
    device = D.resolve(device_spec)
    model = _model_spec(model_id, model_key)
    quant = C.QuantSpec(
        algo=algo, bits=16 if algo == "fp" else bits, group_size=group_size, sym=sym,
        act_order=act_order, true_sequential=true_sequential, percdamp=percdamp,
    )
    calib_spec = None
    if quant.is_data_dependent():
        calib_spec = C.CalibSpec(**(calib.__dict__ if calib is not None and not isinstance(calib, dict) else (calib or {})))
    run = C.RunSpec(
        model=model, quant=quant, calib=calib_spec, dataset=dataset_key,
        eval=C.EvalSpec(seqlen=seqlen, max_windows=max_windows, ce_chunk=ce_chunk, eval_mode=eval_mode, window_batch=window_batch),
        dtype=dtype_override or "auto",
    )
    prep = X.prepare(run, device, deterministic=deterministic)
    row = X.evaluate(prep, run)
    if write:
        row["_written_to"] = X.write_row(row)
    return row


def _model_spec(model_id: str, model_key: str | None) -> C.ModelSpec:
    """Prefer the repo's configs/models entry (pinned revision); else an ad-hoc spec."""
    from .. import paths

    key = model_key or _guess_key(model_id)
    if key and (paths.configs_dir() / "models" / f"{key}.yaml").is_file():
        spec = C.load_model(key)
        if spec.repo == model_id:
            return spec
    return C.ModelSpec(key=key or model_id.split("/")[-1].lower(), repo=model_id, tokenizer_class="unknown")


def _guess_key(model_id: str) -> str:
    return model_id.split("/")[-1].lower()
