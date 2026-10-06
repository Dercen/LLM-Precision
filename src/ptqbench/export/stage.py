"""The export stage: load, quantize with a Recorder, swap in QuantizedLinears, score,
write `layers.json`, `export.json` and `model.mlir`. Behind `ptq export-mlir`.

The quantization is the pipeline's own (`rtn`, `gptq`, `awq_lite`, `hqq` through the
same driver functions `runner/execute.py` calls, with the same RunSpec ids), run once
more with `record=` so the grids survive. The quantized-weight cache is not used: it
holds dequantized weights only.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .. import config as C
from .. import device as D
from .. import paths, provenance
from ..data import calibration as calib_mod
from ..models import families
from ..models import loader as ml
from ..quantizers import fakequant as fq
from ..runner import execute as X
from . import mlir as mlir_mod
from .quantized import QuantizedLinear, swap_in_quantized

FORMAT_VERSION = 1


@dataclass
class ExportOptions:
    representation: str = "decomposed"
    seqlen: int = 128  # example input length for torch.export (static shape unless dynamic_seqlen)
    dynamic_seqlen: bool = False
    write_mlir: bool = True
    bytecode: bool = False
    #: Windows of `score_dataset` on which each layer's output is compared with the
    #: output its pre-rounding weight gives on the same input; 0 disables scoring.
    score_windows: int = 0
    score_dataset: str = "wikitext2"
    score_seqlen: int = 512
    score_seed: int = 0
    eval_mode: str = "auto"  # resident | streamed decision for the quantizer, as in `ptq eval`
    deterministic: bool = False


@dataclass
class ExportResult:
    model: nn.Module
    records: dict[str, fq.QuantRecord]
    modules: dict[str, QuantizedLinear]
    layers: dict[str, dict[str, Any]]
    manifest: dict[str, Any]
    out_dir: Path
    layers_path: Path
    manifest_path: Path
    mlir_path: Path | None = None
    notes: list[str] = field(default_factory=list)


def resolve_model(arg: str) -> C.ModelSpec:
    """A configs/models key, a Hugging Face repo id, or a local directory."""
    if (paths.configs_dir() / "models" / f"{arg}.yaml").is_file():
        return C.load_model(arg)
    from ..eval.run_eval import _model_spec

    return _model_spec(arg, None)


def build_run(model: C.ModelSpec, quant: C.QuantSpec, calib: C.CalibSpec | None, *, dtype: str | None = None) -> C.RunSpec:
    """The RunSpec whose quant_key names these quantized weights (dataset is irrelevant)."""
    calib_spec = calib if quant.is_data_dependent() else None
    if quant.is_data_dependent() and calib_spec is None:
        calib_spec = C.CalibSpec()
    return C.RunSpec(model=model, quant=quant, calib=calib_spec, dataset="wikitext2", dtype=dtype or model.dtype)


@torch.no_grad()
def export_quantized(
    run: C.RunSpec,
    out_dir: str | Path,
    *,
    device: torch.device,
    options: ExportOptions | None = None,
) -> ExportResult:
    options = options or ExportOptions()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths.ensure_dirs()
    numerics = D.lock_numerics(deterministic=options.deterministic)
    notes: list[str] = []
    started = time.perf_counter()

    run = X.resolve_run(run, device)
    mode = options.eval_mode if options.eval_mode != "auto" else X.choose_eval_mode(run, device)
    resident = mode == "resident"
    dtype = ml.parse_dtype(run.dtype.replace("torch.", ""))
    loaded = ml.load(run.model.repo, device=device, dtype=dtype, to_device=resident, revision=run.model.revision)
    model = loaded.model
    fam = families.for_model(model)
    targets = families.target_modules(model, fam)

    keep_reference = options.score_windows > 0 and run.quant.algo != "fp"
    if keep_reference:
        need = int(1.5 * sum(m.weight.numel() * m.weight.element_size() for m in targets.values()))
        if not _host_ram_allows(need):
            keep_reference = False
            notes.append(f"layer-output scoring skipped: keeping {need / 1024**3:.1f} GB of reference weights exceeds free host RAM")
    recorder = fq.Recorder(keep_reference=keep_reference)

    quant_fields, calib_fields = _quantize(loaded, run, device, resident, recorder)
    evaluated = {name: mod.weight.detach() for name, mod in targets.items() if name in recorder.records}
    modules = swap_in_quantized(model, recorder.records, representation=options.representation, family=fam)

    layers = _layer_metrics(modules, recorder.records, evaluated, quant_fields)
    del evaluated
    if keep_reference and modules:
        _score_layer_outputs(loaded, modules, recorder.records, layers, options, device if resident else torch.device("cpu"))
    for rec in recorder.records.values():
        rec.reference = None
    if any(m.scheme == "per_group_permuted" for m in modules.values()):
        notes.append("act_order groups are not contiguous: those layers use explicit ATen dequantization (index_select/sub/mul)")
    if any(m.zero_point_domain == "float" for m in modules.values()):
        notes.append("hqq zero-points are floats: those layers use explicit ATen dequantization")

    mlir_path = None
    mlir_stats: dict[str, Any] = {}
    if options.write_mlir:
        model.to("cpu")
        example = _example_input_ids(loaded, options.seqlen)
        max_seqlen = getattr(model.config, "max_position_embeddings", None)
        module, mlir_stats = mlir_mod.export_torch_dialect(
            model, example, dynamic_seqlen=options.dynamic_seqlen, max_seqlen=max_seqlen
        )
        mlir_path = mlir_mod.write_module(module, out_dir / "model.mlir")
        if options.bytecode:
            mlir_mod.write_module(module, out_dir / "model.mlirbc", bytecode=True)
        mlir_stats["ops"] = _count_ops(module)
        mlir_stats["mlir_bytes"] = mlir_path.stat().st_size
        del module

    manifest = _manifest(run, loaded, quant_fields, calib_fields, modules, layers, options, mlir_path, mlir_stats, numerics, notes, time.perf_counter() - started)
    layers_path = _write_json(out_dir / "layers.json", layers)
    manifest_path = _write_json(out_dir / "export.json", manifest)
    return ExportResult(
        model=model, records=recorder.records, modules=modules, layers=layers, manifest=manifest,
        out_dir=out_dir, layers_path=layers_path, manifest_path=manifest_path, mlir_path=mlir_path, notes=notes,
    )


# ---- quantization dispatch (mirrors runner/execute.prepare, plus record=) --------------


def _quantize(loaded: ml.LoadedModel, run: C.RunSpec, device: torch.device, resident: bool, recorder: fq.Recorder):
    q = run.quant
    quant_fields: dict[str, Any] = {"algo": "fp", "bits": 16, "group_size": None, "sym": None, "quant_seconds": 0.0}
    calib_fields: dict[str, Any] = {"calib": None, "calib_fingerprint": None}
    if q.algo == "fp":
        return quant_fields, calib_fields
    if q.algo == "rtn":
        from ..quantizers import rtn as rtn_mod

        quant_fields = rtn_mod.apply_rtn(loaded.model, bits=q.bits, group_size=q.group_size, sym=q.sym, record=recorder).as_row_fields()
    elif q.algo == "hqq":
        from ..quantizers import hqq_adapter

        quant_fields = hqq_adapter.apply_hqq(loaded.model, bits=q.bits, group_size=q.group_size, device=device, record=recorder).as_row_fields()
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
            quant_fields = gptq_mod.apply_gptq(loaded.model, windows, spec=gspec, device=device, offload=not resident, record=recorder).as_row_fields()
        else:
            from ..quantizers import awq_lite

            aspec = awq_lite.AWQSpec(bits=q.bits, group_size=q.group_size, sym=q.sym, grid=q.awq_grid, clip=q.awq_clip)
            quant_fields = awq_lite.apply_awq_lite(loaded.model, windows, spec=aspec, device=device, offload=not resident, record=recorder).as_row_fields()
    else:
        raise ValueError(f"unknown algo {q.algo!r}")
    return quant_fields, calib_fields


# ---- per-layer accuracy ---------------------------------------------------------------


def _layer_metrics(
    modules: dict[str, QuantizedLinear],
    records: dict[str, fq.QuantRecord],
    evaluated: dict[str, torch.Tensor],
    quant_fields: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Weight-space accuracy per module, plus proof the export reproduces the evaluated weight."""
    per_block_loss = quant_fields.get("gptq_per_block_loss")
    layers: dict[str, dict[str, Any]] = {}
    for name, mod in modules.items():
        rec = records[name]
        w_eval = evaluated[name]
        w_export = mod.dequantize()
        diff = (w_export.float() - w_eval.float()).abs()
        signal = float(w_eval.float().pow(2).mean())
        snr = 10 * math.log10(signal / rec.mse) if rec.mse > 0 and signal > 0 else None
        entry: dict[str, Any] = {
            **mod.describe(),
            "rel_error": rec.rel_error,
            "max_abs_error": rec.max_abs_error,
            "mse": rec.mse,
            "snr_db": snr,
            "scale_min": float(rec.scale.min()),
            "scale_max": float(rec.scale.max()),
            "zero_min": float(rec.zero.min()),
            "zero_max": float(rec.zero.max()),
            "codes_used": int(torch.unique(rec.codes).numel()),
            "roundtrip_max_abs_diff": float(diff.max()) if diff.numel() else 0.0,
            "bitwise_equal_to_evaluated": bool(torch.equal(w_export, w_eval)),
        }
        if rec.algo == "awq_lite":
            entry["error_reference"] = "awq_scaled_clipped_weight"
        if per_block_loss is not None:
            m = re.search(r"\.(\d+)\.", name)
            if m and int(m.group(1)) < len(per_block_loss):
                entry["gptq_block_loss"] = per_block_loss[int(m.group(1))]
        layers[name] = entry
    return layers


def _score_layer_outputs(
    loaded: ml.LoadedModel,
    modules: dict[str, QuantizedLinear],
    records: dict[str, fq.QuantRecord],
    layers: dict[str, dict[str, Any]],
    options: ExportOptions,
    device: torch.device,
) -> None:
    """`output_rel_error`: ||Q(W) x - W_ref x|| / ||W_ref x|| per layer over the score
    windows, with x the layer's input in the quantized model and W_ref the weight the
    quantizer rounded. This is the per-layer objective GPTQ and AWQ minimise."""
    spec = calib_mod.CalibSpec(options.score_dataset, options.score_windows, options.score_seqlen, options.score_seed)
    windows = calib_mod.build(loaded.tokenizer, spec)
    sums = {name: [0.0, 0.0] for name in modules}
    handles = []
    for name, mod in modules.items():
        ref = records[name].reference

        def hook(module, inputs, output, *, name=name, ref=ref):
            x = inputs[0]
            w_ref = ref.to(device=x.device, dtype=x.dtype)
            y_ref = F.linear(x, w_ref, module.bias)
            sums[name][0] += float((output.float() - y_ref.float()).pow(2).sum())
            sums[name][1] += float(y_ref.float().pow(2).sum())

        handles.append(mod.register_forward_hook(hook))
    model = loaded.model
    model.to(device)
    try:
        for i in range(windows.shape[0]):
            model(input_ids=windows[i : i + 1].to(device))
    finally:
        for h in handles:
            h.remove()
    for name, (num, den) in sums.items():
        layers[name]["output_rel_error"] = math.sqrt(num / den) if den > 0 else None
        layers[name]["output_score"] = {"dataset": spec.dataset, "windows": int(windows.shape[0]), "seqlen": int(windows.shape[1])}


def _host_ram_allows(need_bytes: int) -> bool:
    try:
        import psutil

        return psutil.virtual_memory().available > need_bytes
    except Exception:  # noqa: BLE001 - no psutil reading means no guard, not a failure
        return True


# ---- outputs ---------------------------------------------------------------------------


def _example_input_ids(loaded: ml.LoadedModel, seqlen: int) -> torch.Tensor:
    bos = getattr(loaded.tokenizer, "bos_token_id", None)
    fill = int(bos) if bos is not None else 0
    return torch.full((1, seqlen), fill, dtype=torch.int64)


def _count_ops(module: Any) -> dict[str, int]:
    counts: dict[str, int] = {}

    def walk(op):
        counts[op.name] = counts.get(op.name, 0) + 1
        for region in op.regions:
            for block in region.blocks:
                for inner in block.operations:
                    walk(inner.operation)

    walk(module.operation)
    return {k: v for k, v in sorted(counts.items()) if k.startswith(("torch.quantized_decomposed", "torch.operator", "torch.aten.index_select", "torch.aten.mm", "torch.aten.linear"))}


def _matching_rows(quant_key: str) -> list[dict[str, Any]]:
    """Perplexity rows the benchmark already holds for exactly these quantized weights."""
    rows = []
    for p in sorted(paths.runs_dir().glob("*.json")):
        try:
            row = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("quant_key") == quant_key and row.get("status") == "ok":
            rows.append({k: row.get(k) for k in ("run_id", "dataset", "ppl", "partial", "paper_comparable", "seqlen", "n_windows")})
    return rows


def _manifest(run, loaded, quant_fields, calib_fields, modules, layers, options, mlir_path, mlir_stats, numerics, notes, seconds) -> dict[str, Any]:
    rel = [v["rel_error"] for v in layers.values()]
    out_rel = [v["output_rel_error"] for v in layers.values() if v.get("output_rel_error") is not None]
    return {
        "format_version": FORMAT_VERSION,
        "model": run.model.canonical_repo,
        "model_key": run.model.key,
        "loaded_from": run.model.repo,
        "model_revision": loaded.revision,
        "model_type": getattr(loaded.model.config, "model_type", None),
        "dtype": str(loaded.dtype),
        "quant_key": run.quant_key,
        "quant": run.quant.model_dump(),
        **{k: v for k, v in quant_fields.items() if k not in ("gptq_per_block_loss",)},
        **calib_fields,
        "n_quantized_modules": len(modules),
        "representation": options.representation,
        "dequant_ops_used": sorted({m.dequant_op for m in modules.values()}),
        "schemes_used": sorted({m.scheme for m in modules.values()}),
        "files": {
            "layers": "layers.json",
            "mlir": mlir_path.name if mlir_path else None,
            "mlir_bytecode": "model.mlirbc" if (mlir_path and options.bytecode) else None,
        },
        "mlir": {"example_seqlen": options.seqlen, **mlir_stats} if mlir_path else None,
        "summary": {
            "mean_rel_error": sum(rel) / len(rel) if rel else None,
            "max_rel_error": max(rel) if rel else None,
            "mean_output_rel_error": sum(out_rel) / len(out_rel) if out_rel else None,
            "all_bitwise_equal_to_evaluated": all(v["bitwise_equal_to_evaluated"] for v in layers.values()) if layers else None,
        },
        "perplexity_rows": _matching_rows(run.quant_key),
        "export_seconds": round(seconds, 3),
        "notes": notes,
        **numerics.as_row_fields(),
        **provenance.block(),
    }


def _write_json(path: Path, payload: Any) -> Path:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return path
