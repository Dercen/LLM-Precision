"""`ptq run`: the matrix loop. DESIGN.md 9.

Expand -> filter -> group by quant_key -> shard -> for each group: skip what is done,
prepare once, evaluate every pending dataset, write each row atomically. Exceptions
become failed rows (with the traceback beside them) and the loop continues; CUDA OOM
walks a retry ladder first. Ctrl-C finishes the current row.
"""

from __future__ import annotations

import json
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import torch

from .. import config as C
from .. import device as D
from .. import paths
from . import execute as X


@dataclass
class Summary:
    n_runs: int = 0
    n_groups: int = 0
    ok: int = 0
    skipped: int = 0
    failed: int = 0
    already_done: int = 0
    seconds: float = 0.0
    rows: list[str] = field(default_factory=list)


def parse_filter(text: str | None) -> dict[str, set[str]]:
    """'algo=gptq,rtn bits=4 model=opt-125m' -> {field: {values}}; commas separate values."""
    out: dict[str, set[str]] = {}
    if not text:
        return out
    for clause in text.replace(";", " ").split():
        if "=" not in clause:
            raise ValueError(f"filter clause {clause!r} is not key=value")
        key, value = clause.split("=", 1)
        out.setdefault(key, set()).update(v for v in value.split(",") if v)
    return out


def _matches(run: C.RunSpec, flt: dict[str, set[str]]) -> bool:
    fields = {
        "model": run.model.key,
        "algo": run.quant.algo,
        "bits": str(run.quant.bits),
        "group_size": str(run.quant.group_size),
        "dataset": run.dataset,
    }
    return all(fields.get(k) in v for k, v in flt.items())


def plan(exp_path: str | Path, *, filter_text: str | None = None, shard: str | None = None, index: int | None = None, device: torch.device | None = None) -> list[tuple[str, list[C.RunSpec]]]:
    """Ordered (quant_key, runs) groups after filter, shard and index selection."""
    exp = C.load_experiment(exp_path)
    dev = device or D.resolve("auto")
    runs = [X.resolve_run(r, dev) for r in C.expand(exp)]
    flt = parse_filter(filter_text)
    runs = [r for r in runs if _matches(r, flt)]
    groups = sorted(C.group_by_quant_key(runs).items(), key=lambda kv: (kv[1][0].model.key, kv[1][0].quant.algo, -kv[1][0].quant.bits, kv[1][0].quant.group_size, kv[0]))
    if shard:
        k, n = (int(x) for x in shard.split("/"))
        groups = [g for i, g in enumerate(groups) if i % n == k]
    if index is not None:
        groups = [groups[index]]
    return groups


def estimate_seconds(groups: list[tuple[str, list[C.RunSpec]]], hostname: str | None = None) -> tuple[float, list[str]]:
    """From results/timing.json; returns (seconds, notes about missing entries)."""
    import socket

    host = hostname or socket.gethostname()
    notes: list[str] = []
    try:
        timing = json.loads(paths.timing_file().read_text(encoding="utf-8")).get(host, {})
    except (OSError, json.JSONDecodeError):
        timing, notes = {}, ["no results/timing.json for this host"]
    entries = timing.get("entries", {})
    quant = timing.get("quantization", {})
    # windows per dataset key, from the measured counts (OPT tokenizer; close enough for estimates)
    windows = {"wikitext2": 140, "c4": 256, "c4_new": 256, "ptb": 45, "ptb_new": 47}

    def spw(run: C.RunSpec) -> float:
        for mode in ("resident", "streamed"):
            key = f"{run.model.repo}|{run.dtype}|{mode}"
            if key in entries:
                return float(entries[key]["seconds_per_window"])
        notes.append(f"no eval timing for {run.model.key}; using 0.5 s/window")
        return 0.5

    def qsec(run: C.RunSpec) -> float:
        if run.quant.algo in ("fp", "rtn", "hqq"):
            return 5.0
        key = f"{run.model.repo}|{run.quant.algo}"
        entry = quant.get(key) or quant.get(f"{run.model.repo}|gptq")
        if entry:
            for k in ("quant_seconds_resident", "quant_seconds_resident_4bit", "quant_seconds"):
                if entry.get(k):
                    return float(entry[k])
        notes.append(f"no quant timing for {run.model.key}/{run.quant.algo}; using 300 s")
        return 300.0

    total = 0.0
    for _, runs in groups:
        pending = [r for r in runs if not X.is_done(r)]
        if not pending:
            continue
        total += qsec(pending[0]) + 15.0  # load + overhead
        for r in pending:
            total += spw(r) * (r.eval.max_windows or windows.get(r.dataset, 150))
    return total, sorted(set(notes))


class _Interrupt:
    """Ctrl-C finishes the current row, a second Ctrl-C aborts."""

    def __init__(self) -> None:
        self.requested = False
        self._prev = None

    def __enter__(self) -> Self:
        self._prev = signal.signal(signal.SIGINT, self._handle)
        return self

    def __exit__(self, *exc: object) -> None:
        signal.signal(signal.SIGINT, self._prev)

    def _handle(self, signum: int, frame: object) -> None:
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        print("\n[ptq] interrupt: finishing the current row, Ctrl-C again to abort", file=sys.stderr)


def run_matrix(
    exp_path: str | Path,
    *,
    filter_text: str | None = None,
    shard: str | None = None,
    index: int | None = None,
    rerun_incomplete: bool = False,
    use_quant_cache: bool = True,
    device_spec: str = "auto",
    deterministic: bool = False,
    progress: bool = True,
    log=print,
) -> Summary:
    device = D.resolve(device_spec)
    groups = plan(exp_path, filter_text=filter_text, shard=shard, index=index, device=device)
    summary = Summary(n_runs=sum(len(r) for _, r in groups), n_groups=len(groups))
    started = time.perf_counter()

    with _Interrupt() as intr:
        for gi, (qkey, runs) in enumerate(groups):
            pending = [r for r in runs if not X.is_done(r, rerun_incomplete=rerun_incomplete)]
            summary.already_done += len(runs) - len(pending)
            if not pending:
                continue
            head = pending[0]
            label = head.label().rsplit("/", 1)[0]
            log(f"[{gi + 1}/{len(groups)}] {label}  ({len(pending)} dataset(s))  quant_key={qkey}")

            ok, reason = X.check_available(head, device)
            if not ok:
                for r in pending:
                    X.write_row(X.status_row(r, device, "skipped", reason or "unavailable"))
                    summary.skipped += 1
                log(f"    skipped: {reason}")
                continue

            prep = None
            try:
                prep = _prepare_with_ladder(head, device, use_quant_cache, deterministic, log)
                for r in pending:
                    try:
                        row = X.evaluate(prep, r, progress=progress)
                        path = X.write_row(row)
                        summary.ok += 1
                        summary.rows.append(path)
                        log(f"    {r.dataset:10s} ppl={row['ppl']:.4f}  {row['eval_seconds']:.1f}s  mode={row['eval_mode']}")
                    except Exception as exc:  # noqa: BLE001 - every failure becomes a row
                        X.write_row(X.status_row(r, device, "failed", f"{type(exc).__name__}: {exc}"[:300], exc=exc))
                        summary.failed += 1
                        log(f"    {r.dataset:10s} FAILED {type(exc).__name__}: {str(exc)[:120]}")
                    if intr.requested:
                        break
            except Exception as exc:  # noqa: BLE001 - quantization failure fails the whole group
                for r in pending:
                    X.write_row(X.status_row(r, device, "failed", f"{type(exc).__name__}: {exc}"[:300], exc=exc))
                    summary.failed += 1
                log(f"    group FAILED {type(exc).__name__}: {str(exc)[:160]}")
            finally:
                if prep is not None:
                    del prep
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            if intr.requested:
                log("[ptq] stopped after the current row")
                break

    summary.seconds = time.perf_counter() - started
    return summary


def _prepare_with_ladder(run: C.RunSpec, device: torch.device, use_cache: bool, deterministic: bool, log) -> X.Prepared:
    """DESIGN.md 9.4: on CUDA OOM, retry once streamed before giving up.

    The retry must first actually release the failed attempt: a caught exception's
    traceback keeps every frame -- and the half-loaded model in its locals -- alive,
    so empty_cache() alone frees nothing and the retry OOMs the same way (all eleven
    opt-2.7b retries did, 2026-09-22). Drop the exception, collect, then empty.
    """
    try:
        return X.prepare(run, device, use_quant_cache=use_cache, deterministic=deterministic)
    except torch.OutOfMemoryError as exc:
        if device.type != "cuda" or X.choose_eval_mode(X.resolve_run(run, device), device) == "streamed":
            raise
        message = str(exc)[:80]
        exc = None  # release the traceback and everything it pins
    _release_device_memory(device)
    log(f"    OOM resident ({message}); retrying streamed with {D.vram_available_bytes(device) / 1024**3:.1f} GB free")
    return X.prepare(run, device, use_quant_cache=use_cache, deterministic=deterministic, eval_mode="streamed")


def _release_device_memory(device: torch.device) -> None:
    import gc

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def list_groups(groups: list[tuple[str, list[C.RunSpec]]], *, rerun_incomplete: bool = False) -> list[dict[str, Any]]:
    out = []
    for i, (qkey, runs) in enumerate(groups):
        done = sum(X.is_done(r, rerun_incomplete=rerun_incomplete) for r in runs)
        out.append({"index": i, "quant_key": qkey, "label": runs[0].label().rsplit("/", 1)[0], "datasets": [r.dataset for r in runs], "done": done, "total": len(runs)})
    return out


def prefetch(exp_path: str | Path, *, log=print) -> None:
    """Download every model's weights and materialise every dataset and calibration set."""
    from huggingface_hub import snapshot_download

    from ..data import calibration as calib_mod
    from ..data import datasets as ds
    from ..models import loader as ml

    exp = C.load_experiment(exp_path)
    for key in exp.models:
        m = C.load_model(key)
        log(f"weights   {m.repo}")
        snapshot_download(
            m.repo, cache_dir=str(paths.hf_home()), revision=m.revision,
            allow_patterns=["*.json", "*.txt", "*.model", "*.bin", "*.safetensors", "*.tiktoken"],
            ignore_patterns=["*.h5", "*.msgpack", "*.ot", "flax*", "tf_*"],
        )
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(m.repo, cache_dir=str(paths.hf_home()), revision=m.revision)
        for dkey in exp.datasets:
            s = ds.build(dkey, tok, seqlen=exp.eval.seqlen)
            log(f"dataset   {dkey:10s} {s.n_windows} windows for {key}")
        if any(e.algo in C.DATA_DEPENDENT for e in exp.grid):
            w = calib_mod.build(tok, calib_mod.CalibSpec(**exp.calib.model_dump()))
            log(f"calib     {exp.calib.dataset} {tuple(w.shape)} for {key}")
    _ = ml  # loader is exercised by the matrix itself
