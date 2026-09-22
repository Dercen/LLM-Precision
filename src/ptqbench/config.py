"""Experiment configuration. DESIGN.md 9.

A YAML experiment names models (keys into configs/models/), datasets, a calibration
spec, evaluation settings and a grid of quantization cells. `expand()` turns it into
the flat list of RunSpecs the runner executes, one per (model, quant config, dataset).
`quant_key` identifies "these quantized weights" and `run_id` "this row"; both exclude
hostname, device and timestamps so ids are stable across machines and reruns, and
both include dtype so a laptop fp16 row and a hypothetical fp32 row never collide.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from . import paths

PROTOCOL_VERSION = 1

Algo = Literal["fp", "rtn", "gptq", "awq_lite", "hqq"]
DATA_DEPENDENT: frozenset[str] = frozenset({"gptq", "awq_lite"})
DATA_FREE: frozenset[str] = frozenset({"fp", "rtn", "hqq"})


class CalibSpec(BaseModel):
    dataset: str = "c4"
    nsamples: int = 128
    seqlen: int = 2048
    seed: int = 0


class QuantSpec(BaseModel):
    algo: Algo = "fp"
    bits: int = 16
    group_size: int = -1
    sym: bool = False
    act_order: bool = False
    true_sequential: bool = False
    static_groups: bool = False
    percdamp: float = 0.01
    blocksize: int = 128
    awq_grid: int = 20
    awq_clip: bool = True

    @model_validator(mode="after")
    def _consistent(self) -> QuantSpec:
        if self.algo == "fp":
            if self.bits != 16:
                raise ValueError("fp rows are always 16 bits")
            object.__setattr__(self, "group_size", -1)
        elif not 2 <= self.bits <= 8:
            raise ValueError(f"{self.algo} needs 2..8 bits, got {self.bits}")
        if self.group_size not in (-1,) and self.group_size <= 0:
            raise ValueError("group_size must be -1 or positive")
        return self

    def is_data_dependent(self) -> bool:
        return self.algo in DATA_DEPENDENT

    def hash_fields(self) -> dict[str, Any]:
        """The fields that define the quantized weights (not the row)."""
        base: dict[str, Any] = {"algo": self.algo, "bits": self.bits}
        if self.algo == "fp":
            return base
        base.update({"group_size": self.group_size, "sym": self.sym})
        if self.algo == "gptq":
            base.update(
                {
                    "act_order": self.act_order,
                    "true_sequential": self.true_sequential,
                    "static_groups": self.static_groups,
                    "percdamp": self.percdamp,
                    "blocksize": self.blocksize,
                }
            )
        if self.algo == "awq_lite":
            base.update({"awq_grid": self.awq_grid, "awq_clip": self.awq_clip})
        return base


class EvalSpec(BaseModel):
    seqlen: int = 2048
    max_windows: int | None = None
    ce_chunk: int = 256
    eval_mode: Literal["auto", "resident", "streamed"] = "auto"
    window_batch: int = 32

    def hash_fields(self) -> dict[str, Any]:
        # ce_chunk, eval_mode and window_batch change nothing about the number.
        return {"seqlen": self.seqlen, "max_windows": self.max_windows}


class ModelSpec(BaseModel):
    key: str
    repo: str
    mirror: str | None = None
    mirror_revision: str | None = None
    gated: bool = False
    tokenizer_class: str
    dtype: str = "auto"
    revision: str | None = None
    family: str | None = None
    #: The repo the row is filed under (literature join); `repo` is what gets loaded.
    #: Differs only after `with_mirror()` substituted an ungated mirror for a gated repo.
    canonical: str | None = None
    #: Extra group sizes this model gets on top of the grid's (SmolLM2: 576 = 9 x 64).
    extra_group_sizes: list[int] = Field(default_factory=list)
    notes: str | None = None

    @property
    def load_id(self) -> str:
        return self.repo

    @property
    def canonical_repo(self) -> str:
        return self.canonical or self.repo

    def with_mirror(self) -> ModelSpec:
        """This spec pointed at its mirror, keeping the canonical id for the join."""
        if not self.mirror:
            raise ValueError(f"{self.key} has no mirror configured")
        return self.model_copy(update={
            "repo": self.mirror, "revision": self.mirror_revision, "canonical": self.canonical_repo,
        })


class RunSpec(BaseModel):
    model: ModelSpec
    quant: QuantSpec
    calib: CalibSpec | None = None
    dataset: str
    eval: EvalSpec = Field(default_factory=EvalSpec)
    dtype: str = "auto"  # resolved at run time from the device policy

    @property
    def quant_key(self) -> str:
        # Hashes the repo actually loaded: mirror weights are different weights until
        # their SHA-equivalence to the official repo is verified (DESIGN.md 8).
        payload = {
            "model": self.model.repo,
            "revision": self.model.revision or "unpinned",
            "dtype": self.dtype,
            "quant": self.quant.hash_fields(),
            "calib": self.calib.model_dump() if self.calib else None,
        }
        return _short_hash(payload)

    @property
    def run_id(self) -> str:
        payload = {
            "quant_key": self.quant_key,
            "dataset": self.dataset,
            "eval": self.eval.hash_fields(),
            "protocol_version": PROTOCOL_VERSION,
        }
        return _short_hash(payload)

    def label(self) -> str:
        q = self.quant
        tag = q.algo if q.algo == "fp" else f"{q.algo}{q.bits}g{q.group_size}"
        if q.algo == "gptq" and q.act_order:
            tag += "-ao"
        return f"{self.model.key}/{tag}/{self.dataset}"


class GridCell(BaseModel):
    bits: list[int]
    group_size: list[int] = Field(default_factory=lambda: [-1])


class GridEntry(BaseModel):
    algo: Algo
    cells: list[GridCell] = Field(default_factory=list)
    #: QuantSpec overrides applied to every cell of this entry.
    options: dict[str, Any] = Field(default_factory=dict)


class ExperimentSpec(BaseModel):
    name: str
    extends: str | None = None
    models: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    calib: CalibSpec = Field(default_factory=CalibSpec)
    eval: EvalSpec = Field(default_factory=EvalSpec)
    grid: list[GridEntry] = Field(default_factory=list)
    #: QuantSpec overrides per family, e.g. {llama: {act_order: true}} (DESIGN.md 6).
    family_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)


def _short_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_experiment(path: str | Path) -> ExperimentSpec:
    """Load a YAML experiment, resolving `extends` relative to the file."""
    path = Path(path)
    raw = _load_yaml(path)
    chain = [raw]
    seen = {path.resolve()}
    while chain[-1].get("extends"):
        parent = (path.parent / chain[-1]["extends"]).resolve()
        if parent in seen:
            raise ValueError(f"circular extends at {parent}")
        seen.add(parent)
        chain.append(_load_yaml(parent))
    merged: dict[str, Any] = {}
    for layer in reversed(chain):
        merged = _deep_merge(merged, {k: v for k, v in layer.items() if k != "extends"})
    merged.setdefault("name", path.stem)
    return ExperimentSpec.model_validate(merged)


def load_model(key: str) -> ModelSpec:
    path = paths.configs_dir() / "models" / f"{key}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no model config {path}")
    raw = _load_yaml(path)
    raw.setdefault("key", key)
    return ModelSpec.model_validate(raw)


def _family_of(model: ModelSpec) -> str:
    if model.family:
        return model.family
    repo = model.repo.lower()
    if "opt" in repo:
        return "opt"
    if "llama" in repo or "smollm" in repo:
        return "llama"
    return "unknown"


def expand(exp: ExperimentSpec) -> list[RunSpec]:
    """The full product, in a deterministic order: model, grid entry, bits, gs, dataset."""
    runs: list[RunSpec] = []
    for key in exp.models:
        model = load_model(key)
        fam_over = exp.family_overrides.get(_family_of(model), {})
        for entry in exp.grid:
            base_opts = {**fam_over, **entry.options}
            if entry.algo == "fp":
                quant = QuantSpec(algo="fp", **{k: v for k, v in base_opts.items() if k in QuantSpec.model_fields})
                runs.extend(_rows_for(model, quant, None, exp))
                continue
            cells = entry.cells or [GridCell(bits=[4])]
            seen: set[tuple[int, int]] = set()
            for cell in cells:
                group_sizes = list(cell.group_size)
                for extra in model.extra_group_sizes:
                    if extra not in group_sizes and -1 in group_sizes:
                        group_sizes.append(extra)
                for bits in cell.bits:
                    for gs in group_sizes:
                        if (bits, gs) in seen:
                            continue
                        seen.add((bits, gs))
                        quant = QuantSpec(algo=entry.algo, bits=bits, group_size=gs, **base_opts)
                        calib = exp.calib if quant.is_data_dependent() else None
                        runs.extend(_rows_for(model, quant, calib, exp))
    return runs


def _rows_for(model: ModelSpec, quant: QuantSpec, calib: CalibSpec | None, exp: ExperimentSpec) -> list[RunSpec]:
    return [
        RunSpec(model=model, quant=quant, calib=calib, dataset=ds, eval=exp.eval, dtype=model.dtype)
        for ds in exp.datasets
    ]


def group_by_quant_key(runs: list[RunSpec]) -> dict[str, list[RunSpec]]:
    groups: dict[str, list[RunSpec]] = {}
    for run in runs:
        groups.setdefault(run.quant_key, []).append(run)
    return groups
