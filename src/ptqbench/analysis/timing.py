"""results/runs -> results/timing.json. DESIGN.md 2b and 9.1.

`ptq run --dry-run` sizes a matrix from seconds-per-window and quantization seconds
per (model, dtype, eval_mode) and (model, algo). Hand-entered numbers age; every
finished row carries the real ones, so this derives them and merges the result into
timing.json per hostname -- measured values replace hand-entered ones, entries with
no rows are kept.
"""

from __future__ import annotations

import json
import socket
import statistics
from pathlib import Path
from typing import Any

from .. import paths


def derive(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """(eval entries, quantization entries) from complete, non-partial rows."""
    eval_acc: dict[str, dict[str, float]] = {}
    quant_acc: dict[str, dict[str, float]] = {}
    seen_quant: set[str] = set()
    for row in rows:
        if row.get("status") != "ok" or row.get("partial") or not row.get("n_windows"):
            continue
        key = f"{row['model']}|{row['dtype']}|{row.get('eval_mode', 'resident')}"
        acc = eval_acc.setdefault(key, {"seconds": 0.0, "windows": 0, "peak_vram": 0.0, "rows": 0})
        acc["seconds"] += float(row["eval_seconds"])
        acc["windows"] += int(row["n_windows"])
        acc["peak_vram"] = max(acc["peak_vram"], float(row.get("peak_vram_gb") or 0.0))
        acc["rows"] += 1

        algo = row.get("algo")
        qkey = row.get("quant_key")
        if algo in ("gptq", "awq_lite") and qkey and qkey not in seen_quant and not row.get("quant_cache_hit"):
            secs = float(row.get("quant_seconds") or 0.0)
            if secs > 0:
                seen_quant.add(qkey)
                mode = row.get("quant_eval_mode", row.get("eval_mode", "resident"))
                q = quant_acc.setdefault(f"{row['model']}|{algo}|{mode}", {"samples": []})
                q["samples"].append(secs)

    eval_entries = {
        key: {
            "seconds_per_window": round(a["seconds"] / a["windows"], 4),
            "windows_measured": a["windows"],
            "rows": a["rows"],
            "peak_vram_gb": round(a["peak_vram"], 3),
            "source": "derived",
        }
        for key, a in eval_acc.items()
    }
    quant_entries = {
        key: {
            "quant_seconds": round(statistics.median(q["samples"]), 1),
            "n_configs": len(q["samples"]),
            "min": round(min(q["samples"]), 1),
            "max": round(max(q["samples"]), 1),
            "source": "derived",
        }
        for key, q in quant_acc.items()
    }
    return eval_entries, quant_entries


def refresh(rows: list[dict[str, Any]], path: Path | None = None, hostname: str | None = None) -> Path:
    path = path or paths.timing_file()
    host = hostname or socket.gethostname()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    block = data.setdefault(host, {})
    block.setdefault("entries", {})
    block.setdefault("quantization", {})
    eval_entries, quant_entries = derive(rows)
    block["entries"].update(eval_entries)
    for key, entry in quant_entries.items():
        model, algo, mode = key.split("|")
        # The runner looks up "<model>|<algo>"; keep the mode as detail inside.
        block["quantization"][f"{model}|{algo}"] = {**entry, "mode": mode, "quant_seconds_resident": entry["quant_seconds"]}
    block["derived_at"] = __import__("datetime").datetime.now().isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path
