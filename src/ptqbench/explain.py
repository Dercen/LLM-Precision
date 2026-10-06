"""Plain-language text for the surfaces a newcomer meets: the wizard's menus, the
sentence after a run, and the "why did this not produce a number" lines on the
by-model pages. One module, so every surface uses the same words. Nothing here
touches a measurement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# The labels `ptq aggregate` prints on the by-model pages. summary.md keeps raw keys.
ALGO_NAMES = {
    "rtn": "RTN (plain rounding)",
    "gptq": "GPTQ",
    "awq_lite": "AWQ-lite",
    "hqq": "HQQ",
}
DATASET_NAMES = {
    "wikitext2": "WikiText-2 (Wikipedia articles)",
    "c4_new": "C4 (web pages)",
    "ptb_new": "PTB (1980s news sentences)",
    "c4": "C4, random windows",
    "ptb": "PTB, original split",
}

# Menu text for the wizard: (label, one line of what it means for the person choosing).
METHOD_MENU: list[tuple[str, str, str]] = [
    ("fp", "Full precision", "the unmodified model; the baseline every other number is compared with"),
    ("rtn", "RTN (plain rounding)", "round each weight to the nearest allowed value; no sample text, seconds"),
    ("gptq", "GPTQ", "rounds one column at a time and corrects its own errors; needs sample text, minutes"),
    ("awq_lite", "AWQ-lite", "protects the weights that matter most, then rounds; needs sample text"),
    ("hqq", "HQQ", "plain rounding on a better-placed grid; no sample text"),
]
DATASET_MENU: list[tuple[str, str]] = [
    ("wikitext2", "the standard test in every paper"),
    ("c4_new", "the papers' second column"),
    ("ptb_new", "the papers' third column"),
    ("c4", "variant; not what the papers report"),
    ("ptb", "variant; not what the papers report"),
]
BITS_MENU: list[tuple[int, str]] = [
    (8, "almost no loss, half the size"),
    (4, "the usual target, a quarter the size (recommended)"),
    (3, "noticeable loss"),
    (2, "usually breaks the model"),
]
GROUPING_MENU: list[tuple[int, str]] = [
    (128, "groups of 128: recommended; what the Llama papers use"),
    (64, "groups of 64: slightly better, slightly bigger"),
    (-1, "per row: the GPTQ paper's setting for OPT models"),
]
CALIB_MENU: list[tuple[str, str]] = [
    ("c4", "C4 web text, 128 passages: what the GPTQ paper used (downloads once)"),
    ("pile_val", "The Pile, 128 short passages: what the AWQ paper used"),
    ("wikitext2", "WikiText-2: quick, no extra download"),
]

RECOMMENDED_BITS = 4
RECOMMENDED_GROUP_SIZE = 128


def plain(text: str) -> str:
    """Rich markup ('[dim]Step 1:[/dim] ...') as plain text, for logs outside a terminal."""
    from rich.text import Text

    return Text.from_markup(str(text)).plain


def grouping(group_size: int | None) -> str:
    """-1 (or none) -> 'per row'; 128 -> 'groups of 128'."""
    if group_size is None or int(group_size) == -1:
        return "per row"
    return f"groups of {int(group_size)}"


def method_name(algo: str) -> str:
    return "Full precision" if algo == "fp" else ALGO_NAMES.get(algo, algo)


def dataset_name(key: str) -> str:
    return DATASET_NAMES.get(key, key)


# ---- why a run produced no number ------------------------------------------------------

_EXTRA_FOR_MODULE = {"hqq": "hqq", "bitsandbytes": "bnb", "lm_eval": "lmeval", "torch_mlir": "mlir"}


def explain_reason(reason: str | None) -> str:
    """One or two sentences for a skip or failure reason, ending with what to do.

    Reasons come from registry.check, execute.check_available and the matrix runner's
    `ExceptionName: message` rows; anything unrecognised falls through to the log file.
    """
    if not reason:
        return "No reason was recorded."
    r = str(reason)
    head, _, rest = r.partition(":")
    if head == "import_error":
        module = rest.split(":")[0].split(".")[0]
        extra = _EXTRA_FOR_MODULE.get(module)
        fix = f"run `uv sync --extra {extra}` (keep the extras you already use)" if extra else f"install the `{module}` package"
        return f"The `{module}` library is not installed on this machine. Fix: {fix}."
    if head == "group_size_indivisible":
        gs, _, dims = rest.partition(":")
        widths = dims.strip("[] ")
        return (
            f"Groups of {gs} do not divide this model's layer widths ({widths}). "
            "Fix: choose groups of 64 or per row."
        )
    if head == "bits_unsupported":
        return f"This method cannot do {rest}-bit. Fix: choose 2, 3, 4 or 8 bits, which every method supports."
    if head == "device_unsupported":
        return f"This method cannot run on the {rest}. Fix: use a machine with an NVIDIA GPU."
    if head == "unknown_algo":
        return f"There is no method called `{rest}`. Fix: use fp, rtn, gptq, awq_lite or hqq."
    if r == "unavailable":
        return "This configuration is not available on this machine."
    low = r.lower()
    if low == "oom" or "out of memory" in low or "outofmemoryerror" in low:
        return (
            "The GPU ran out of memory. Fix: close other programs that use the GPU, or run with "
            "`--device cpu`; the matrix runner also retries one layer at a time by itself."
        )
    if any(s in low for s in ("not a valid model identifier", "repositorynotfound", "gatedrepo", "401 client error", "403 client error")):
        return (
            "The model could not be downloaded: the name is wrong, the machine is offline, or the "
            "model is gated and needs a Hugging Face login. Fix: check the name on huggingface.co; "
            "for a gated model accept its licence and run `uv run hf auth login`."
        )
    if any(s in low for s in ("connectionerror", "name resolution", "timed out", "max retries", "offline")):
        return (
            "A download failed because the machine is offline. Fix: connect to the internet, or run "
            "`uv run ptq prefetch <experiment>` while online and try again."
        )
    if head == "KeyboardInterrupt":
        return "Stopped by the user (Ctrl-C)."
    return (
        f"Failed with `{r[:160]}`. Fix: the `.log` file next to the result row in "
        "results/raw/runs/ has the full error."
    )


# ---- the sentence after a run ------------------------------------------------------------


def baseline_ppl(model: str, dataset: str, *, runs_dir: Path | None = None, allow_partial: bool = False) -> tuple[float, bool] | None:
    """The full-precision perplexity already measured for (model, dataset): (ppl, partial).

    Full rows win over quick previews; among equals the newest. None when nothing fits.
    """
    if runs_dir is None:
        from . import paths

        runs_dir = paths.runs_dir()
    best: tuple[tuple[bool, str], float, bool] | None = None
    for p in Path(runs_dir).glob("*.json"):
        try:
            row = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("algo") != "fp" or row.get("status") != "ok" or row.get("ppl") is None:
            continue
        if row.get("model") != model or row.get("dataset") != dataset:
            continue
        partial = bool(row.get("partial"))
        if partial and not allow_partial:
            continue
        key = (not partial, str(row.get("finished_at") or ""))
        if best is None or key > best[0]:
            best = (key, float(row["ppl"]), partial)
    return None if best is None else (best[1], best[2])


def verdict(row: dict[str, Any], baseline: tuple[float, bool] | None) -> str:
    """What the number means, in one or two sentences a newcomer can act on.

    Thresholds follow results/README.md: under 5% worse is hard to notice, 5 to 20%
    is noticeable, anything at 2x or more is effectively broken.
    """
    model = row.get("model_key") or row.get("model") or "the model"
    text = dataset_name(row["dataset"])
    ppl = float(row["ppl"])
    approx = " (quick preview, approximate)" if row.get("partial") else ""
    if row.get("algo") == "fp":
        return (
            f"Baseline{approx}: {model} scores {ppl:.2f} on {text}. Every quantized run of this "
            "model is compared with that number; lower is better."
        )
    setting = f"{row['bits']} bits with {method_name(row['algo'])}, {grouping(row.get('group_size'))}"
    if baseline is None:
        return (
            f"At {setting}, {model} scores {ppl:.2f} on {text}{approx}. No full-precision baseline "
            "has been measured for this model on this text yet, so the loss cannot be stated: "
            'pick "Full precision" in the method menu to measure it.'
        )
    base, base_partial = baseline
    ratio = ppl / base
    if ratio < 1:
        how = f"{(1 - ratio) * 100:.1f}% better"
    elif ratio < 2:
        how = f"{(ratio - 1) * 100:.1f}% worse"
    else:
        how = f"{ratio:.1f}× worse"
    if ratio < 1.05:
        tail = "Under 5% is hard to notice in use."
    elif ratio < 1.2:
        tail = "Between 5% and 20% is noticeable but usable."
    elif ratio < 2:
        tail = "This is a clear loss of quality."
    else:
        tail = "The model is effectively broken at this setting."
    note = " (against a quick-preview baseline)" if base_partial else ""
    return f"At {setting}, {model} is {how} than the original on {text}{approx}{note}: {ppl:.2f} vs {base:.2f}. {tail}"
