"""Copy regression tests: the plain-language surfaces stay plain, and the guides' links
resolve. Add a word here when a usability session shows people do not know it."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ptqbench import doctor as Dr
from ptqbench import explain as E
from ptqbench import paths

pytestmark = pytest.mark.smoke
ROOT = paths.repo_root()

# Raw keys and shorthand that must not appear in text a newcomer reads first.
JARGON = ("c4_new", "ptb_new", "awq_lite", "act_order", "group_size", "quant_key", "g128", "g64", "fp16 baseline", "RunSpec")

GUIDES = ["README.md", "HOW-IT-WORKS.md", "results/README.md", "docs/EXPORT-MLIR.md", "docs/USABILITY-TEST.md",
          "docs/DESIGN.md", "docs/RESULTS.md", "docs/SERVER.md"]


def _menu_texts() -> list[str]:
    out = [name for _, name, _ in E.METHOD_MENU] + [blurb for _, _, blurb in E.METHOD_MENU]
    out += [blurb for _, blurb in E.DATASET_MENU] + list(E.DATASET_NAMES.values())
    out += [blurb for _, blurb in E.BITS_MENU] + [blurb for _, blurb in E.GROUPING_MENU] + [blurb for _, blurb in E.CALIB_MENU]
    return out


def test_menus_carry_no_raw_keys():
    for text in _menu_texts():
        assert text.strip(), "an empty menu line"
        for word in JARGON:
            assert word not in text, f"{word!r} in menu text {text!r}"


def test_every_reason_explanation_has_a_fix_or_is_final():
    samples = ["import_error:hqq.core.quantize:ModuleNotFoundError", "group_size_indivisible:128:[576, 1536]",
               "bits_unsupported:5", "device_unsupported:cpu", "unknown_algo:x", "oom",
               "OSError: x is not a valid model identifier", "ConnectionError: timed out", "RuntimeError: odd"]
    for reason in samples:
        text = E.explain_reason(reason)
        assert "Fix:" in text, reason
        assert text.endswith("."), reason
    assert E.explain_reason("KeyboardInterrupt: ").endswith(".")


def test_doctor_fixes_are_sentences_people_can_act_on():
    for c in Dr.run_checks(probe_internet=False):
        if c.status in (Dr.FAIL, Dr.WARN):
            assert c.fix and len(c.fix) > 15, c
        for word in JARGON:
            assert word not in c.detail and word not in c.fix, (c, word)


def test_verdict_vocabulary_matches_the_results_guide():
    guide = (ROOT / "results/README.md").read_text(encoding="utf-8")
    assert "under 5%" in guide.lower() and "5–20%" in guide or "5-20%" in guide
    base = (27.66, False)
    assert "hard to notice" in E.verdict({"model": "m", "dataset": "wikitext2", "algo": "rtn", "bits": 8, "group_size": 128, "ppl": 27.7}, base)
    assert "noticeable" in E.verdict({"model": "m", "dataset": "wikitext2", "algo": "rtn", "bits": 4, "group_size": 128, "ppl": 30.0}, base)


@pytest.mark.parametrize("guide", GUIDES)
def test_guide_links_resolve(guide):
    path = ROOT / guide
    text = path.read_text(encoding="utf-8")
    for label, target in re.findall(r"(?<!!)\[([^\]]*)\]\(([^)\s]+)\)", text):
        if re.match(r"^(https?:|mailto:)", target):
            continue
        file_part = target.split("#")[0]
        if not file_part:
            continue  # same-page anchor
        resolved = (path.parent / file_part).resolve()
        assert resolved.exists(), f"{guide}: [{label}]({target}) -> {resolved} does not exist"


def test_readme_doors_point_at_real_entry_points():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "## Start here" in readme and readme.index("## Start here") < readme.index("## Quick start")
    assert "./install.sh" in readme and "./ptq" in readme and "docs/archive/" in readme
    for anchor in ("#quick-start-new-machine", "#running-the-matrix", "#adding-a-model"):
        heading = anchor[1:].replace("-", " ")
        assert heading.split()[0] in readme.lower(), anchor


def test_launcher_and_installer_are_executable():
    for name in ("ptq", "install.sh", "scripts/desktop-shortcut.sh"):
        p = ROOT / name
        assert p.is_file() and p.stat().st_mode & 0o111, f"{name} is not executable"
        assert p.read_text().startswith("#!/usr/bin/env bash")


def test_colab_notebook_is_valid_and_points_home():
    import json

    nb = json.loads((ROOT / "notebooks/ptq_colab.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and nb["cells"]
    sources = ["".join(c["source"]) for c in nb["cells"]]
    assert any("git clone" in s and "LLM-Precision" in s for s in sources)
    assert any("./install.sh" in s for s in sources) and any("demo_answers" in s for s in sources)
    assert Path(ROOT / "notebooks/ptq_colab.ipynb").stat().st_size < 50_000
