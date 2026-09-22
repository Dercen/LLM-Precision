"""references/literature.yaml must stay joinable and internally consistent."""

from __future__ import annotations

import pytest
import yaml

from ptqbench import paths

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def lit():
    with open(paths.references_dir() / "literature.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_every_row_has_the_join_keys(lit):
    for row in lit["rows"]:
        for key in ("model", "dataset_key", "algo", "bits", "ppl", "source", "table"):
            assert key in row, f"row missing {key}: {row}"


def test_data_dependent_rows_declare_calibration(lit):
    """PLAN.md 13: the join must never pair an awq_lite row with the wrong calib."""
    for row in lit["rows"]:
        if row["algo"] in ("gptq", "awq_lite") and not row.get("protocol_uncertain"):
            assert "calib" in row, f"data-dependent row without calib: {row}"
            assert row["calib"] in lit["calibrations"], row["calib"]


def test_sources_are_declared(lit):
    for row in lit["rows"]:
        assert row["source"] in lit["sources"], row["source"]


def test_opt_ptb_and_c4_use_the_resolved_new_variants(lit):
    """M2 resolved these to the --new-eval keys; nothing may reintroduce the old ones."""
    keys = {row["dataset_key"] for row in lit["rows"]}
    assert "ptb_new" in keys and "c4_new" in keys
    assert "ptb" not in keys and "c4" not in keys


def test_no_duplicate_cells_within_a_source(lit):
    seen = set()
    for row in lit["rows"]:
        cell = (row["model"], row["dataset_key"], row["algo"], row["bits"],
                row.get("group_size"), row["source"])
        assert cell not in seen, f"duplicate literature cell: {cell}"
        seen.add(cell)
