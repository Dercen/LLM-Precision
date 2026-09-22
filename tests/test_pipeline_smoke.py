"""End-to-end on CPU with tiny settings: the CI proof that the pipeline runs at all.

Mirrors PLAN.md 9's smoke config: wikitext2 calibration (6 MB, no C4 download),
short sequences, a handful of windows. Nothing here checks accuracy -- the gates do.
"""

from __future__ import annotations

import math

import pytest
import torch

from ptqbench.data import calibration as C
from ptqbench.eval.run_eval import run_single_eval

pytestmark = pytest.mark.smoke

MODEL = "facebook/opt-125m"
SMOKE_CALIB = C.CalibSpec("wikitext2", nsamples=4, seqlen=256, seed=0)


def _row(**kw):
    return run_single_eval(
        model_id=MODEL, dataset_key="wikitext2", device_spec="cpu", seqlen=256,
        max_windows=2, calib=SMOKE_CALIB, write=False, **kw,
    )


@pytest.mark.parametrize("algo,bits", [("fp", 16), ("rtn", 4), ("gptq", 4)])
def test_each_algorithm_runs_on_cpu(algo, bits):
    row = _row(algo=algo, bits=bits)
    assert row["status"] == "ok"
    assert math.isfinite(row["ppl"]) and row["ppl"] > 1
    assert row["partial"] is True and row["paper_comparable"] is False
    assert row["dtype"] == "torch.float32"
    assert row["algo"] == algo
    if algo == "gptq":
        assert row["calib"] == {"dataset": "wikitext2", "nsamples": 4, "seqlen": 256, "seed": 0}
        assert row["n_quantized_modules"] == 72


def test_streamed_mode_runs_on_cpu():
    """Offload is a no-op on CPU but the code path must still work there."""
    row = _row(algo="fp", eval_mode="streamed")
    assert row["eval_mode"] == "streamed"
    assert math.isfinite(row["ppl"])


def test_hqq_adapter_placeholder():
    """PLAN.md 6: hqq rows are skipped, not crashed, when the backend is absent.

    The adapter lands at M4; until then this pins the import that it will use.
    """
    pytest.importorskip("hqq.core.quantize")
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear  # noqa: F401

    assert torch  # keep torch imported for the monkeypatch the M4 test will add
