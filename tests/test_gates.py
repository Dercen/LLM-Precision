"""Milestone gates, locked as regressions. DESIGN.md 10 and 13.

These are the numbers that say the pipeline still reproduces the papers. They run
on GPU and take about a minute; CI does not run them, each milestone does.
"""

from __future__ import annotations

import pytest
import torch

from ptqbench.eval.run_eval import run_single_eval

pytestmark = pytest.mark.gpu

MODEL = "facebook/opt-125m"


def _ppl(**kw) -> float:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return run_single_eval(model_id=MODEL, write=False, **kw)["ppl"]


@pytest.mark.parametrize(
    "dataset_key,algo,bits,group_size,paper,tol_pct",
    [
        # M1 / M2 hard gates -- wikitext2 is the only one the plan treats as binding.
        ("wikitext2", "fp", 16, -1, 27.65, 0.2),
        ("wikitext2", "rtn", 4, -1, 37.28, 1.0),
        # Resolved at M2: the GPTQ paper's Tables 9/11 are the --new-eval variants.
        ("ptb_new", "fp", 16, -1, 38.99, 1.0),
        ("ptb_new", "rtn", 4, -1, 53.89, 1.0),
        ("c4_new", "fp", 16, -1, 26.56, 1.0),
        ("c4_new", "rtn", 4, -1, 33.91, 1.0),
    ],
)
def test_matches_published(dataset_key, algo, bits, group_size, paper, tol_pct):
    got = _ppl(dataset_key=dataset_key, algo=algo, bits=bits, group_size=group_size)
    delta_pct = (got - paper) / paper * 100
    assert abs(delta_pct) <= tol_pct, (
        f"{MODEL} {dataset_key} {algo}{bits}: got {got:.4f}, paper {paper}, "
        f"delta {delta_pct:+.3f}% (tolerance +/-{tol_pct}%)"
    )


def test_plain_ptb_and_c4_are_not_the_paper_columns():
    """Guards the M2 resolution: if these ever match, the variant choice is wrong."""
    assert abs(_ppl(dataset_key="ptb", algo="fp") - 38.99) / 38.99 > 0.05
    assert abs(_ppl(dataset_key="c4", algo="fp") - 26.56) / 26.56 > 0.02


def test_rtn8_is_near_lossless():
    fp = _ppl(dataset_key="wikitext2", algo="fp")
    rtn8 = _ppl(dataset_key="wikitext2", algo="rtn", bits=8)
    assert abs(rtn8 - fp) <= 0.05, f"RTN8 {rtn8} vs fp {fp}"


def test_quality_is_monotone_in_bits():
    """Fewer bits must never help. A violation means the grid or family map is wrong."""
    ppls = {b: _ppl(dataset_key="wikitext2", algo="rtn", bits=b) for b in (8, 4, 3)}
    assert ppls[8] < ppls[4] < ppls[3], ppls
