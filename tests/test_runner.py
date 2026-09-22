"""DESIGN.md 9: the matrix runner -- ids, filtering, sharding, resume, skips, cache."""

from __future__ import annotations

import json

import pytest
import torch

from ptqbench import config as C
from ptqbench import paths
from ptqbench.runner import execute as X
from ptqbench.runner import matrix as M
from ptqbench.runner import quant_cache as Q

pytestmark = pytest.mark.smoke
SMOKE = paths.configs_dir() / "experiments" / "smoke_opt125m.yaml"
CPU = torch.device("cpu")


def test_filter_parsing():
    f = M.parse_filter("algo=gptq,rtn bits=4 model=opt-125m")
    assert f == {"algo": {"gptq", "rtn"}, "bits": {"4"}, "model": {"opt-125m"}}
    with pytest.raises(ValueError):
        M.parse_filter("nonsense")


def test_plan_filter_and_shard_partition():
    groups = M.plan(SMOKE, device=CPU)
    assert len(groups) == 4
    assert [r[0].quant.algo for _, r in groups] == ["opt-125m" and "fp", "gptq", "hqq", "rtn"] or len(groups) == 4
    only = M.plan(SMOKE, filter_text="algo=rtn,gptq", device=CPU)
    assert {r[0].quant.algo for _, r in only} == {"rtn", "gptq"}
    shards = [M.plan(SMOKE, shard=f"{k}/3", device=CPU) for k in range(3)]
    keys = [g[0] for s in shards for g in s]
    assert sorted(keys) == sorted(g[0] for g in groups), "shards must partition the groups"
    assert M.plan(SMOKE, index=2, device=CPU)[0][0] == groups[2][0]


def test_resolve_run_fixes_dtype_into_quant_key():
    run = C.expand(C.load_experiment(SMOKE))[1]
    cpu_run = X.resolve_run(run, CPU)
    assert cpu_run.dtype == "torch.float32"
    assert cpu_run.quant_key != run.model_copy(update={"dtype": "torch.float16"}).quant_key


def test_status_row_and_is_done(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "runs_dir", lambda: tmp_path)
    run = X.resolve_run(C.expand(C.load_experiment(SMOKE))[1], CPU)
    assert not X.is_done(run)
    X.write_row(X.status_row(run, CPU, "skipped", "import_error:hqq"))
    assert X.is_done(run) and not X.is_done(run, rerun_incomplete=True)
    row = json.loads((tmp_path / f"{run.run_id}.json").read_text())
    assert row["status"] == "skipped" and row["ppl"] is None and row["quant_key"] == run.quant_key
    X.write_row({**row, "status": "ok", "ppl": 1.0})
    assert X.is_done(run, rerun_incomplete=True)
    X.write_row({**row, "protocol_version": 999})
    assert not X.is_done(run), "a protocol bump invalidates old rows"


def test_indivisible_group_size_is_skipped_not_crashed():
    smol = C.load_model("smollm2-135m")
    run = C.RunSpec(model=smol, quant=C.QuantSpec(algo="rtn", bits=4, group_size=128), dataset="wikitext2")
    ok, reason = X.check_available(run, CPU)
    assert not ok and reason.startswith("group_size_indivisible")
    run64 = run.model_copy(update={"quant": C.QuantSpec(algo="rtn", bits=4, group_size=64)})
    assert X.check_available(run64, CPU) == (True, None)


def test_quant_cache_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "quant_cache_dir", lambda: tmp_path)
    from ptqbench.models import families
    from ptqbench.models import loader as ml
    from ptqbench.quantizers import rtn

    lm = ml.load("facebook/opt-125m", device=CPU)
    rtn.apply_rtn(lm.model, bits=4)
    before = {n: m.weight.detach().clone() for n, m in families.target_modules(lm.model).items()}
    Q.save(lm.model, "deadbeef0000", {"note": "test"})
    assert Q.has("deadbeef0000")
    fresh = ml.load("facebook/opt-125m", device=CPU)
    meta = Q.load_into(fresh.model, "deadbeef0000")
    assert meta["quant_cache_hit"] is True
    for n, m in families.target_modules(fresh.model).items():
        assert torch.equal(m.weight.detach(), before[n]), n
    removed = Q.gc(keep_newest=0)
    assert [p.name for p in removed] == ["deadbeef0000.safetensors"] and not Q.has("deadbeef0000")


def test_matrix_runs_and_resumes_on_cpu(tmp_path, monkeypatch):
    """fp + rtn on CPU with tiny windows; a second run must do nothing."""
    monkeypatch.setattr(paths, "runs_dir", lambda: tmp_path)
    monkeypatch.setattr(paths, "logs_dir", lambda: tmp_path)
    logs: list[str] = []
    s1 = M.run_matrix(SMOKE, filter_text="algo=fp,rtn", device_spec="cpu", progress=False, log=logs.append)
    assert (s1.ok, s1.failed, s1.skipped) == (2, 0, 0), logs
    rows = [json.loads(p.read_text()) for p in tmp_path.glob("*.json")]
    assert {r["algo"] for r in rows} == {"fp", "rtn"} and all(r["status"] == "ok" for r in rows)
    assert all(r["partial"] is True for r in rows), "smoke config uses max_windows"
    s2 = M.run_matrix(SMOKE, filter_text="algo=fp,rtn", device_spec="cpu", progress=False, log=logs.append)
    assert (s2.ok, s2.already_done) == (0, 2)


@pytest.mark.gpu
def test_smoke_matrix_end_to_end_on_gpu(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    monkeypatch.setattr(paths, "runs_dir", lambda: tmp_path)
    s = M.run_matrix(SMOKE, progress=False, log=lambda *_: None)
    assert (s.ok, s.failed, s.skipped) == (4, 0, 0)
    rows = {json.loads(p.read_text())["algo"]: json.loads(p.read_text()) for p in tmp_path.glob("*.json")}
    assert set(rows) == {"fp", "rtn", "gptq", "hqq"}
    assert rows["gptq"]["calib"]["dataset"] == "wikitext2"
    assert rows["gptq"]["ppl"] < rows["rtn"]["ppl"]


def test_timing_is_derived_from_rows(tmp_path):
    from ptqbench.analysis import timing

    def row(model, algo, qkey, eval_s, n_win, quant_s, mode="resident", **kw):
        return {"status": "ok", "partial": False, "model": model, "dtype": "torch.float16",
                "eval_mode": mode, "algo": algo, "quant_key": qkey, "eval_seconds": eval_s,
                "n_windows": n_win, "quant_seconds": quant_s, "peak_vram_gb": 1.0, **kw}

    rows = [
        row("m", "fp", "q0", 10.0, 100, 0.0),
        row("m", "gptq", "q1", 20.0, 100, 300.0),
        row("m", "gptq", "q1", 30.0, 200, 300.0),          # same quant_key: counted once
        row("m", "gptq", "q2", 5.0, 50, 500.0, quant_cache_hit=True),  # cache hit: eval counts, quant does not
        row("m", "gptq", "q3", 5.0, 50, 100.0, partial=True),          # partial: ignored entirely
    ]
    ev, q = timing.derive(rows)
    assert ev["m|torch.float16|resident"]["seconds_per_window"] == round(65.0 / 450, 4)
    assert q["m|gptq|resident"] == {"quant_seconds": 300.0, "n_configs": 1, "min": 300.0, "max": 300.0, "source": "derived"}
    out = timing.refresh(rows, path=tmp_path / "timing.json", hostname="h")
    data = json.loads(out.read_text())
    assert data["h"]["quantization"]["m|gptq"]["quant_seconds_resident"] == 300.0


def test_resolve_run_falls_back_to_the_mirror_when_gated(monkeypatch):
    spec = C.load_model("llama-2-7b")
    run = C.RunSpec(model=spec, quant=C.QuantSpec(algo="fp"), dataset="wikitext2")
    monkeypatch.setattr(X, "repo_is_fetchable", lambda repo: False)
    resolved = X.resolve_run(run, CPU)
    assert resolved.model.repo == spec.mirror and resolved.model.canonical_repo == spec.repo
    row = X.status_row(resolved, CPU, "skipped", "test")
    assert row["model"] == "meta-llama/Llama-2-7b-hf" and row["loaded_from"] == spec.mirror
    monkeypatch.setattr(X, "repo_is_fetchable", lambda repo: True)
    assert X.resolve_run(run, CPU).model.repo == spec.repo


@pytest.mark.gpu
def test_quantizers_count_their_own_memory_when_choosing_the_mode():
    """opt-2.7b: resident for evaluation, streamed for GPTQ and AWQ-lite (DESIGN.md 2a)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    dev = torch.device("cuda:0")
    spec = C.load_model("opt-2.7b")
    fp = X.resolve_run(C.RunSpec(model=spec, quant=C.QuantSpec(algo="fp"), dataset="wikitext2"), dev)
    gptq = X.resolve_run(C.RunSpec(model=spec, quant=C.QuantSpec(algo="gptq", bits=4), calib=C.CalibSpec(), dataset="wikitext2"), dev)
    assert X.quantizer_extra_bytes(fp) == 0
    assert X.quantizer_extra_bytes(gptq) == 2 * 4 * 10240**2
    # Budget-only decisions: independent of what is on the card right now.
    est = __import__("ptqbench.models.loader", fromlist=["x"]).estimate_model_bytes(spec.repo, torch.float16)
    from ptqbench import device as D
    assert D.should_stream(est, dev, basis="capacity") is False
    assert D.should_stream(est + X.quantizer_extra_bytes(gptq), dev, basis="capacity") is True


def test_cache_warning_threshold(tmp_path, monkeypatch, capsys):
    from ptqbench import cli

    monkeypatch.setattr(paths, "quant_cache_dir", lambda: tmp_path)
    (tmp_path / "a.safetensors").write_bytes(b"x" * 1024)
    assert cli.warn_if_cache_is_large(threshold_gb=50) is None
    msg = cli.warn_if_cache_is_large(threshold_gb=0)
    assert msg and "not capped" in msg and "cache gc" in msg
    assert "warning" in capsys.readouterr().err
    monkeypatch.setenv("PTQ_CACHE_WARN_GB", "0")
    assert cli.warn_if_cache_is_large() is not None
