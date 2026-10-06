"""`ptq ui`: the endpoints behind the page, with the run itself stubbed so the test is
about the plumbing: options, validation, one run to a verdict, state, events, results."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from ptqbench import ui as U
from ptqbench import wizard as W

pytestmark = pytest.mark.smoke


def _fake_rows(a: W.Answers, *, device_spec="auto", write=True, log=print):
    log("[dim]Step 1:[/dim] pretending to load")
    return [{
        "model": "facebook/opt-125m", "model_key": a.model_key, "dataset": d, "algo": a.algo, "bits": a.bits,
        "group_size": a.group_size, "ppl": 27.66 if a.algo == "fp" else 29.45, "partial": a.quick, "status": "ok",
        "n_windows": 20 if a.quick else 140, "eval_mode": "resident", "eval_seconds": 1.5, "quant_seconds": 0.2,
        "peak_vram_gb": 0.5, "calib": None, "_written_to": None,
    } for d in a.datasets]


@pytest.fixture
def ui(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "execute_answers", _fake_rows)
    monkeypatch.setattr(U, "W", W)
    ui = U.Ui(device_spec="cpu", site_dir=tmp_path / "site", build_site=True)
    monkeypatch.setattr(ui, "rebuild_site", lambda keys: None)
    monkeypatch.setattr(W, "estimate", lambda a, device_spec="auto": "under a minute")
    return ui


@pytest.fixture
def server(ui):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), U.make_handler(ui))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _get(url: str):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, json.loads(r.read()) if "json" in r.headers.get("Content-Type", "") else r.read().decode()


def _post(url: str, body: dict):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_options_speak_plainly(ui):
    o = ui.options()
    assert o["recommended_bits"] == 4 and o["device"] == "cpu"
    assert any(m["key"] == "opt-125m" for m in o["models"])
    smol = next(m for m in o["models"] if m["key"] == "smollm2-135m")
    assert smol["recommended_group_size"] == 64 and smol["act_order_default"] is True
    assert o["datasets"][0]["name"] == "WikiText-2 (Wikipedia articles)"
    assert len(o["demo"]) == 2 and "full precision" in o["demo"][0]


def test_answers_validation(ui):
    a = ui.answers_from({"model_key": "opt-125m", "algo": "gptq", "bits": 4, "act_order": "default"})
    assert a.group_size == 128 and a.act_order is None and a.calib_dataset == "c4"
    with pytest.raises(ValueError, match="unknown model"):
        ui.answers_from({"model_key": "nope"})
    with pytest.raises(ValueError, match="bits"):
        ui.answers_from({"model_key": "opt-125m", "bits": 5})
    with pytest.raises(ValueError, match="texts"):
        ui.answers_from({"model_key": "opt-125m", "datasets": ["moon"]})
    assert ui.start({"model_key": "opt-125m", "bits": 5})[0] == 400


def test_demo_runs_to_a_verdict_over_http(server):
    status, page = _get(server + "/")
    assert status == 200 and "Show me something in two minutes" in page
    status, body = _post(server + "/api/run", {"demo": True})
    assert status == 200 and len(body["plan"]) == 2 and body["plan"][0]["estimate"] == "under a minute"
    for _ in range(100):
        _, state = _get(server + "/api/state")
        if state["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert state["status"] == "done", state.get("error")
    assert len(state["rows"]) == 2 and state["rows"][1]["ppl"] == 29.45
    assert "Baseline" in state["verdicts"][0]
    assert "6.5% worse" in state["verdicts"][1] and "noticeable but usable" in state["verdicts"][1]
    assert any("Step 1: pretending to load" in line for line in state["log"]), "markup stripped, stage logged"
    status, page = _get(server + "/results/")
    assert status == 200 and "Results by model" in page
    status, body = _post(server + "/api/run", {"model_key": "opt-125m", "algo": "rtn", "bits": 4, "quick": True})
    assert status == 200


def test_busy_and_bad_requests(ui, server):
    with ui.lock:
        ui.state.status = "running"
    status, body = _post(server + "/api/run", {"demo": True})
    assert status == 409 and "already in progress" in body["error"]
    with ui.lock:
        ui.state.status = "idle"
    status, body = _post(server + "/api/run", {"model_key": "nope"})
    assert status == 400 and "unknown model" in body["error"]
    try:
        _get(server + "/results/../../etc/passwd")
        raise AssertionError("path traversal served a file")
    except urllib.error.HTTPError as e:
        assert e.code == 404


def test_events_stream_ends_with_done(server):
    _post(server + "/api/run", {"demo": True})
    with urllib.request.urlopen(server + "/api/events", timeout=15) as r:
        seen = b""
        while b"event: done" not in seen:
            chunk = r.readline()
            if not chunk:
                break
            seen += chunk
        seen += r.readline()  # the data line that follows the done event
    assert b"event: done" in seen and b'"status": "done"' in seen
