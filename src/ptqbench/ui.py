"""`ptq ui`: the wizard in a browser, on this machine, with no terminal to learn.

Built on the same functions as the menu (`wizard.Answers`, `build_runs`,
`execute_answers`), so a run started from the page is a real result row with the same
ids and provenance. No new dependencies: Python's http.server, one HTML page, and a
server-sent-events stream for the live log. After a run it refreshes the tables, the
model's charts and the static site, which it serves under /results/.
"""

from __future__ import annotations

import json
import mimetypes
import threading
import time
import webbrowser
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from . import explain as E
from . import paths
from . import wizard as W


@dataclass
class RunState:
    status: str = "idle"  # idle | running | done | failed
    plan: list[dict[str, str]] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    verdicts: list[str] = field(default_factory=list)
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None


class Ui:
    """State and actions behind the endpoints; testable without a socket."""

    def __init__(self, *, device_spec: str = "auto", site_dir: Path | None = None, build_site: bool = True):
        self.device_spec = device_spec
        self.site_dir = Path(site_dir) if site_dir else paths.repo_root() / "site"
        self.state = RunState()
        self.lock = threading.Lock()
        self._thread: threading.Thread | None = None
        if build_site:
            self._try_build_site()

    # ---- read ----------------------------------------------------------------------

    def options(self) -> dict[str, Any]:
        from . import config as C
        from . import device as D

        models = []
        for m in W.model_choices():
            spec = C.load_model(m["key"])
            models.append({**m, "recommended_group_size": W.recommended_group_size(m["key"]),
                           "act_order_default": C._family_of(spec) == "llama"})
        return {
            "device": str(D.resolve(self.device_spec)),
            "models": models,
            "datasets": [{"key": k, "name": E.dataset_name(k), "blurb": b} for k, b in E.DATASET_MENU],
            "methods": [{"key": k, "name": n, "blurb": b} for k, n, b in E.METHOD_MENU],
            "bits": [{"value": b, "blurb": blurb} for b, blurb in E.BITS_MENU],
            "grouping": [{"value": g, "blurb": blurb} for g, blurb in E.GROUPING_MENU],
            "calib": [{"key": k, "blurb": b} for k, b in E.CALIB_MENU],
            "recommended_bits": E.RECOMMENDED_BITS,
            "demo": [W._summary(a) for a in W.demo_answers()],
        }

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return asdict(self.state)

    def doctor(self) -> list[dict[str, str]]:
        from . import doctor

        return [asdict(c) for c in doctor.run_checks()]

    # ---- run -----------------------------------------------------------------------

    def answers_from(self, payload: dict[str, Any]) -> W.Answers:
        keys = {m["key"] for m in W.model_choices()}
        model_key = str(payload.get("model_key", ""))
        if model_key not in keys:
            raise ValueError(f"unknown model {model_key!r}")
        datasets = [str(d) for d in (payload.get("datasets") or ["wikitext2"])]
        known = {k for k, _ in E.DATASET_MENU}
        if not datasets or not set(datasets) <= known:
            raise ValueError("pick at least one of the listed texts")
        algo = str(payload.get("algo", "rtn"))
        if algo not in {k for k, _, _ in E.METHOD_MENU}:
            raise ValueError(f"unknown method {algo!r}")
        a = W.Answers(model_key=model_key, datasets=datasets, algo=algo, quick=bool(payload.get("quick", False)))
        if algo != "fp":
            a.bits = int(payload.get("bits") or E.RECOMMENDED_BITS)
            if a.bits not in W.BITS:
                raise ValueError(f"bits must be one of {W.BITS}")
            a.group_size = int(payload.get("group_size") or W.recommended_group_size(model_key))
            act = payload.get("act_order")
            a.act_order = None if act in (None, "", "default") else bool(act)
            a.calib_dataset = str(payload.get("calib_dataset") or "c4")
        return a

    def start(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """Begin a run in the background; returns (http status, body)."""
        from . import device as D
        from .runner import execute as X

        with self.lock:
            if self.state.status == "running":
                return 409, {"error": "a run is already in progress; wait for it to finish"}
        try:
            plan = W.demo_answers() if payload.get("demo") else [self.answers_from(payload)]
        except (ValueError, KeyError) as exc:
            return 400, {"error": str(exc)}
        device = D.resolve(self.device_spec)
        ok, reason = X.check_available(W.build_runs(plan[-1])[0], device)
        if not ok:
            return 400, {"error": E.explain_reason(reason)}
        described = [{"summary": W._summary(a), "estimate": W.estimate(a, self.device_spec)} for a in plan]
        with self.lock:
            self.state = RunState(status="running", plan=described, started_at=time.time())
        self._thread = threading.Thread(target=self._run, args=(plan,), daemon=True)
        self._thread.start()
        return 200, {"ok": True, "plan": described}

    def _log(self, message: Any) -> None:
        with self.lock:
            self.state.log.append(E.plain(message))

    def _run(self, plan: list[W.Answers]) -> None:
        try:
            for i, a in enumerate(plan, start=1):
                self._log(f"Run {i} of {len(plan)}: {W._summary(a)}")
                rows = W.execute_answers(a, device_spec=self.device_spec, log=self._log)
                for r in rows:
                    baseline = E.baseline_ppl(r["model"], r["dataset"], allow_partial=bool(r["partial"]))
                    with self.lock:
                        self.state.rows.append(_row_summary(r))
                        self.state.verdicts.append(E.verdict(r, baseline))
                    self._log(f"Result: {E.dataset_name(r['dataset'])} perplexity {r['ppl']:.4f}")
            self._log("Updating the tables, charts and results pages")
            self.rebuild_site([a.model_key for a in plan])
            with self.lock:
                self.state.status = "done"
        except Exception as exc:  # noqa: BLE001 - the page shows a sentence, never a traceback
            with self.lock:
                self.state.error = E.explain_reason(f"{type(exc).__name__}: {exc}")
                self.state.status = "failed"
        finally:
            with self.lock:
                self.state.finished_at = time.time()

    def rebuild_site(self, model_keys: list[str]) -> None:
        from . import site
        from .analysis import plots
        from .analysis.aggregate import aggregate

        aggregate()
        plots.plot_all(models=model_keys)
        site.build(self.site_dir)

    def _try_build_site(self) -> None:
        from . import site

        try:
            site.build(self.site_dir)
        except Exception as exc:  # noqa: BLE001 - no results yet is not an error for the UI
            self._log(f"results pages not built yet ({type(exc).__name__}); run something first")


def _row_summary(r: dict[str, Any]) -> dict[str, Any]:
    paper, src = W.paper_number(r)
    return {
        "text": E.dataset_name(r["dataset"]),
        "ppl": round(float(r["ppl"]), 4),
        "paper": paper,
        "paper_source": src,
        "vs_paper_pct": round((r["ppl"] - paper) / paper * 100, 2) if paper else None,
        "passages": r.get("n_windows"),
        "partial": bool(r.get("partial")),
        "ran": r.get("eval_mode"),
        "seconds": round(float(r.get("eval_seconds") or 0) + float(r.get("quant_seconds") or 0)),
        "gpu_gb": r.get("peak_vram_gb"),
        "saved_to": r.get("_written_to"),
    }


# ---- HTTP ------------------------------------------------------------------------------


def make_handler(ui: Ui):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet: the console is for the run log
            pass

        def _json(self, status: int, body: Any) -> None:
            data = json.dumps(body, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                data = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif path == "/api/options":
                self._json(200, ui.options())
            elif path == "/api/state":
                self._json(200, ui.snapshot())
            elif path == "/api/doctor":
                self._json(200, ui.doctor())
            elif path == "/api/events":
                self._events()
            elif path.startswith("/results/"):
                self._static(path.removeprefix("/results/") or "index.html")
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                self._json(400, {"error": "the request was not JSON"})
                return
            if path == "/api/run":
                status, body = ui.start(payload)
                self._json(status, body)
            else:
                self._json(404, {"error": "not found"})

        def _static(self, rel: str) -> None:
            root = ui.site_dir.resolve()
            target = (root / rel).resolve()
            if target.is_dir():
                target = target / "index.html"
            if not target.is_relative_to(root) or not target.is_file():
                self._json(404, {"error": "no results page yet; run something first"})
                return
            ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _events(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            cursor = 0
            try:
                while True:
                    snap = ui.snapshot()
                    for line in snap["log"][cursor:]:
                        self.wfile.write(f"data: {json.dumps({'log': line})}\n\n".encode())
                    cursor = len(snap["log"])
                    if snap["status"] in ("done", "failed"):
                        self.wfile.write(f"event: done\ndata: {json.dumps(snap, default=str)}\n\n".encode())
                        self.wfile.flush()
                        return
                    if snap["status"] == "idle":
                        self.wfile.write(b"event: idle\ndata: {}\n\n")
                        self.wfile.flush()
                        return
                    self.wfile.flush()
                    time.sleep(0.5)
            except (BrokenPipeError, ConnectionResetError):
                return

    return Handler


def serve(*, host: str = "127.0.0.1", port: int = 8765, device_spec: str = "auto", open_browser: bool = True) -> int:
    ui = Ui(device_spec=device_spec)
    server = ThreadingHTTPServer((host, port), make_handler(ui))
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"ptq ui at {url}  (Ctrl-C stops it)")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ptq-bench</title>
<style>
:root{--ink:#1b1b1b;--muted:#5d5c58;--line:#e1e0d9;--bg:#fcfcfb;--accent:#2a78d6;--ok:#1baf7a;--warn:#c98a00;--bad:#d64545}
*{box-sizing:border-box}body{margin:0;font:16px/1.5 -apple-system,"Segoe UI",Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg)}
main{max-width:900px;margin:0 auto;padding:16px}h1{font-size:1.6em;margin:.3em 0}h2{font-size:1.2em;margin-top:1.4em}
.card{border:1px solid var(--line);border-radius:8px;padding:14px 16px;margin:12px 0;background:#fff}
button{font:inherit;padding:8px 14px;border-radius:6px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer}
button.secondary{background:#fff;color:var(--accent)}button:disabled{opacity:.5;cursor:default}
label{display:block;margin:6px 0}select,input[type=number]{font:inherit;padding:4px 6px}
.blurb{color:var(--muted);font-size:.92em}pre#log{background:#f3f3ef;padding:10px;max-height:260px;overflow:auto;white-space:pre-wrap;font-size:.9em}
table{border-collapse:collapse;margin:.6em 0}th,td{border:1px solid var(--line);padding:5px 9px;text-align:left}th{background:#f3f3ef}
.verdict{border-left:4px solid var(--accent);padding:8px 12px;background:#f3f7fd;margin:8px 0}
.ok{color:var(--ok)}.warn{color:var(--warn)}.fail{color:var(--bad)}.info{color:var(--muted)}
details{margin:8px 0}summary{cursor:pointer;color:var(--accent)}#error{color:var(--bad)}
</style></head><body><main>
<h1>ptq-bench</h1>
<p class="blurb">Measure how much a language model loses when its weights are squeezed to fewer bits. Runs on this machine; every result is saved and shows up in <a href="/results/" target="_blank">the results pages</a>.</p>

<div class="card">
  <button id="demo">Show me something in two minutes</button>
  <span class="blurb" id="demo-blurb"></span>
  <button class="secondary" id="doctor" style="float:right">Check this machine</button>
  <div id="doctor-out"></div>
</div>

<div class="card">
  <h2 style="margin-top:0">Or choose yourself</h2>
  <label>Which model? <select id="model"></select></label>
  <div>Which text should it be tested on? <div id="datasets"></div></div>
  <div style="margin-top:8px">Which method? <div id="methods"></div></div>
  <label>How many bits per weight? <select id="bits"></select></label>
  <details><summary>Advanced settings (the defaults are the recommended ones)</summary>
    <label>Grouping <select id="grouping"></select></label>
    <label>GPTQ act-order <select id="act"><option value="default">the family's default (on for Llama-style models)</option><option value="1">on</option><option value="0">off</option></select></label>
    <label>Calibration text <select id="calib"></select></label>
  </details>
  <label><input type="checkbox" id="quick"> Quick preview: 20 passages instead of all of them (faster, approximate, not compared with the papers)</label>
  <button id="run">Run</button> <span class="blurb" id="device"></span>
</div>

<div class="card" id="progress" style="display:none">
  <h2 style="margin-top:0">Running</h2>
  <div id="plan"></div>
  <pre id="log"></pre>
  <div id="error"></div>
</div>

<div class="card" id="results" style="display:none">
  <h2 style="margin-top:0">Result</h2>
  <div id="verdicts"></div>
  <table id="table"></table>
  <p class="blurb">Saved as a result row. <a href="/results/" target="_blank">All results</a> are refreshed.</p>
</div>

<script>
const $ = (id) => document.getElementById(id);
let OPT = null;
async function load() {
  OPT = await (await fetch('/api/options')).json();
  $('device').textContent = 'runs on ' + OPT.device;
  $('demo-blurb').textContent = OPT.demo.join(' → ');
  $('model').innerHTML = OPT.models.map(m => `<option value="${m.key}">${m.key} — ${m.cached ? 'already downloaded' : 'will download'}${m.gated ? ' (gated: an open mirror is used)' : ''}</option>`).join('');
  $('datasets').innerHTML = OPT.datasets.map(d => `<label><input type="checkbox" name="ds" value="${d.key}" ${d.key === 'wikitext2' ? 'checked' : ''}> ${d.name} <span class="blurb">${d.blurb}</span></label>`).join('');
  $('methods').innerHTML = OPT.methods.map(m => `<label><input type="radio" name="algo" value="${m.key}" ${m.key === 'rtn' ? 'checked' : ''}> <b>${m.name}</b> <span class="blurb">${m.blurb}</span></label>`).join('');
  $('bits').innerHTML = OPT.bits.map(b => `<option value="${b.value}" ${b.value === OPT.recommended_bits ? 'selected' : ''}>${b.value}-bit — ${b.blurb}</option>`).join('');
  $('grouping').innerHTML = OPT.grouping.map(g => `<option value="${g.value}">${g.blurb}</option>`).join('');
  $('calib').innerHTML = OPT.calib.map(c => `<option value="${c.key}">${c.key} — ${c.blurb}</option>`).join('');
  $('model').onchange = () => { const m = OPT.models.find(x => x.key === $('model').value); $('grouping').value = m.recommended_group_size; };
  $('model').onchange();
}
function payload() {
  const m = OPT.models.find(x => x.key === $('model').value);
  return {
    model_key: $('model').value,
    datasets: [...document.querySelectorAll('input[name=ds]:checked')].map(e => e.value),
    algo: document.querySelector('input[name=algo]:checked').value,
    bits: +$('bits').value, group_size: +$('grouping').value,
    act_order: $('act').value === 'default' ? null : $('act').value === '1',
    calib_dataset: $('calib').value, quick: $('quick').checked,
  };
}
async function start(body) {
  $('run').disabled = $('demo').disabled = true;
  $('error').textContent = ''; $('log').textContent = ''; $('results').style.display = 'none';
  const r = await fetch('/api/run', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  const j = await r.json();
  $('progress').style.display = '';
  if (!r.ok) { $('error').textContent = j.error; $('run').disabled = $('demo').disabled = false; return; }
  $('plan').innerHTML = j.plan.map(p => `<div><b>Plan:</b> ${p.summary}<br><span class="blurb">estimated time: ${p.estimate}</span></div>`).join('');
  const es = new EventSource('/api/events');
  es.onmessage = (e) => { $('log').textContent += JSON.parse(e.data).log + '\n'; $('log').scrollTop = 1e9; };
  es.addEventListener('done', (e) => { es.close(); finish(JSON.parse(e.data)); });
  es.onerror = () => { es.close(); poll(); };
}
async function poll() {
  const s = await (await fetch('/api/state')).json();
  $('log').textContent = s.log.join('\n');
  if (s.status === 'running') setTimeout(poll, 2000); else finish(s);
}
function finish(s) {
  $('run').disabled = $('demo').disabled = false;
  if (s.status === 'failed') { $('error').textContent = 'The run failed. ' + s.error; return; }
  $('results').style.display = '';
  $('verdicts').innerHTML = s.verdicts.map(v => `<div class="verdict">${v}</div>`).join('');
  const head = '<tr><th>text</th><th>perplexity (lower is better)</th><th>paper\'s number</th><th>vs paper</th><th>passages</th><th>ran</th><th>time</th></tr>';
  $('table').innerHTML = head + s.rows.map(r => `<tr><td>${r.text}</td><td>${r.ppl}</td><td>${r.paper ? r.paper.toFixed(2) + ' (' + r.paper_source + ')' : '—'}</td><td>${r.vs_paper_pct == null ? '—' : (r.vs_paper_pct > 0 ? '+' : '') + r.vs_paper_pct + '%'}</td><td>${r.passages}${r.partial ? ' (preview)' : ''}</td><td>${r.ran}</td><td>${r.seconds}s</td></tr>`).join('');
}
$('demo').onclick = () => start({demo: true});
$('run').onclick = () => start(payload());
$('doctor').onclick = async () => {
  $('doctor-out').innerHTML = '<span class="blurb">checking…</span>';
  const checks = await (await fetch('/api/doctor')).json();
  const sym = {ok: '✔', info: '·', warn: '!', fail: '✘'};
  $('doctor-out').innerHTML = checks.map(c => `<div class="${c.status}">${sym[c.status]} <b>${c.title}:</b> ${c.detail}${c.fix ? ' <span class="blurb">→ ' + c.fix + '</span>' : ''}</div>`).join('');
};
load();
</script>
</main></body></html>
"""
