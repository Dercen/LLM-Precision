"""`ptq doctor`: every check returns a Check, red lines carry a fix, and the exit code says
whether anything is red."""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from ptqbench import doctor as Dr

pytestmark = pytest.mark.smoke


def test_every_check_runs_and_is_well_formed(monkeypatch):
    monkeypatch.setattr(Dr, "_internet_ok", lambda timeout=4.0: False)
    checks = Dr.run_checks()
    assert [c.title for c in checks][:3] == ["Environment", "GPU", "GPU arithmetic"]
    assert len(checks) == len(Dr.CHECKS)
    for c in checks:
        assert c.status in (Dr.OK, Dr.INFO, Dr.WARN, Dr.FAIL)
        assert c.title and c.detail
        if c.status == Dr.FAIL:
            assert c.fix, f"{c.title} is red without a fix"
    internet = next(c for c in checks if c.title == "Internet")
    assert internet.status == Dr.WARN and "prefetch" in internet.fix
    assert next(c for c in checks if c.title == "Environment").status == Dr.OK


def test_offline_skips_the_probe(monkeypatch):
    def boom(timeout=4.0):
        raise AssertionError("must not probe")

    monkeypatch.setattr(Dr, "_internet_ok", boom)
    internet = next(c for c in Dr.run_checks(probe_internet=False) if c.title == "Internet")
    assert internet.status == Dr.INFO and "--offline" in internet.detail


def test_a_crashing_check_is_reported_not_fatal(monkeypatch):
    def broken():
        raise RuntimeError("disk fell off")

    monkeypatch.setattr(Dr, "CHECKS", [broken, Dr.check_results])
    checks = Dr.run_checks(probe_internet=False)
    assert checks[0].status == Dr.WARN and "disk fell off" in checks[0].detail
    assert checks[1].title == "Results so far"


def _render(checks) -> tuple[int, str]:
    buf = io.StringIO()
    rc = Dr.print_report(checks, console=Console(file=buf, width=120, force_terminal=False))
    return rc, buf.getvalue()


def test_report_exit_code_and_wording():
    rc, text = _render([Dr.Check(Dr.OK, "GPU", "fine"), Dr.Check(Dr.INFO, "Results so far", "none yet")])
    assert rc == 0 and "All good" in text and "./ptq opens the menu" in text
    rc, text = _render([Dr.Check(Dr.WARN, "Internet", "offline", "connect")])
    assert rc == 0 and "1 note(s)" in text and "→ connect" in text
    rc, text = _render([Dr.Check(Dr.FAIL, "Environment", "wrong python", "run ./install.sh")])
    assert rc == 1 and "1 problem(s) to fix first" in text and "run ./install.sh" in text


def test_cli_doctor_offline():
    from ptqbench import cli

    assert cli.main(["doctor", "--offline"]) in (0, 1)
