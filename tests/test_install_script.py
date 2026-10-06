"""install.sh picks the torch build from the NVIDIA driver version, and the ./ptq launcher
refuses politely before the environment exists."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess

import pytest

from ptqbench import paths

pytestmark = pytest.mark.smoke
ROOT = paths.repo_root()


def _fake_nvidia_smi(tmp_path, driver: str | None):
    """A PATH whose nvidia-smi prints `driver`, or has no nvidia-smi at all."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("bash", "head", "tr", "sed", "dirname", "readlink", "cat"):
        real = shutil.which(tool)
        if real:
            os.symlink(real, bin_dir / tool)
    if driver is not None:
        script = bin_dir / "nvidia-smi"
        script.write_text(f"#!/usr/bin/env bash\necho '{driver}'\n")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(bin_dir)


@pytest.mark.parametrize("driver,expected", [
    ("595.84", "cu130"), ("580.65.06", "cu130"), ("579.9", "cu126"), ("535.183.01", "cu126"),
    ("470.256", "cpu"), ("garbage", "cpu"), (None, "cpu"),
])
def test_detect_only_picks_the_build_from_the_driver(tmp_path, driver, expected):
    env = {"PATH": _fake_nvidia_smi(tmp_path, driver), "HOME": str(tmp_path)}
    out = subprocess.run(["bash", str(ROOT / "install.sh"), "--detect-only"], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.strip() == expected


def test_flags_override_detection(tmp_path):
    env = {"PATH": _fake_nvidia_smi(tmp_path, "595.84"), "HOME": str(tmp_path)}
    cpu = subprocess.run(["bash", str(ROOT / "install.sh"), "--cpu", "--detect-only"], capture_output=True, text=True, env=env, check=True)
    assert cpu.stdout.strip() == "cpu"
    cu = subprocess.run(["bash", str(ROOT / "install.sh"), "--cuda", "126", "--detect-only"], capture_output=True, text=True, env=env, check=True)
    assert cu.stdout.strip() == "cu126"
    bad = subprocess.run(["bash", str(ROOT / "install.sh"), "--nonsense"], capture_output=True, text=True, env=env, check=False)
    assert bad.returncode == 2 and "unknown option" in bad.stderr


def test_launcher_refuses_before_install(tmp_path):
    fake_root = tmp_path / "repo"
    (fake_root / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "ptq", fake_root / "ptq")
    out = subprocess.run(["bash", str(fake_root / "ptq"), "doctor"], capture_output=True, text=True, check=False)
    assert out.returncode == 1 and "run ./install.sh first" in out.stderr
