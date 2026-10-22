"""Checker negative controls reach the shipped entrypoints, without touching source inputs."""

from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_RUNNER = _ROOT / "scripts" / "check_seeds.py"
_SPEC = _ROOT / "docs" / "prospective_capture.md"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_RUNNER), *args],
        cwd=_ROOT,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )


def test_every_checker_refuses_its_seed() -> None:
    before = hashlib.sha256(_SPEC.read_bytes()).hexdigest()
    done = _run()
    assert done.returncode == 0, done.stdout + done.stderr
    checkers = sorted(path.stem for path in _RUNNER.parent.glob("check_*.py") if path != _RUNNER)
    for checker in checkers:
        assert f"{checker} REFUSED " in done.stdout
    assert f"{len(checkers)}/{len(checkers)} refused" in done.stdout
    assert "controls_firing 8 of 8" in done.stdout
    assert f"prospective_capture sha256 before={before} after={before}" in done.stdout
    assert hashlib.sha256(_SPEC.read_bytes()).hexdigest() == before


def test_inert_control_fails_and_names_the_checker() -> None:
    before = _SPEC.read_bytes()
    done = _run("--inert", "check_prospective_capture:NC4")
    assert done.returncode == 1, done.stdout + done.stderr
    assert "check_prospective_capture NC4 SEED-INERT" in done.stdout
    assert "check_prospective_capture SEED-INERT" in done.stdout
    assert "11/12 refused" in done.stdout
    assert "controls_firing 7 of 8" in done.stdout
    assert _SPEC.read_bytes() == before
