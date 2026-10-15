"""`.agent/spec-lint` grades `.agent/spec.md` shape inside the decisive gate.

The lint is a template file, copied in whole at each refresh; each planted spec is its shipped
firing input.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
LINT = ROOT / ".agent" / "spec-lint"
SPEC = ROOT / ".agent" / "spec.md"


def _lint(spec: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["sh", str(LINT), str(spec)], capture_output=True, text=True, check=False)


def _phase_history(text: str) -> str:
    return text.rstrip("\n") + "\nThe phase opened after the last refresh.\n"


def _stray_tasks_line(text: str) -> str:
    assert text.count("\n## Tasks\n\n") == 1
    return text.replace("\n## Tasks\n\n", "\n## Tasks\n\nA note that is not a row.\n")


def _bold_phase_line(text: str) -> str:
    assert text.count("\n## Phase\n\n") == 1
    return text.replace("\n## Phase\n\n", "\n## Phase\n\n**")


def test_live_spec_lints_clean() -> None:
    result = _lint(SPEC)
    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")


@pytest.mark.parametrize(
    ("plant", "finding"),
    [
        (_phase_history, "line(s) of history"),
        (_stray_tasks_line, "Tasks row ="),
        (_bold_phase_line, "Phase line 1 ="),
    ],
)
def test_planted_spec_defect_fails(plant, finding: str, tmp_path: pathlib.Path) -> None:
    planted = tmp_path / "spec.md"
    planted.write_text(plant(SPEC.read_text(encoding="utf-8")), encoding="utf-8")
    result = _lint(planted)
    assert result.returncode == 1
    assert finding in result.stdout
