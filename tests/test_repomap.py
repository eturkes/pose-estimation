"""Drift guard for the agent repo map (``.claude/repomap.md``).

``scripts/repomap.py`` generates a grep-able symbol index that agents use to
jump straight to a definition instead of reading a whole module. This test
fails if the committed map is stale, mirroring ``test_public_api.py``'s role
for the public API surface — drift is this repo's documented #1 failure mode.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "repomap.py"
MAP = ROOT / ".claude" / "repomap.md"


def test_committed_map_is_current():
    """The committed map matches freshly generated output (regenerate if this fails)."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "repomap.md is stale — run: python scripts/repomap.py\n" + result.stderr
    )


def test_map_indexes_usable_jump_targets():
    """The committed map carries real path:line targets for Python and R."""
    text = MAP.read_text(encoding="utf-8")
    # A re-exported Python function (stable public API).
    assert "def process_frame(" in text
    assert "src/pose_estimation/processing.py:" in text
    # Qualified class methods are greppable as Class.method.
    assert "def OneEuroFilter." in text
    # R analysis functions are indexed too.
    assert "analysis/clinical_features.R:" in text
