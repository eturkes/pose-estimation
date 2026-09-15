"""Where the review UI reads from, and what it does when a tree is absent.

Every published tree is gitignored, so a clone carries none of them.  The UI
degrades per source rather than refusing to start: an absent tree turns its view
into a stated gap instead of a traceback, so a clone still runs and states what it
cannot show.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROTOTYPE_DIR = PACKAGE_DIR.parent
STATIC_DIR = PACKAGE_DIR / "static"

#: prototype/review-ui/review_ui/config.py -> repository root.
DEFAULT_REPO = PROTOTYPE_DIR.parent.parent


@dataclass(frozen=True)
class Paths:
    """Resolved data roots.  `REVIEW_UI_REPO` overrides the repository root."""

    repo: Path

    @classmethod
    def resolve(cls, repo: str | os.PathLike[str] | None = None) -> Paths:
        root = Path(repo or os.environ.get("REVIEW_UI_REPO") or DEFAULT_REPO)
        return cls(repo=root.expanduser().resolve())

    @property
    def inventory(self) -> Path:
        return self.repo / "inventory"

    @property
    def sessions(self) -> Path:
        return self.repo / "sessions"

    @property
    def qualification(self) -> Path:
        return self.repo / "qualification"

    @property
    def calibration_qc(self) -> Path:
        return self.repo / "calibration_qc"

    @property
    def cohort(self) -> Path:
        return self.repo / "cohort"

    @property
    def run(self) -> Path:
        return self.repo / "output" / "corpus-2d"

    def status(self) -> dict[str, bool]:
        """Which sources this process can serve."""
        return {
            "inventory": (self.inventory / "census.json").is_file(),
            "sessions": (self.sessions / "events.csv").is_file(),
            "qualification": (self.qualification / "qualification.json").is_file(),
            "calibration_qc": (self.calibration_qc / "calibration_qc.json").is_file(),
            "cohort": (self.cohort / "cohort.json").is_file(),
            "run": (self.run / "run_report.json").is_file(),
        }
