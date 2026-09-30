"""M2.9.4 reviewer S08: scope the filtering claim to the omitted extra stage."""

from pathlib import Path

import numpy as np

from pose_estimation import rtmlib_smoothing

_ROOT = Path(__file__).resolve().parents[1]


def test_s08_no_lowpass_claim_excludes_the_existing_smoother() -> None:
    assert Path(rtmlib_smoothing.__file__).resolve().is_relative_to(_ROOT)
    smoother = rtmlib_smoothing.KeypointSmoother(rest_cutoff=0.05, hand_rest_cutoff=0.15)
    still = np.full((1, 133, 2), 200.0)
    scores = np.ones((1, 133))
    moved = still.copy()
    moved[0, 9, 0] = 210.0
    smoother(still, scores, 0.0, track_ids=[7])
    filtered, _ = smoother(moved, scores, 1 / 30, track_ids=[7])
    assert filtered is not None
    wrist_x = float(filtered[0, 9, 0])
    assert 200.0 < wrist_x < 210.0, "Probe must witness the live smoother's attenuation"

    document = _ROOT / "docs" / "technical" / "analysis.md"
    claim = "the pipeline runs no low-pass stage ahead of it."
    assert claim not in document.read_text(encoding="utf-8"), (
        f"{document}: blanket no-filter claim contradicts the shipped KeypointSmoother; "
        f"synthetic wrist raw=210.0, filtered={wrist_x:.9f}. "
        "Scope the omission to an additional post-hoc/fixed-cutoff low-pass stage."
    )
