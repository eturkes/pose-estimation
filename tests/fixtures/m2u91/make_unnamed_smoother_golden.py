"""Regenerate `unnamed_smoother_golden.npz`: the pre-M2.9.1 KeypointSmoother on a fixed stimulus.

M2.9.1 D07 promises `track_ids=None` leaves smoothing unchanged.  Comparing two instances
of the new code cannot see a regression, so the oracle is the base commit's own class,
read from git and run on the stimulus `test_m2u91_subject_tracker.py` replays.

    python tests/fixtures/m2u91/make_unnamed_smoother_golden.py
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

BASE = "ff117de"
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_name("unnamed_smoother_golden.npz")
K = 133
FRAMES = 30


def stimulus():
    """Two people with seeded noise; every 7th frame from index 4 is empty."""
    rng = np.random.default_rng(209109)
    index = np.arange(K, dtype=np.float64)

    def person(x: float) -> np.ndarray:
        return np.column_stack((x + index % 11, 200.0 + index % 13))

    for frame in range(FRAMES):
        points = np.stack([person(100.0), person(500.0)]) + rng.normal(0, 3, (2, K, 2))
        scores = rng.uniform(0.5, 1.0, (2, K))
        if frame % 7 == 4:
            yield frame, None, None
        else:
            yield frame, points, scores


def base_smoother():
    source = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{BASE}:src/pose_estimation/rtmlib_smoothing.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "base_rtmlib_smoothing.py"
        path.write_text(source, encoding="utf-8")
        # The package-qualified name makes the relative imports resolve in pose_estimation.
        spec = importlib.util.spec_from_file_location("pose_estimation._base_smoothing", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load the {BASE} smoother")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.KeypointSmoother()


def main() -> int:
    smoother = base_smoother()
    arrays: dict[str, np.ndarray] = {}
    known: dict[int, int] = {}
    for frame, points, scores in stimulus():
        kps, sc = smoother(points, scores, frame / 30)
        keys = [known.setdefault(key, len(known)) for key in smoother.output_track_keys()]
        arrays[f"keys_{frame}"] = np.array(keys, dtype=np.int64)
        if kps is not None:
            arrays[f"kps_{frame}"], arrays[f"sc_{frame}"] = kps, sc
    np.savez_compressed(OUT, allow_pickle=False, **arrays)
    print(f"wrote {OUT.name}: {len(arrays)} arrays from {BASE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
