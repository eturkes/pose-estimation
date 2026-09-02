#!/usr/bin/env python
"""Reproduce the ``rtmlib.PoseTracker`` state freeze that M2.8.2 D01 disables.

Every number in ``.agent/archive/contract-m2u82.md`` §2 re-derives from this script.
It uses stub detector and pose models, so it needs no corpus, no model weights and no
accelerator -- the defect is in the tracker's state machine, not in inference.

The defect: ``PoseTracker.__call__`` reorders the CURRENT frame's keypoints by
PERSISTENT track id.  ``track_by_iou`` mints ``track_id = next_id++`` for any unmatched
box above ``MIN_AREA``, so one missed IoU match indexes a one-person keypoint array at
``[1]`` and raises ``IndexError`` on a path that returns before ``frame_cnt += 1`` and
before ``bboxes_last_frame`` is replaced.  Both freeze for the rest of the source, and
the residue of the frozen counter decides which failure you get.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from rtmlib import PoseTracker

# Latency of one detector call and one pose call, milliseconds.  Measured on this
# machine at the shipped configuration (det CPU / pose NPU) and used only to turn call
# counts into the per-frame cost the corpus pilot reported, so the two are comparable.
DET_MS = 350.0
POSE_MS = 8.0


class _StubDetector:
    """Return one large person box per call and count the calls."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, _image: np.ndarray) -> list[list[float]]:
        self.calls += 1
        return [[0.0, 0.0, 300.0, 400.0, 0.9]]


class _StubPose:
    """One person per box, jumping once so exactly one IoU match fails.

    Keypoints span an area above ``PoseTracker.MIN_AREA`` because the tracker builds its
    boxes from ``pose_to_bbox(keypoints)`` rather than from the detector, and a box under
    that floor is dropped instead of being assigned a new track id.
    """

    def __init__(self, jump_at: int) -> None:
        self.calls = 0
        self.jump_at = jump_at
        self.empty_bbox_calls = 0

    def __call__(self, _image: np.ndarray, bboxes: list | None = None) -> tuple:
        boxes = list(bboxes) if bboxes is not None else []
        self.calls += 1
        if not boxes:
            # RTMPose substitutes the WHOLE FRAME here, so a starved box list silently
            # turns a top-down model into a full-frame estimator.
            self.empty_bbox_calls += 1
        n = max(len(boxes), 1)
        x = 900.0 if self.calls == self.jump_at else 0.0
        keypoints = np.tile(np.array([[x, 0.0], [x + 150.0, 350.0]]), (n, 1, 1))
        return keypoints, np.ones((n, 2))


def _tracker(*, jump_at: int, tracking: bool, det_frequency: int) -> PoseTracker:
    """Build a tracker over stubs without running ``__init__``'s model loading."""
    tracker = object.__new__(PoseTracker)
    tracker.det_model = _StubDetector()
    tracker.pose_model = _StubPose(jump_at)
    tracker.det_categories = None
    tracker.det_mode = None
    tracker.det_frequency = det_frequency
    tracker.tracking = tracking
    tracker.tracking_thr = 0.3
    tracker.reset()
    return tracker


def run_scenario(*, jump_at: int, tracking: bool, frames: int, det_frequency: int) -> dict:
    """Drive ``frames`` stub frames and report the tracker's end state."""
    tracker = _tracker(jump_at=jump_at, tracking=tracking, det_frequency=det_frequency)
    image = np.zeros((480, 640, 3), np.uint8)
    for _ in range(frames):
        tracker(image)
    det_calls = tracker.det_model.calls
    return {
        "tracking": tracking,
        "jump_at": jump_at,
        "frames": frames,
        "frame_cnt": tracker.frame_cnt,
        "frozen": tracker.frame_cnt < frames,
        "det_calls": det_calls,
        "expected_det_calls": math.ceil(frames / det_frequency),
        "whole_frame_pose_calls": tracker.pose_model.empty_bbox_calls,
        "next_id": tracker.next_id,
        "modelled_ms_per_frame": round((det_calls * DET_MS + frames * POSE_MS) / frames, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=140)
    parser.add_argument("--det-frequency", type=int, default=7)
    args = parser.parse_args()

    # ``jump_at`` is the pose call on which the person jumps.  Its value selects the
    # residue of the frozen counter, which is what selects the failure mode.
    scenarios = [
        ("healthy", 0, True),
        ("frozen_residue_nonzero", 4, True),
        ("frozen_residue_zero", 8, True),
        ("fixed_residue_nonzero", 4, False),
        ("fixed_residue_zero", 8, False),
    ]
    rows = []
    for name, jump_at, tracking in scenarios:
        row = run_scenario(
            jump_at=jump_at,
            tracking=tracking,
            frames=args.frames,
            det_frequency=args.det_frequency,
        )
        row["scenario"] = name
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    healthy = next(r for r in rows if r["scenario"] == "healthy")
    frozen = [r for r in rows if r["scenario"].startswith("frozen_")]
    fixed = [r for r in rows if r["scenario"].startswith("fixed_")]

    verdicts = {
        # tracking=True survives only while every IoU match lands.
        "healthy_advances": not healthy["frozen"],
        "tracking_true_freezes_on_one_miss": all(r["frozen"] for r in frozen),
        # The two frozen residues are the two measured corpus bands.
        "starved_residue_runs_whole_frame_pose": any(
            r["whole_frame_pose_calls"] > 0 for r in frozen
        ),
        "detecting_residue_runs_detector_every_frame": any(
            r["det_calls"] > r["expected_det_calls"] for r in frozen
        ),
        # tracking=False is the fix: no freeze, correct cadence, no whole-frame pose.
        "tracking_false_never_freezes": all(not r["frozen"] for r in fixed),
        "tracking_false_holds_detector_cadence": all(
            r["det_calls"] == r["expected_det_calls"] for r in fixed
        ),
        "tracking_false_never_starves_bboxes": all(r["whole_frame_pose_calls"] == 0 for r in fixed),
    }
    print(json.dumps({"verdicts": verdicts}, sort_keys=True))
    failed = sorted(k for k, ok in verdicts.items() if not ok)
    if failed:
        print(json.dumps({"failed_verdicts": failed}, sort_keys=True))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
