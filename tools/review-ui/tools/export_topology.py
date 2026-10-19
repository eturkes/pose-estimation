#!/usr/bin/env python3
"""Export the pipeline's own skeleton topology and palette for the browser overlay.

The overlay must draw the skeleton the pipeline draws, so the chains, segments
and colours come out of `pose_estimation.drawing` rather than being retyped in
JavaScript.  Colours there are BGR (OpenCV); the browser gets hex RGB.

Regenerate from the repository root, which is where `pose_estimation` resolves::

    env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync \
        python tools/review-ui/tools/export_topology.py
"""

from __future__ import annotations

import json
from pathlib import Path

from pose_estimation import drawing, export

OUT = Path(__file__).resolve().parents[1] / "review_ui" / "static" / "topology.json"


def _hex(bgr: tuple[int, int, int]) -> str:
    blue, green, red = bgr
    return f"#{red:02x}{green:02x}{blue:02x}"


def _chains(chains) -> list[dict]:
    return [{"points": list(points), "group": group} for points, group in chains]


def _segments(segments) -> list[dict]:
    return [{"points": list(pair), "group": group} for pair, group in segments]


def main() -> int:
    payload = {
        "source": "pose_estimation.drawing",
        "coord_normalization": export.COORD_NORMALIZATION,
        "body": {
            "names": list(export.BODY_KEYPOINT_NAMES),
            "chains": _chains(drawing.FULL_BODY_CHAINS),
            "segments": _segments(drawing.FULL_BODY_SEGMENTS),
            "colors": {k: _hex(v) for k, v in drawing.FULL_BODY_COLOR_MAP.items()},
        },
        "arm": {
            "names": list(export.ARM_KEYPOINT_NAMES),
            "chains": _chains(drawing.BODY_CHAINS),
            "segments": _segments(drawing.BODY_SEGMENTS),
            "colors": {k: _hex(v) for k, v in drawing.BODY_COLOR_MAP.items()},
        },
        "hand": {
            "count": export.HAND_KEYPOINT_COUNT,
            "chains": _chains(drawing.HAND_CHAINS),
            "segments": _segments(drawing.HAND_SEGMENTS),
            "colors": {k: _hex(v) for k, v in drawing.HAND_COLOR_MAP.items()},
            "point_colors": [_hex(c) for c in drawing.HAND_KEYPOINT_COLORS],
        },
        "bridge_color": _hex(drawing.BRIDGE_COLOR),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
