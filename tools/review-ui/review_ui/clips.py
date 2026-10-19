"""Clip index and per-clip landmark series for the player view.

The index joins the three published tables that already key one another —
`sessions/events.csv`, `sessions/placements.csv`, `inventory/assets.csv` — and the
run manifest that gives every canonical asset exactly one disposition.  A clip is
addressed by `(event_id, camera_name)`, the same key the run tree and the
diagnostics rows use, and a request resolves only through the built index, so no
path in a request ever reaches the filesystem.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import Paths

#: Landmark CSVs carry no `z` for a 2D run: it exports identically 0.0.
BODY_SUFFIXES = ("_x", "_y", "_vis")
HAND_SUFFIXES = ("_x", "_y", "_conf")
MEDIA_SUFFIXES = (".mov", ".MOV", ".mp4", ".MP4", ".m4v", ".mkv", ".avi")


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _number(text: str | None, cast: type) -> Any:
    try:
        return cast(text)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _media_path(directory: Path, stem: str) -> Path | None:
    for suffix in MEDIA_SUFFIXES:
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


@lru_cache(maxsize=4)
def index(paths: Paths) -> list[dict[str, Any]]:
    """One row per placed camera artifact, ordered by task, side, event, camera."""
    clips: list[dict[str, Any]] = []
    events = {row["event_id"]: row for row in _rows(paths.sessions / "events.csv")}
    assets = {row["asset_id"]: row for row in _rows(paths.inventory / "assets.csv")}
    manifest = {row["asset_id"]: row for row in _rows(paths.run / "run_manifest.csv")}

    for placement in _rows(paths.sessions / "placements.csv"):
        event_id = placement.get("event_id") or ""
        camera = placement.get("camera_name") or ""
        if not event_id or not camera:
            continue
        event = events.get(event_id, {})
        asset = assets.get(placement.get("asset_id", ""), {})
        directory = paths.sessions / event_id
        clips.append(
            {
                "event_id": event_id,
                "camera_name": camera,
                "subject_ordinal": _number(event.get("subject_ordinal"), int),
                "task": event.get("task") or asset.get("task"),
                "side": event.get("side") or asset.get("side"),
                "view": asset.get("view"),
                "run_index": _number(event.get("run_index"), int),
                "n_cameras": _number(event.get("n_cameras"), int),
                "view_conflict": event.get("view_conflict", ""),
                "width": _number(asset.get("reported_width"), int),
                "height": _number(asset.get("reported_height"), int),
                "fps": _number(asset.get("reported_avg_fps"), float),
                "frames": _number(asset.get("reported_frame_count"), int),
                "duration_s": _number(asset.get("nominal_duration_s"), float),
                "rotation_deg": _number(asset.get("reported_rotation_deg"), int),
                "codec": asset.get("reported_fourcc"),
                "disposition": manifest.get(placement.get("asset_id", ""), {}).get("disposition"),
                "has_video": _media_path(directory, camera) is not None,
                "has_landmarks": (paths.run / event_id / f"{camera}.csv").is_file(),
            }
        )
    clips.sort(
        key=lambda clip: (
            str(clip["task"]),
            str(clip["side"]),
            clip["event_id"],
            clip["camera_name"],
        )
    )
    _number_families(clips)
    return clips


def _number_families(clips: list[dict[str, Any]]) -> None:
    """Give every recording event a display ordinal shared by its views.

    Task and side alone repeat across the whole corpus, so a list keyed on them
    shows hundreds of identical labels and hides the one structure that matters
    here — which rows are the two or three views of the *same* event.  The
    ordinal is positional, so the list stays free of `event_id`, which is
    patient-adjacent and would otherwise reach a committed screenshot.
    """
    numbers: dict[str, int] = {}
    for clip in clips:
        if clip["event_id"] not in numbers:
            numbers[clip["event_id"]] = len(numbers) + 1
        clip["family_no"] = numbers[clip["event_id"]]


def find(paths: Paths, event_id: str, camera_name: str) -> dict[str, Any] | None:
    for clip in index(paths):
        if clip["event_id"] == event_id and clip["camera_name"] == camera_name:
            return clip
    return None


def video_path(paths: Paths, clip: dict[str, Any]) -> Path | None:
    return _media_path(paths.sessions / clip["event_id"], clip["camera_name"])


def landmark_path(paths: Paths, clip: dict[str, Any]) -> Path:
    return paths.run / clip["event_id"] / f"{clip['camera_name']}.csv"


def _column_plan(header: list[str]) -> dict[str, Any]:
    """Derive the landmark layout from the header itself.

    The shipped schema names every column `body_<keypoint>_<axis>` and
    `<side>_hand_<i>_<axis>`, so the keypoint order, the tracking mode and the
    hand count all read off the header.  Deriving them keeps a rename visible
    instead of silently shifting a transcribed index.
    """
    prefix = "body" if any(name.startswith("body_") for name in header) else "arm"
    body_names = [
        name[len(prefix) + 1 : -2]
        for name in header
        if name.startswith(f"{prefix}_") and name.endswith("_x")
    ]
    hands: dict[str, list[int]] = {}
    for side in ("left", "right"):
        indices = sorted(
            int(name[len(side) + 6 : -2])
            for name in header
            if name.startswith(f"{side}_hand_") and name.endswith("_x")
        )
        hands[side] = indices
    return {"prefix": prefix, "body_names": body_names, "hands": hands}


def _series(row: dict[str, str], keys: list[str]) -> list[float | None]:
    out: list[float | None] = []
    for key in keys:
        raw = row.get(key, "")
        if raw == "" or raw is None:
            out.append(None)
            continue
        try:
            out.append(round(float(raw), 5))
        except ValueError:
            out.append(None)
    return out


def landmarks(paths: Paths, clip: dict[str, Any]) -> dict[str, Any]:
    """Per-frame landmark series for one clip, one person per frame.

    Coordinates stay exactly as exported — normalised by one scalar,
    `max(frame_w, frame_h)` — so the browser applies the same similarity map the
    features were computed under rather than a second, different one.
    """
    path = landmark_path(paths, clip)
    if not path.is_file():
        return {"clip": clip, "frames": 0, "reason": "no_landmark_csv"}

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        header = list(reader.fieldnames or ())
        plan = _column_plan(header)
        prefix = plan["prefix"]
        body_keys = [
            f"{prefix}_{name}{suffix}" for name in plan["body_names"] for suffix in BODY_SUFFIXES
        ]
        hand_keys = {
            side: [
                f"{side}_hand_{i}{suffix}" for i in plan["hands"][side] for suffix in HAND_SUFFIXES
            ]
            for side in ("left", "right")
        }
        by_frame: dict[int, dict[str, str]] = {}
        persons_max = 0
        for row in reader:
            frame = _number(row.get("frame_idx"), int)
            person = _number(row.get("person_idx"), int) or 0
            if frame is None:
                continue
            persons_max = max(persons_max, person + 1)
            if frame not in by_frame or person < int(by_frame[frame].get("person_idx") or 0):
                by_frame[frame] = row

    order = sorted(by_frame)
    scale = max(clip.get("width") or 0, clip.get("height") or 0) or None
    return {
        "clip": clip,
        "frames": len(order),
        "frame_idx": order,
        "timestamp_sec": [_number(by_frame[f].get("timestamp_sec"), float) for f in order],
        "persons_max": persons_max,
        "scale": scale,
        "body_names": plan["body_names"],
        "hand_points": len(plan["hands"]["left"]),
        "body": [_series(by_frame[f], body_keys) for f in order],
        "hand_left": [_series(by_frame[f], hand_keys["left"]) for f in order],
        "hand_right": [_series(by_frame[f], hand_keys["right"]) for f in order],
    }
