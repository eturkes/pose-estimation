"""User labelling page: wrist, fingertip and arm points on the benchmark's confirmation frames.

`scripts/bench.py user-sample` writes the job list and one JPEG per job under
`.scratch/bench/user/`; this module serves those frames by job number and stores one
codebook-v2 record per job in `labels.jsonl` beside them. The browser never sees an event id:
requests carry the job number, and the frame key is joined server-side. This is the one place
the review UI writes, and it writes only there.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Paths

SIDES = ("L", "R")
HAND_JOINTS = ("wrist", "thumb_tip", "index_mcp", "index_tip", "little_tip")
BODY_JOINTS = ("shoulder", "elbow", "wrist")
JOINT_STATES = ("vis", "occ", "oof", "?")


def _root(paths: Paths) -> Path:
    return paths.repo / ".scratch" / "bench" / "user"


def jobs(paths: Paths) -> list[dict[str, Any]]:
    path = _root(paths) / "jobs.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


def frame_path(paths: Paths, job: int) -> Path | None:
    path = _root(paths) / "frames" / f"{job:03d}.jpg"
    return path if path.is_file() else None


def _key(meta: dict[str, Any]) -> str:
    return f"{meta['event_id']}/{meta['camera_name']}#{meta['frame']}"


def _records(paths: Paths) -> dict[str, dict[str, Any]]:
    """Stored codebook-v2 records by frame key."""
    path = _root(paths) / "labels.jsonl"
    if not path.is_file():
        return {}
    records = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return {r["frame"]["key"]: r for r in records}


def summary(paths: Paths) -> dict[str, Any]:
    stored = _records(paths)
    return {"n": len(jobs(paths)), "done": [j["job"] for j in jobs(paths) if _key(j) in stored]}


def answer(paths: Paths, job: int) -> dict[str, Any] | None:
    """The browser's view of a stored record: one entry per `side.part.joint`."""
    meta = next((j for j in jobs(paths) if j["job"] == job), None)
    record = None if meta is None else _records(paths).get(_key(meta))
    if record is None:
        return None
    points = {}
    for side, data in record["sides"].items():
        for part in ("joints", "body"):
            for name, joint in data[part].items():
                entry = {"state": joint["state"]}
                if "xy" in joint:
                    entry["x"], entry["y"] = joint["xy"]
                points[f"{side}.{'hand' if part == 'joints' else 'body'}.{name}"] = entry
    return {"task": record["task"], "points": points}


def _hand_state(states: list[str]) -> str:
    if all(s == "vis" for s in states):
        return "vis"
    if "vis" in states:
        return "part"
    if states and all(s == "oof" for s in states):
        return "oof"
    if "occ" in states or "oof" in states:
        return "occ"
    return "?"


def _joint(point: dict[str, Any] | None) -> dict[str, Any]:
    if not point or point.get("state") not in JOINT_STATES:
        return {"state": "?"}
    if point["state"] == "vis":
        return {"state": "vis", "xy": [round(float(point["x"]), 1), round(float(point["y"]), 1)]}
    return {"state": point["state"]}


def save(paths: Paths, job: int, payload: dict[str, Any], width: int, height: int) -> None:
    """Store the job's codebook-v2 record, replacing an earlier one."""
    listed = {j["job"]: j for j in jobs(paths)}
    if job not in listed:
        raise KeyError(job)
    meta = listed[job]
    points = payload.get("points", {})
    sides = {}
    for side in SIDES:
        hand = {name: _joint(points.get(f"{side}.hand.{name}")) for name in HAND_JOINTS}
        # One wrist click serves both conventions: the arm's wrist is the hand's root.
        arm = {**points, f"{side}.body.wrist": points.get(f"{side}.hand.wrist")}
        body = {name: _joint(arm.get(f"{side}.body.{name}")) for name in BODY_JOINTS}
        state = _hand_state([j["state"] for j in hand.values()])
        visible = [j["xy"] for j in hand.values() if "xy" in j]
        extent = None
        if state in ("vis", "part") and visible:
            xs, ys = [p[0] for p in visible], [p[1] for p in visible]
            pad = 0.1 * max(max(xs) - min(xs), max(ys) - min(ys), 10.0)
            extent = [
                round(max(0.0, min(xs) - pad), 1),
                round(max(0.0, min(ys) - pad), 1),
                round(min(float(width), max(xs) + pad), 1),
                round(min(float(height), max(ys) + pad), 1),
            ]
        sides[side] = {"hand": state, "extent": extent, "joints": hand, "body": body}
    record = {
        "frame": {"key": _key(meta), "index": int(meta["frame"]), "width": width, "height": height},
        "labeller": "user",
        "task": payload.get("task") if payload.get("task") in ("task", "setup", "?") else "?",
        "patient": None,
        "others": [],
        "sides": sides,
    }
    records = _records(paths)
    records[_key(meta)] = record
    path = _root(paths) / "labels.jsonl"
    path.write_text(
        "".join(json.dumps(records[k], sort_keys=True) + "\n" for k in sorted(records)),
        encoding="utf-8",
    )
