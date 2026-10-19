"""Codebook v2 — model-agnostic labels on raw frames (contract M4.1 D01).

A record = one frame, labelled with no model output in view. Coordinates are source-frame pixels.
`validate` names every defect instead of stopping at the first, so a labeller fixes a file in one
pass.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

TASK_STATES = ("task", "setup", "?")
HAND_STATES = ("vis", "part", "occ", "oof", "?")
JOINT_STATES = ("vis", "occ", "oof", "?")
OWNERS = ("patient", "other", "ambig")
SIDES = ("L", "R")
#: Hand joints the clinical features read, by COCO-WholeBody hand index (0-20).
HAND_JOINTS = {"wrist": 0, "thumb_tip": 4, "index_mcp": 5, "index_tip": 8, "little_tip": 20}
BODY_JOINTS = ("shoulder", "elbow", "wrist")


def _box_defects(box: Any, width: int, height: int, where: str) -> list[str]:
    if not (isinstance(box, list) and len(box) == 4 and all(_is_number(v) for v in box)):
        return [f"{where}: box must be [x0, y0, x1, y1]"]
    x0, y0, x1, y1 = box
    out = []
    if not (x1 > x0 and y1 > y0):
        out.append(f"{where}: box has non-positive area")
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        out.append(f"{where}: box leaves the frame")
    return out


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _joint_defects(joint: Any, width: int, height: int, where: str) -> list[str]:
    if not isinstance(joint, dict):
        return [f"{where}: joint must be an object"]
    out = []
    state = joint.get("state")
    if state not in JOINT_STATES:
        out.append(f"{where}: unknown state {state!r}")
    xy = joint.get("xy")
    if state == "vis":
        if not (isinstance(xy, list) and len(xy) == 2 and all(_is_number(v) for v in xy)):
            out.append(f"{where}: vis joint needs xy")
        elif not (0 <= xy[0] < width and 0 <= xy[1] < height):
            out.append(f"{where}: xy outside the frame")
    elif xy is not None:
        out.append(f"{where}: xy on a {state} joint")
    if "own" in joint and joint["own"] not in OWNERS:
        out.append(f"{where}: unknown owner {joint['own']!r}")
    extra = set(joint) - {"state", "xy", "own"}
    if extra:
        out.append(f"{where}: unknown keys {sorted(extra)}")
    return out


def _side_defects(side: Any, width: int, height: int, where: str) -> list[str]:
    if not isinstance(side, dict):
        return [f"{where}: side must be an object"]
    out = []
    hand = side.get("hand")
    if hand not in HAND_STATES:
        out.append(f"{where}.hand: unknown state {hand!r}")
    extent = side.get("extent")
    if hand in ("vis", "part"):
        if extent is None:
            out.append(f"{where}.extent: required for a {hand} hand")
        else:
            out.extend(_box_defects(extent, width, height, f"{where}.extent"))
    elif extent is not None:
        out.append(f"{where}.extent: set on a {hand} hand")
    joints = side.get("joints", {})
    if not isinstance(joints, dict):
        return [*out, f"{where}.joints: must be an object"]
    unknown = set(joints) - set(HAND_JOINTS)
    if unknown:
        out.append(f"{where}.joints: unknown joints {sorted(unknown)}")
    if hand in ("vis", "part"):
        missing = set(HAND_JOINTS) - set(joints)
        if missing:
            out.append(f"{where}.joints: missing {sorted(missing)}")
    for name, joint in joints.items():
        if name in HAND_JOINTS:
            out.extend(_joint_defects(joint, width, height, f"{where}.joints.{name}"))
    states = [j.get("state") for j in joints.values() if isinstance(j, dict)]
    if hand == "vis" and any(s in ("occ", "oof") for s in states):
        out.append(f"{where}: vis hand with a hidden joint (use part)")
    if hand == "part" and "vis" not in states:
        out.append(f"{where}: part hand with no visible joint")
    if hand in ("occ", "oof") and "vis" in states:
        out.append(f"{where}: {hand} hand with a visible joint")
    body = side.get("body", {})
    if not isinstance(body, dict):
        return [*out, f"{where}.body: must be an object"]
    unknown = set(body) - set(BODY_JOINTS)
    if unknown:
        out.append(f"{where}.body: unknown joints {sorted(unknown)}")
    missing = set(BODY_JOINTS) - set(body)
    if missing:
        out.append(f"{where}.body: missing {sorted(missing)}")
    for name in BODY_JOINTS:
        if name in body:
            out.extend(_joint_defects(body[name], width, height, f"{where}.body.{name}"))
    extra = set(side) - {"hand", "extent", "joints", "body"}
    if extra:
        out.append(f"{where}: unknown keys {sorted(extra)}")
    return out


def validate(record: Any) -> list[str]:
    """Every defect of one label record; empty = valid."""
    if not isinstance(record, dict):
        return ["record must be an object"]
    frame = record.get("frame")
    if not (
        isinstance(frame, dict)
        and isinstance(frame.get("key"), str)
        and frame["key"]
        and isinstance(frame.get("index"), int)
        and frame["index"] >= 0
        and isinstance(frame.get("width"), int)
        and isinstance(frame.get("height"), int)
        and frame["width"] > 0
        and frame["height"] > 0
    ):
        return ["frame: needs key, index >= 0, width > 0, height > 0"]
    width, height = frame["width"], frame["height"]
    out = []
    if not isinstance(record.get("labeller"), str) or not record["labeller"]:
        out.append("labeller: required")
    if record.get("task") not in TASK_STATES:
        out.append(f"task: unknown state {record.get('task')!r}")
    patient = record.get("patient")
    if patient is not None:
        out.extend(_box_defects(patient, width, height, "patient"))
    others = record.get("others", [])
    if not isinstance(others, list):
        out.append("others: must be a list of boxes")
    else:
        for i, box in enumerate(others):
            out.extend(_box_defects(box, width, height, f"others[{i}]"))
    sides = record.get("sides")
    if not isinstance(sides, dict) or set(sides) != set(SIDES):
        out.append("sides: needs exactly L and R")
    else:
        for name in SIDES:
            out.extend(_side_defects(sides[name], width, height, f"sides.{name}"))
    if "note" in record and not isinstance(record["note"], str):
        out.append("note: must be a string")
    extra = set(record) - {"frame", "labeller", "task", "patient", "others", "sides", "note"}
    if extra:
        out.append(f"record: unknown keys {sorted(extra)}")
    return out


def dumps(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(dumps(r) + "\n" for r in records), encoding="utf-8")


def consensus(first: dict, second: dict) -> dict:
    """Two labellers' records → agreed states, mean points; disagreements become `?`."""
    out: dict[str, Any] = {"patient": None, "sides": {}}
    if first["patient"] and second["patient"]:
        out["patient"] = list(np.mean([first["patient"], second["patient"]], axis=0))
    for side in SIDES:
        s1, s2 = first["sides"][side], second["sides"][side]
        hand = s1["hand"] if s1["hand"] == s2["hand"] else "?"
        joints = {}
        for part, names in (("joints", HAND_JOINTS), ("body", BODY_JOINTS)):
            for name in names:
                j1, j2 = s1[part].get(name, {}), s2[part].get(name, {})
                state = j1.get("state") if j1.get("state") == j2.get("state") else "?"
                entry = {"state": state}
                if state == "vis":
                    entry["xy"] = list(np.mean([j1["xy"], j2["xy"]], axis=0))
                joints[f"{part}.{name}"] = entry
        ext = [e for e in (s1.get("extent"), s2.get("extent")) if e]
        out["sides"][side] = {
            "hand": hand,
            "joints": joints,
            "extent": list(np.mean(ext, axis=0)) if ext else None,
        }
    return out
