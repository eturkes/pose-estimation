"""Labelling sheet sets + answer ingest (contract M4.1 D01-D02).

A sheet set = PNG sheets named by opaque tokens + `manifest.json` (token → kind, frame key, side,
sheet map) + `jobs.json` (what the labeller sees: token, kind, side, joints). Labellers answer in
grid units; `ingest_*` converts answers to codebook-v2 records in source-frame pixels.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .labels import BODY_JOINTS, HAND_JOINTS, SIDES
from .render import SheetMap, crop_sheet

SIDE_WORDS = {"L": "LEFT", "R": "RIGHT"}


@dataclass(frozen=True)
class SheetEntry:
    token: str
    kind: str  # "frame" | "body" | "hand"
    key: str  # frame key: "<event>/<camera>#<index>" or "coco/<image_id>/<ann_id>"
    index: int
    width: int
    height: int
    side: str  # "" for frame + body sheets
    map: dict[str, float]


def token(seed: int, *key: object) -> str:
    return hashlib.sha256(repr((seed, *key)).encode()).hexdigest()[:12]


def _caption(canvas: np.ndarray, text: str) -> np.ndarray:
    strip = np.full((28, canvas.shape[1], 3), 255, np.uint8)
    cv2.putText(strip, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return np.vstack([strip, canvas])


def write_sheet(
    out: Path,
    entries: list[SheetEntry],
    canvas: np.ndarray,
    sm: SheetMap,
    *,
    seed: int,
    kind: str,
    key: str,
    index: int,
    shape: tuple[int, int],
    side: str = "",
    caption: str = "",
) -> SheetEntry:
    """Write one sheet (+ optional caption strip above it) and record its manifest entry.

    A caption shifts the sheet down by 28 px; the map stays in the uncaptioned sheet's pixels, so
    readers use grid units, never raw sheet pixels.
    """
    t = token(seed, kind, key, side)
    image = _caption(canvas, caption) if caption else canvas
    cv2.imwrite(str(out / f"{t}.png"), image)
    entry = SheetEntry(t, kind, key, index, shape[1], shape[0], side, asdict(sm))
    entries.append(entry)
    return entry


def hand_crop(image: np.ndarray, extent: Sequence[float]) -> tuple[np.ndarray, SheetMap]:
    """Square crop at 2 x the extent's longer edge (>= 96 px) around its centre."""
    x0, y0, x1, y1 = extent
    side = max(96.0, 2.0 * max(x1 - x0, y1 - y0))
    return crop_sheet(image, ((x0 + x1) / 2, (y0 + y1) / 2), side)


def body_crop(image: np.ndarray, box: Sequence[float]) -> tuple[np.ndarray, SheetMap]:
    """Square crop at 1.2 x the person box's longer edge around its centre."""
    x0, y0, x1, y1 = box
    side = 1.2 * max(x1 - x0, y1 - y0)
    return crop_sheet(image, ((x0 + x1) / 2, (y0 + y1) / 2), side)


def save(out: Path, entries: Sequence[SheetEntry], seed: int) -> None:
    manifest = {e.token: asdict(e) for e in entries}
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8"
    )
    order = sorted(entries, key=lambda e: token(seed, "order", e.token))
    jobs = [
        {
            "sheet": f"{e.token}.png",
            "kind": e.kind,
            "side": e.side,
            "joints": list(HAND_JOINTS)
            if e.kind == "hand"
            else list(BODY_JOINTS)
            if e.kind == "body"
            else [],
        }
        for e in order
    ]
    (out / "jobs.json").write_text(json.dumps(jobs, indent=1), encoding="utf-8")


def load_manifest(out: Path) -> dict[str, SheetEntry]:
    raw = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    return {t: SheetEntry(**e) for t, e in raw.items()}


def _map(entry: SheetEntry) -> SheetMap:
    return SheetMap(**entry.map)


def frame_box(sm: SheetMap, box: Any) -> list[float] | None:
    if box is None:
        return None
    x0, y0 = sm.to_frame(box[0], box[1])
    x1, y1 = sm.to_frame(box[2], box[3])
    return [
        round(min(x0, x1), 1),
        round(min(y0, y1), 1),
        round(max(x0, x1), 1),
        round(max(y0, y1), 1),
    ]


def _frame_joint(sm: SheetMap, answer: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    joint: dict[str, Any] = {"state": answer.get("state", "?")}
    if joint["state"] == "vis" and answer.get("uv") is not None:
        x, y = sm.to_frame(*answer["uv"])
        joint["xy"] = [
            round(min(max(x, 0.0), width - 0.5), 1),
            round(min(max(y, 0.0), height - 0.5), 1),
        ]
    if answer.get("own"):
        joint["own"] = answer["own"]
    return joint


def ingest(
    sheet_dir: Path, answers: Sequence[dict[str, Any]], labeller: str
) -> list[dict[str, Any]]:
    """Merge frame / body / hand sheet answers into one codebook-v2 record per frame key.

    A frame needs its frame-sheet answer; body and hand answers fill in its joints. Joints a
    labeller never answered stay absent, so `labels.validate` names them.
    """
    manifest = load_manifest(sheet_dir)
    records: dict[str, dict[str, Any]] = {}
    later = []
    for answer in answers:
        entry = manifest[Path(answer["sheet"]).stem]
        if entry.kind != "frame":
            later.append((entry, answer))
            continue
        sm = _map(entry)
        records[entry.key] = {
            "frame": {
                "key": entry.key,
                "index": entry.index,
                "width": entry.width,
                "height": entry.height,
            },
            "labeller": labeller,
            "task": answer.get("task", "?"),
            "patient": frame_box(sm, answer.get("patient")),
            "others": [frame_box(sm, b) for b in answer.get("others", [])],
            "sides": {
                s: {
                    "hand": answer.get(s, {}).get("hand", "?"),
                    "extent": frame_box(sm, answer.get(s, {}).get("extent")),
                    "joints": {},
                    "body": {},
                }
                for s in SIDES
            },
            "note": answer.get("note", ""),
        }
    for entry, answer in later:
        record = records.get(entry.key)
        if record is None:
            raise ValueError(f"{entry.token}: no frame-sheet answer for its frame")
        sm = _map(entry)
        if entry.kind == "body":
            for s in SIDES:
                for name, joint in answer.get(s, {}).items():
                    record["sides"][s]["body"][name] = _frame_joint(
                        sm, joint, entry.width, entry.height
                    )
        else:
            for name, joint in answer.get("joints", {}).items():
                record["sides"][entry.side]["joints"][name] = _frame_joint(
                    sm, joint, entry.width, entry.height
                )
    # A frame with no patient gets no body sheet: its arm joints are out of frame by the frame
    # answer itself. A body joint left unanswered on a frame WITH a patient stays absent, so
    # `labels.validate` names it.
    for record in records.values():
        if record["patient"] is None:
            for side in record["sides"].values():
                for name in BODY_JOINTS:
                    side["body"].setdefault(name, {"state": "oof"})
    return list(records.values())
