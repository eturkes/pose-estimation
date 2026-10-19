"""Public ground truth loaders → one record shape the qualification and screening code reads.

COCO-WholeBody v1.0: `keypoints` = COCO body-17 `(x, y, v)`, v 0 unlabelled / 1 labelled occluded
/ 2 visible; `lefthand_kpts` / `righthand_kpts` = 21 `(x, y, c)` per anatomical side, valid only
where `*_valid`; c = 2 on visible annotated joints (other values occur on semi-automatic labels).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BODY17 = (
    "nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow",
    "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
)  # fmt: skip


@dataclass(frozen=True)
class GtHand:
    image_id: int
    ann_id: int
    side: str  # "L" | "R", anatomical
    box: tuple[float, float, float, float]  # x, y, w, h
    kpts: np.ndarray  # (21, 3) x, y, c

    @property
    def diagonal(self) -> float:
        return float(np.hypot(self.box[2], self.box[3]))


@dataclass(frozen=True)
class GtPerson:
    image_id: int
    ann_id: int
    box: tuple[float, float, float, float]
    body: np.ndarray  # (17, 3) x, y, v
    area: float
    hands: tuple[GtHand, ...]
    #: `foot` (6, 3) + `face` (68, 3) rows; a part the annotation marks invalid reads all zeros.
    extra: dict[str, np.ndarray] = field(default_factory=dict)


def load_coco_wholebody(path: Path) -> tuple[dict[int, str], list[GtPerson]]:
    """`(image_id → file name, persons)`; crowd annotations dropped."""
    data = json.loads(path.read_text(encoding="utf-8"))
    files = {img["id"]: img["file_name"] for img in data["images"]}
    persons = []
    for ann in data["annotations"]:
        if ann.get("iscrowd"):
            continue
        hands = []
        for side, key in (("L", "lefthand"), ("R", "righthand")):
            if ann.get(f"{key}_valid"):
                hands.append(
                    GtHand(
                        ann["image_id"],
                        ann["id"],
                        side,
                        tuple(ann[f"{key}_box"]),
                        np.asarray(ann[f"{key}_kpts"], dtype=float).reshape(21, 3),
                    )
                )
        extra = {
            part: (
                np.asarray(ann[f"{part}_kpts"], dtype=float).reshape(n, 3)
                if ann.get(f"{part}_valid")
                else np.zeros((n, 3))
            )
            for part, n in (("foot", 6), ("face", 68))
        }
        persons.append(
            GtPerson(
                ann["image_id"],
                ann["id"],
                tuple(ann["bbox"]),
                np.asarray(ann["keypoints"], dtype=float).reshape(17, 3),
                float(ann["area"]),
                tuple(hands),
                extra,
            )
        )
    return files, persons
