"""Hand ground truth from manipulation / in-the-wild sets → one `GtHandFrame` shape (contract M4.2 D02).

Both sets annotate hands only, so a whole-body arm needs a person box: the screening asks the
shipped detector for one (`person_box`), and a hand no detector box covers is counted, never
dropped silently. Joint order in both sets = COCO-WholeBody's hand order (wrist, then four joints
per finger from thumb to little finger, base to tip).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GtHandFrame:
    image: Path
    side: str  # "L" | "R", anatomical
    kpts: np.ndarray  # (21, 2)
    valid: np.ndarray  # (21,) bool — annotated (and, for HInt, inside the image)
    occluded: np.ndarray  # (21,) bool — annotated as occluded (HInt); all False where unknown
    box: tuple[float, float, float, float]  # hand box x0, y0, x1, y1

    @property
    def diagonal(self) -> float:
        return float(np.hypot(self.box[2] - self.box[0], self.box[3] - self.box[1]))


def _box_of(points: np.ndarray) -> tuple[float, float, float, float]:
    lo, hi = points.min(axis=0), points.max(axis=0)
    return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])


#: dex-ycb-toolkit `dex_ycb.py` `_MANO_JOINTS`, verbatim order.
DEXYCB_JOINTS = (
    "wrist",
    "thumb_mcp", "thumb_pip", "thumb_dip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "little_mcp", "little_pip", "little_dip", "little_tip",
)  # fmt: skip
#: COCO-WholeBody hand order (mmpose `coco_wholebody_hand`): wrist, then four joints per finger,
#: thumb → little, base → tip; named in the toolkit's vocabulary.
COCO_WB_HAND = (
    "wrist",
    *(f"{finger}_{joint}" for finger in ("thumb", "index", "middle", "ring", "little")
      for joint in ("mcp", "pip", "dip", "tip")),
)  # fmt: skip
#: COCO-WholeBody hand slot → DexYCB joint index, joined by name.
DEXYCB_TO_COCO_WB = tuple(DEXYCB_JOINTS.index(name) for name in COCO_WB_HAND)


def dexycb(root: Path, every: int = 12) -> Iterator[GtHandFrame]:
    """Every `every`-th labelled colour frame of each sequence + camera under `root` (one subject).

    `labels_*.npz` `joint_2d` = (1, 21, 2), all -1 where the hand has no annotation.
    """
    import yaml

    for meta_path in sorted(root.glob("*/meta.yml")):
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        side = {"left": "L", "right": "R"}[meta["mano_sides"][0]]
        for camera in sorted(p for p in meta_path.parent.iterdir() if p.is_dir()):
            for label in sorted(camera.glob("labels_*.npz"))[::every]:
                joints = np.load(label)["joint_2d"][0].astype(float)[list(DEXYCB_TO_COCO_WB)]
                valid = (joints >= 0).all(axis=1)
                if valid.sum() < 21:
                    continue
                image = camera / label.name.replace("labels_", "color_").replace(".npz", ".jpg")
                yield GtHandFrame(image, side, joints, valid, np.zeros(21, bool), _box_of(joints))


def hint(root: Path) -> Iterator[GtHandFrame]:
    """HInt records: one JSON list per image, side from the `_l` / `_r` file-name suffix."""
    for path in sorted(root.glob("*.json")):
        side = {"l": "L", "r": "R"}[path.stem.rsplit("_", 1)[1]]
        for record in json.loads(path.read_text(encoding="utf-8")):
            kpts = np.asarray(record["keypoints"], dtype=float)
            exist = np.asarray(record["existence"], dtype=float) > 0
            occluded = np.asarray(record["occlusion"], dtype=float) > 0
            x0, y0, x1, y1 = record["bbox"][0]
            yield GtHandFrame(
                path.with_suffix(".jpg"), side, kpts, exist, occluded, (x0, y0, x1, y1)
            )


def person_box(boxes: np.ndarray, hand: GtHandFrame) -> np.ndarray | None:
    """The detector box holding most of the hand box (>= half its area); largest on ties."""
    if len(boxes) == 0:
        return None
    hx0, hy0, hx1, hy1 = hand.box
    area = max((hx1 - hx0) * (hy1 - hy0), 1e-9)
    ix = np.clip(np.minimum(boxes[:, 2], hx1) - np.maximum(boxes[:, 0], hx0), 0, None)
    iy = np.clip(np.minimum(boxes[:, 3], hy1) - np.maximum(boxes[:, 1], hy0), 0, None)
    cover = ix * iy / area
    if cover.max() < 0.5:
        return None
    size = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best = np.lexsort((size, cover))[-1]
    return boxes[best]
