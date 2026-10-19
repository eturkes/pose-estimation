"""Blinded pairwise judgment sheets + tally (contract M4.1 D03).

Each pair of candidate points for one joint is shown twice with the two marker colours swapped.
A sheet carries the crop, two markers and nothing else: its file name is an opaque token, and the
token → arm mapping lives only in the sealed key the judge never receives.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .render import crop_sheet

#: Marker colours (BGR); both read clearly on skin, cloth and table tones.
COLOURS = {"yellow": (0, 255, 255), "blue": (255, 96, 0)}
RESPONSES = ("yellow", "blue", "both-ok", "both-wrong", "?")
#: Error-difference bins (fraction of the normaliser) for judge accuracy.
BINS = (0.0, 0.05, 0.1, 0.2, float("inf"))


@dataclass(frozen=True)
class Pair:
    pair_id: str
    image: str  # resolved by the caller's loader; never written to a sheet
    joint: str
    a: tuple[float, float]
    b: tuple[float, float]
    arm_a: str
    arm_b: str
    side: str  # the joint's anatomical side, "L" | "R"
    crop: float | None = None  # crop edge, frame px; None = 4 x the points' separation (>= 64)


def _token(seed: int, *key: object) -> str:
    return hashlib.sha256(repr((seed, *key)).encode()).hexdigest()[:16]


def make_sheets(
    pairs: Sequence[Pair],
    load_image: Callable[[str], np.ndarray],
    sheet_dir: Path,
    key_path: Path,
    seed: int,
) -> list[dict[str, str]]:
    """Write two swapped sheets per pair + the sealed key; return the judge's job list."""
    sheet_dir.mkdir(parents=True, exist_ok=True)
    jobs, key = [], []
    for pair in pairs:
        image = load_image(pair.image)
        first_yellow_is_a = int(_token(seed, "colour", pair.pair_id), 16) & 1 == 1
        for swap in (0, 1):
            yellow_is_a = first_yellow_is_a != bool(swap)
            token = _token(seed, "sheet", pair.pair_id, swap)
            centre = ((pair.a[0] + pair.b[0]) / 2, (pair.a[1] + pair.b[1]) / 2)
            sep = float(np.hypot(pair.a[0] - pair.b[0], pair.a[1] - pair.b[1]))
            crop = pair.crop if pair.crop is not None else max(64.0, 4 * sep)
            canvas, sm = crop_sheet(image, centre, max(crop, 3 * sep))
            for point, colour in (
                (pair.a, "yellow" if yellow_is_a else "blue"),
                (pair.b, "blue" if yellow_is_a else "yellow"),
            ):
                x, y = (round(v) for v in sm.to_sheet(*point))
                cv2.circle(canvas, (x, y), 7, COLOURS[colour], 2)
                cv2.circle(canvas, (x, y), 1, COLOURS[colour], -1)
            cv2.imwrite(str(sheet_dir / f"{token}.png"), canvas)
            jobs.append({"sheet": f"{token}.png", "joint": pair.joint})
            key.append(
                {
                    "token": token,
                    "pair_id": pair.pair_id,
                    "swap": swap,
                    "yellow": pair.arm_a if yellow_is_a else pair.arm_b,
                    "blue": pair.arm_b if yellow_is_a else pair.arm_a,
                }
            )
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(
        "".join(json.dumps(k, sort_keys=True) + "\n" for k in key), encoding="utf-8"
    )
    order = sorted(range(len(jobs)), key=lambda i: _token(seed, "order", jobs[i]["sheet"]))
    return [jobs[i] for i in order]


def decode(key: Iterable[dict], answers: dict[str, str]) -> list[dict[str, str]]:
    """One row per answered sheet: `pair_id`, `swap`, `outcome` = an arm name or a non-decisive class."""
    rows = []
    for k in key:
        response = answers.get(f"{k['token']}.png")
        if response is None:
            continue
        if response not in RESPONSES:
            raise ValueError(f"unknown response {response!r}")
        outcome = k[response] if response in COLOURS else response
        rows.append({"pair_id": k["pair_id"], "swap": str(k["swap"]), "outcome": outcome})
    return rows


def bin_of(margin: float) -> int:
    for i in range(len(BINS) - 1):
        if BINS[i] <= margin < BINS[i + 1]:
            return i
    raise ValueError(f"margin {margin} outside the bins")
