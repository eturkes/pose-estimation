"""Frozen corpus frame samples for labelling (contract M4.1 D04).

Selection hashes `(seed, key)` instead of drawing from an RNG stream, so one stratum's draw never
shifts another's and a re-run with the same inputs returns the same frames. Frame positions span
the whole decoded clip: the published task span equals the whole clip on most assets, and the
labeller's `task` field is what grades trimming.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

VIEWS = ("above", "left", "right")


@dataclass(frozen=True)
class Clip:
    event_id: str
    camera_name: str
    subject: int
    task: str
    side: str
    view: str
    n_frames: int


@dataclass(frozen=True)
class SampleFrame:
    event_id: str
    camera_name: str
    frame: int
    stratum: str  # "main" | "stress"
    subject: int
    view: str


def _rank(seed: int, *key: object) -> int:
    digest = hashlib.sha256(repr((seed, *key)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _draw(seed: int, tag: str, clip: Clip, k: int, used: set[tuple[str, str, int]]) -> int:
    """Frame `k` of `clip`'s draw: the hashed position, else the next unused one in hashed order.

    The first probe alone decides every draw that does not collide, so adding distinctness left
    every collision-free sample unchanged.
    """
    first = _rank(seed, tag, clip.event_id, clip.camera_name, k) % clip.n_frames
    if (clip.event_id, clip.camera_name, first) not in used:
        return first
    order = sorted(
        range(clip.n_frames),
        key=lambda f: _rank(seed, tag, clip.event_id, clip.camera_name, "order", f),
    )
    for frame in order:
        if (clip.event_id, clip.camera_name, frame) not in used:
            return frame
    raise ValueError("clip exhausted")  # unreachable: callers check capacity first


def _take(
    seed: int, tag: str, cell: Sequence[Clip], quota: int, used: set
) -> list[tuple[Clip, int]]:
    """`quota` distinct frames, round-robin over `cell` (already in hashed order)."""
    free = {
        c: c.n_frames - sum(1 for u in used if u[:2] == (c.event_id, c.camera_name)) for c in cell
    }
    if sum(free.values()) < quota:
        raise ValueError(f"{tag}: quota {quota} exceeds {sum(free.values())} distinct frames")
    out, k = [], 0
    while len(out) < quota:
        c = cell[k % len(cell)]
        if free[c] > 0:
            frame = _draw(seed, tag, c, k, used)
            used.add((c.event_id, c.camera_name, frame))
            free[c] -= 1
            out.append((c, frame))
        k += 1
    return out


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def corpus_clips(sessions: Path, inventory: Path, run: Path) -> list[Clip]:
    """Every placed camera artifact with a published diagnostics row, ordered like the review UI."""
    events = {r["event_id"]: r for r in _rows(sessions / "events.csv")}
    assets = {r["asset_id"]: r for r in _rows(inventory / "assets.csv")}
    clips = []
    for placement in _rows(sessions / "placements.csv"):
        event_id, camera = placement.get("event_id", ""), placement.get("camera_name", "")
        diag = run / event_id / f"{camera}_diag.csv"
        if not event_id or not camera or not diag.is_file():
            continue
        n_frames = int(_rows(diag)[0]["n_frames_decoded"])
        asset = assets[placement["asset_id"]]
        event = events[event_id]
        clips.append(
            Clip(
                event_id,
                camera,
                int(event["subject_ordinal"]),
                event["task"],
                event["side"],
                asset["view"],
                n_frames,
            )
        )
    clips.sort(key=lambda c: (c.task, c.side, c.event_id, c.camera_name))
    return clips


def family_labels(clips: Sequence[Clip]) -> dict[tuple[str, str], str]:
    """`#NNN task/side/view` per clip — the review UI's positional event ordinals
    (`review_ui.clips._number_families`), which key the M2.10 watch-set records."""
    numbers: dict[str, int] = {}
    labels = {}
    for c in sorted(clips, key=lambda c: (c.task, c.side, c.event_id, c.camera_name)):
        numbers.setdefault(c.event_id, len(numbers) + 1)
        labels[(c.event_id, c.camera_name)] = (
            f"#{numbers[c.event_id]:03d} {c.task}/{c.side}/{c.view}"
        )
    return labels


def corpus_sample(
    clips: Iterable[Clip],
    seed: int,
    watch: set[tuple[str, str]],
    subjects: int = 8,
    per_view: int = 3,
    stress: int = 30,
) -> list[SampleFrame]:
    """`subjects` x `VIEWS` x `per_view` frames outside the watch set + `stress` watch-set frames."""
    pool = [c for c in clips if c.n_frames > 0]
    main = [c for c in pool if (c.event_id, c.camera_name) not in watch]
    by_cell: dict[tuple[int, str], list[Clip]] = {}
    for c in main:
        by_cell.setdefault((c.subject, c.view), []).append(c)
    eligible = sorted(
        {c.subject for c in main if all((c.subject, v) in by_cell for v in VIEWS)},
        key=lambda s: _rank(seed, "subject", s),
    )
    if len(eligible) < subjects:
        raise ValueError(f"{len(eligible)} subjects carry every view outside the watch set")
    out: list[SampleFrame] = []
    used: set[tuple[str, str, int]] = set()
    for subject in eligible[:subjects]:
        for view in VIEWS:
            cell = sorted(
                by_cell[(subject, view)],
                key=lambda c: _rank(seed, "clip", c.event_id, c.camera_name),
            )
            for c, frame in _take(seed, "frame", cell, per_view, used):
                out.append(SampleFrame(c.event_id, c.camera_name, frame, "main", subject, view))
    watched = sorted(
        (c for c in pool if (c.event_id, c.camera_name) in watch),
        key=lambda c: _rank(seed, "stress", c.event_id, c.camera_name),
    )
    if stress and not watched:
        raise ValueError("no watch-set clip for the stress stratum")
    if stress:
        for c, frame in _take(seed, "stress-frame", watched, stress, used):
            out.append(SampleFrame(c.event_id, c.camera_name, frame, "stress", c.subject, c.view))
    return out


def read_frames(path: Path, indices: Iterable[int]):
    """Decode `path` in order and yield `(index, BGR frame)` for each requested index.

    Sequential decode reproduces the pipeline's frame numbering exactly (seeking does not on
    every codec here), through the same auto-orientation `open_capture` requests.
    """
    from pose_estimation.video_io import open_capture

    wanted = sorted(set(indices))
    cap = open_capture(str(path))
    if cap is None:
        raise OSError(f"cannot open {path.name}")
    try:
        index = 0
        for target in wanted:
            while index < target:
                if not cap.grab():
                    raise EOFError(f"{path.name}: ends before frame {target}")
                index += 1
            ok, frame = cap.read()
            if not ok:
                raise EOFError(f"{path.name}: ends at frame {target}")
            index += 1
            yield target, frame
    finally:
        cap.release()
