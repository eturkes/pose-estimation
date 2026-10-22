"""Production-pipeline replay over a frame window (contract M4.1 D05(e)).

`run.main` runs unchanged; only its capture is narrowed: `WindowCapture` decodes and discards the
frames before `start` and reports end of stream after `stop`, while every `get` (presentation
timestamps included) answers from the real capture. The replayed rows therefore pass through the
exact tracker, hygiene, smoother, bone-length, hand-gate and export code a corpus run uses.
"""

from __future__ import annotations

import contextlib
import csv
from pathlib import Path

#: The corpus driver's defaults (`scripts/corpus_run_2d.py`), camera survey included: it trims
#: setup frames outside the task span, so a replay without it poses frames production never
#: exports. The pose device follows the model's registry entry, so a `--model` override needs no
#: device.
CORPUS_ARGS = (
    "--headless",
    "--model=rtmw-l-384",
    "--tracking=body",
    "--tracker=subject",
    "--det-device=GPU",
    "--det-frequency=1",
    "--single-subject",
    "--drop-lower-body",
    "--drop-hips-camera=above",
    "--hand-gate",
    "--camera-survey",
)


class WindowCapture:
    """A `cv2.VideoCapture` view over decoded frames `[start, stop]`."""

    def __init__(self, capture, start: int, stop: int):
        self._cap = capture
        for _ in range(start):
            if not capture.grab():
                break
        self._left = stop - start + 1

    def read(self):
        if self._left <= 0:
            return False, None
        self._left -= 1
        return self._cap.read()

    def isOpened(self):  # noqa: N802 — cv2 interface
        return self._cap.isOpened()

    def get(self, prop):
        return self._cap.get(prop)

    def set(self, prop, value):
        return self._cap.set(prop, value)

    def release(self):
        self._cap.release()


@contextlib.contextmanager
def _windowed(start: int, stop: int):
    from pose_estimation import run

    original = run.open_capture

    def opener(source, display=None):
        cap = original(source, display=display)
        return None if cap is None else WindowCapture(cap, start, stop)

    # setattr: ty refuses rebinding a module-level function by assignment.
    setattr(run, "open_capture", opener)  # noqa: B010
    try:
        yield
    finally:
        setattr(run, "open_capture", original)  # noqa: B010


def replay_window(media: Path, start: int, stop: int, out_dir: Path, extra=()) -> Path:
    """Run the corpus configuration over frames `[start, stop]`; return the landmark CSV.

    The CSV's `frame_idx` counts from `start` (= 0); add `start` for the source index.
    """
    from pose_estimation import run

    out_dir.mkdir(parents=True, exist_ok=True)
    # The runner prints the source path and every written file; those lines name a recording,
    # so they go to the window's own log, never to the command's stdout (data boundary).
    with (
        (out_dir / "replay.log").open("w", encoding="utf-8") as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
        _windowed(start, stop),
    ):
        run.main(["--source", str(media), f"--output-dir={out_dir}", *CORPUS_ARGS, *extra])
    return out_dir / f"{media.stem}.csv"


def row_at(csv_path: Path, frame: int) -> dict[str, str] | None:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row["frame_idx"]) == frame:
                return row
    return None


_META = {"video", "frame_idx", "timestamp_sec", "person_idx"}


def landmark_cells(row: dict[str, str] | None) -> dict[str, str]:
    if row is None:
        return {}
    return {k: v for k, v in row.items() if k not in _META and not k.startswith("cam_")}


#: A replayed point "equals" the whole-clip run within this (frame-normalised, ~2 px at 1920):
#: the smoother's state decays geometrically, so byte equality never arrives.
PREROLL_TOLERANCE = 0.001


def read_point_delta(want: dict[str, str], got: dict[str, str]) -> tuple[float, int]:
    """Max |dx|, |dy| over points both rows expose to the R stage (vis / conf > 0), and the
    count of points one row exposes and the other does not."""
    worst, flips = 0.0, 0
    for k in want.keys() | got.keys():
        if not k.endswith("_x"):
            continue
        base = k[:-2]
        conf = f"{base}_vis" if base.startswith(("body_", "arm_")) else f"{base}_conf"

        def seen(row: dict[str, str], k=k, conf=conf) -> bool:
            v = row.get(conf) or ""
            return v not in ("", "NA") and float(v) > 0 and row.get(k, "") != ""

        a_seen, b_seen = seen(want), seen(got)
        if a_seen != b_seen:
            flips += 1
        elif a_seen:
            y = f"{base}_y"
            delta = max(abs(float(want[k]) - float(got[k])), abs(float(want[y]) - float(got[y])))
            worst = max(worst, delta)
    return worst, flips
