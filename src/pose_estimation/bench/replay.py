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

#: The corpus driver's defaults (`scripts/corpus_run_2d.py`) minus the camera survey (`run.py`
#: default off): on an untrimmed clip the survey adds camera columns and moves nothing else. The
#: pose device follows the model's registry entry, so a `--model` override needs no device.
CORPUS_ARGS = (
    "--headless",
    "--model=rtmw-l",
    "--tracking=body",
    "--tracker=subject",
    "--det-device=GPU",
    "--det-frequency=1",
    "--single-subject",
    "--drop-lower-body",
    "--drop-hips-camera=above",
    "--hand-gate",
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
    with _windowed(start, stop):
        run.main(["--source", str(media), f"--output-dir={out_dir}", *CORPUS_ARGS, *extra])
    return out_dir / f"{media.stem}.csv"


def row_at(csv_path: Path, frame: int) -> dict[str, str] | None:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row["frame_idx"]) == frame:
                return row
    return None
