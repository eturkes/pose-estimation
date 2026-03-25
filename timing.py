"""Lightweight per-frame timing instrumentation."""

import time


class FrameTimer:
    """Accumulates wall-clock duration for named pipeline stages.

    Usage::

        timer = FrameTimer()
        timer.start("pose_detection")
        ...
        timer.stop("pose_detection")
        print(timer.to_dict())   # {"pose_detection": 0.0042, ...}
    """

    def __init__(self):
        self._starts = {}
        self._totals = {}

    def start(self, name):
        """Begin timing *name*.  Nesting the same name is not supported."""
        self._starts[name] = time.perf_counter()

    def stop(self, name):
        """Stop timing *name* and accumulate elapsed seconds."""
        elapsed = time.perf_counter() - self._starts.pop(name)
        self._totals[name] = self._totals.get(name, 0.0) + elapsed

    def to_dict(self):
        """Return ``{stage: total_seconds}`` rounded to 6 decimal places."""
        return {k: round(v, 6) for k, v in self._totals.items()}

    def reset(self):
        """Clear all accumulated timings for a new frame."""
        self._starts.clear()
        self._totals.clear()
