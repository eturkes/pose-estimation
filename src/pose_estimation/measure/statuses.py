"""The published status alphabets.

A leaf module with no heavy imports: ingestion validates against these, and a
consumer that only reads a table must not pull PyAV and NumPy in behind them.

The split that matters is published against unpublished.  ``estimate_drift``
has its own status vocabulary, and none of it reaches a column — the sync
schema carries ``drift_ppm``/``drift_se`` and no drift status — so a drift
token appearing in ``status_audio`` is a producer that crossed two alphabets.
"""

from __future__ import annotations

AUDIO_STATUSES = frozenset(
    {
        "ok",
        "short_audio",
        "silent",
        "no_feasible_lag",
        "no_background",
        "boundary_peak",
        "low_confidence",
    }
)

DRIFT_STATUSES = frozenset(
    {"short_overlap", "insufficient_windows", "degenerate_regression", "global_abstention"}
)

VISUAL_STATUSES = frozenset(
    {
        "ok",
        "insufficient_overlap",
        "edge_peak",
        "undefined_confidence",
        "low_peak_correlation",
        "low_prominence",
        "ambiguous_peak",
        # Raised by the axis driver, not the estimator: a trace that never
        # loaded is still the visual instrument declining to speak.
        "signal_absent",
    }
)

# An accepted row publishes its statistics.  A gate rejects an estimate, it
# never erases one, so an empty cell means no peak was computed rather than a
# peak that failed — which is P39's instrument/gate separation seen from the
# data side.  Only these columns are required, and only when the status is ok.
REQUIRED_WHEN_OK: dict[str, tuple[str, ...]] = {
    "status_audio": ("offset_audio_s", "conf_audio", "peak_ratio_audio"),
    "status_visual": ("offset_visual_s", "conf_visual", "peak_corr_visual"),
}
