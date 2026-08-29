"""Does the rigidity ELIGIBLE POPULATION saturate in the RANSAC instrument, or keep growing?

R2 gates on ``drift_p95 <= DRIFT_P95_GATE_PX`` and estimates with an independent
``RANSAC_THRESHOLD_PX``.  The decoupling is proven: sweeping the estimator while
the gate holds leaves most published drift trajectories nonmonotonic.  That
answers what the statistic does, not who gets one.  Raising the instrument from
4 px to 8 px moved eligibility from 286 assets to 298, so the instrument itself
recruits assets, and a headline denominator is only a corpus property if that
recruitment stops.

Every support criterion -- inlier count, grid coverage, valid fraction -- is
non-decreasing in the threshold, so an asset eligible at 8 px cannot be lost by
raising it, and the whole question lives in the assets that carry no verdict.
The sweep runs there and spot-checks the monotonicity it relies on.

Rerun:
    uv run --no-sync python scripts/probe_rigidity_saturation.py \
        --inventory inventory --corpus videos/3-cam --measurements measurements
"""

from __future__ import annotations

import argparse
import collections
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from pose_estimation import inventory, measure, sessions
from pose_estimation.measure import rigidity

THRESHOLDS = (4.0, 6.0, 8.0, 12.0, 20.0, 32.0, 48.0)
BASELINE_PX = 8.0
MONOTONICITY_SAMPLE = 20
NO_VERDICT = ("unmeasurable", "excluded_orientation", "error")


def _init(threshold: float) -> None:
    rigidity.RANSAC_THRESHOLD_PX = float(threshold)


def _one(argument: tuple[rigidity.Asset, str]) -> tuple[str, str]:
    asset, path = argument
    result = rigidity._worker((asset, path))
    return result.asset_id, result.rigidity_flag


def _sweep(
    arguments: list[tuple[rigidity.Asset, str]], threshold: float, workers: int
) -> dict[str, str]:
    if not arguments:
        return {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(threshold,)) as pool:
        return dict(pool.map(_one, arguments, chunksize=1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--measurements", required=True)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args(argv)

    if rigidity.RANSAC_THRESHOLD_PX != BASELINE_PX:
        raise RuntimeError("the committed instrument is no longer the baseline this probe anchors")
    if rigidity.RANSAC_THRESHOLD_PX == rigidity.DRIFT_P95_GATE_PX:
        raise RuntimeError("the estimator threshold and the accept gate are coupled")

    inventory.validate_generation(arguments.inventory)
    sidecar = measure.validate(arguments.measurements, inventory_dir=Path(arguments.inventory))
    published = {
        key[0]: row["rigidity_flag"] for key, row in measure.load_axis(sidecar, "rigidity").items()
    }
    baseline_counts = collections.Counter(published.values())
    eligible_at_baseline = baseline_counts["rigid"] + baseline_counts["camera_motion"]

    assets = sorted(rigidity.load_assets(arguments.inventory), key=lambda a: a.asset_id)
    by_id = {asset.asset_id: asset for asset in assets}
    unmeasurable = [by_id[key] for key, flag in sorted(published.items()) if flag == "unmeasurable"]
    eligible_ids = [
        key for key, flag in sorted(published.items()) if flag in {"rigid", "camera_motion"}
    ]

    def paths(subset: list[rigidity.Asset]) -> list[tuple[rigidity.Asset, str]]:
        return [
            (asset, str(sessions.resolve_source(arguments.corpus, asset.source_relative)))
            for asset in subset
        ]

    rows: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        flags = _sweep(paths(unmeasurable), threshold, arguments.workers)
        counts = collections.Counter(flags.values())
        recovered = sorted(key for key, flag in flags.items() if flag not in NO_VERDICT)
        rows.append(
            {
                "ransac_threshold_px": threshold,
                "recovered_from_unmeasurable": len(recovered),
                "eligible_population": eligible_at_baseline + len(recovered),
                "flags": dict(sorted(counts.items())),
            }
        )
        print(
            f"thr={threshold:5.1f} recovered={len(recovered):3} "
            f"eligible={eligible_at_baseline + len(recovered):4}"
        )

    # The sweep prices only the assets that carry no verdict, which is sound
    # while eligibility never falls as the threshold rises.  Spot-check that on
    # already-eligible assets at the loosest setting rather than assuming it.
    sample = [
        by_id[key] for key in eligible_ids[:: max(1, len(eligible_ids) // MONOTONICITY_SAMPLE)]
    ]
    sample = sample[:MONOTONICITY_SAMPLE]
    top_flags = _sweep(paths(sample), THRESHOLDS[-1], arguments.workers)
    lost = sorted(key for key, flag in top_flags.items() if flag in NO_VERDICT)

    baseline_row = next(row for row in rows if row["ransac_threshold_px"] == BASELINE_PX)
    if baseline_row["eligible_population"] != eligible_at_baseline:
        raise RuntimeError("the probe disagrees with the published baseline it anchors on")
    growth = [
        rows[index]["eligible_population"] - rows[index - 1]["eligible_population"]
        for index in range(1, len(rows))
    ]
    report = {
        "population": {
            "assets": len(assets),
            "published_flags": dict(sorted(baseline_counts.items())),
            "eligible_at_baseline": eligible_at_baseline,
            "swept": len(unmeasurable),
        },
        "baseline_px": BASELINE_PX,
        "accept_gate_px": rigidity.DRIFT_P95_GATE_PX,
        "thresholds": rows,
        "eligible_growth_per_step": growth,
        "saturated_above_baseline": all(
            value == 0
            for row, value in zip(rows[1:], growth, strict=True)
            if row["ransac_threshold_px"] > BASELINE_PX
        ),
        "monotonicity_spot_check": {
            "sampled_eligible_assets": len(sample),
            "threshold_px": THRESHOLDS[-1],
            "lost_eligibility": len(lost),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if lost:
        raise RuntimeError("eligibility fell as the threshold rose; the sweep population is wrong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
