"""Sweep the independent MAGSAC threshold against published rigidity drift."""

from __future__ import annotations

import itertools
import math
import statistics as st
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from pose_estimation import inventory, sessions
from pose_estimation.measure import rigidity

THRESHOLDS = (4.0, 6.0, 8.0, 12.0, 20.0, 32.0)
N_ASSETS = 20
CORPUS = Path("videos/3-cam")


def _init(threshold: float) -> None:
    rigidity.RANSAC_THRESHOLD_PX = float(threshold)


def _one(asset: rigidity.Asset) -> tuple[str, float, float, float]:
    path = sessions.resolve_source(CORPUS, asset.source_relative)
    out = rigidity.analyze_asset(asset, path)
    return out.asset_id, out.drift_median_px, out.drift_p95_px, out.inliers_median


def _is_nondecreasing(values: list[float]) -> bool:
    return all(left <= right for left, right in itertools.pairwise(values))


def main() -> None:
    if rigidity.RANSAC_THRESHOLD_PX == rigidity.DRIFT_P95_GATE_PX:
        raise RuntimeError("the estimator threshold and accept gate are coupled")
    inventory.validate_generation("inventory")
    assets = rigidity.load_assets("inventory")
    assets = assets[:: max(1, len(assets) // N_ASSETS)][:N_ASSETS]
    by_threshold: dict[float, dict[str, tuple[float, float, float]]] = {}
    print(
        f"n_assets={len(assets)} thresholds={THRESHOLDS} "
        f"accept_gate_px={rigidity.DRIFT_P95_GATE_PX}"
    )
    for threshold in THRESHOLDS:
        with ProcessPoolExecutor(max_workers=4, initializer=_init, initargs=(threshold,)) as pool:
            measured = {
                asset_id: (drift_median, drift_p95, inliers)
                for asset_id, drift_median, drift_p95, inliers in pool.map(_one, assets)
                if math.isfinite(drift_p95)
            }
        by_threshold[threshold] = measured
        medians = [st.median([values[index] for values in measured.values()]) for index in range(3)]
        print(
            f"thr={threshold:5.1f} n={len(measured):3} "
            f"drift_p95_median={medians[1]:7.3f} "
            f"drift_median_median={medians[0]:7.3f} inliers_median={medians[2]:7.1f}"
        )
    complete = set.intersection(*(set(results) for results in by_threshold.values()))
    trajectories = [
        [by_threshold[threshold][asset_id][1] for threshold in THRESHOLDS] for asset_id in complete
    ]
    nonmonotonic = sum(not _is_nondecreasing(values) for values in trajectories)
    print(
        f"complete_trajectories={len(trajectories)} "
        f"nonmonotonic_drift_p95={nonmonotonic} "
        f"monotonic_drift_p95={len(trajectories) - nonmonotonic}"
    )
    if not nonmonotonic:
        raise RuntimeError("sampled drift_p95 still tracks the estimator threshold monotonically")


if __name__ == "__main__":
    main()
