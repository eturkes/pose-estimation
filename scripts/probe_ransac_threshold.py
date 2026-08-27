"""Is residual_p95 an instrument-noise estimate, or is it the RANSAC threshold showing through?

`_match_sample` sets the MAGSAC inlier threshold from DRIFT_P95_GATE_PX, the same constant P21
accepts against, so no residual above the gate can be reported and the gate is judged against a
quantity it pins. Sweeping the threshold alone separates the two readings: a truncated statistic
tracks the threshold, a real matching-precision floor plateaus.
"""

import importlib.util
import math
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

THRESHOLDS = (4.0, 6.0, 8.0, 12.0, 20.0, 32.0)
N_ASSETS = 20
CORPUS = Path("videos/3-cam")


def _load():
    spec = importlib.util.spec_from_file_location("geo", "scripts/geometry_qualification.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["geo"] = module
    spec.loader.exec_module(module)
    return module


geo = _load()


def _init(threshold: float) -> None:
    global geo
    geo = _load()
    geo.DRIFT_P95_GATE_PX = threshold


def _one(asset):
    out = geo.analyze_asset(asset, CORPUS)
    return (
        out.asset_id,
        out.drift_median_px,
        out.drift_p95_px,
        out.residual_p95_px,
        out.inliers_median,
    )


def main() -> None:
    # Selected straight from the registry rather than from a prior run's verdicts: the probe has to
    # rerun from committed state, and a non-finite result is simply dropped below.
    assets = geo.load_assets(Path("inventory/assets.csv"))
    assets = assets[:: max(1, len(assets) // N_ASSETS)][:N_ASSETS]
    print(f"n_assets={len(assets)} thresholds={THRESHOLDS}")

    for threshold in THRESHOLDS:
        with ProcessPoolExecutor(max_workers=4, initializer=_init, initargs=(threshold,)) as pool:
            results = [r for r in pool.map(_one, assets) if math.isfinite(r[3])]
        if not results:
            print(f"thr={threshold:5.1f} no measurable assets")
            continue
        med = [st.median([r[i] for r in results]) for i in (1, 2, 3, 4)]
        ratio = st.median([r[2] / r[3] for r in results if r[3] > 0])
        res_max = max(r[3] for r in results)
        print(
            f"thr={threshold:5.1f} n={len(results):3} residual_p95 med={med[2]:7.3f} max={res_max:7.3f} "
            f"| drift_p95 med={med[1]:7.3f} | drift_median med={med[0]:7.3f} "
            f"| ratio med={ratio:6.3f} | inliers med={med[3]:7.1f}"
        )


if __name__ == "__main__":
    main()
