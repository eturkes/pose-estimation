"""M4.5a bout qualification driver (contract `.agent/archive/contract-m4u5a.md` D04 + D06).

  measure-noise  D06: residual proxy (`bout_qualification.noise_proxy`) per asset-side over the
                 run's landmark CSVs — confidence-masked + camera-stabilized wrist, as the R
                 stage reads it — then divided by the proxy's measured response on known noise.
                 Prints corpus aggregates only, no asset identity.
  qualify        D04: development bank → every grid cell through the shipped R detector
                 (`analysis/bout_qualification.R`); noise-free bank → admissibility; ranking;
                 the chosen cell re-scored on the untouched qualification bank.

Usage (repo root):
  scripts/gate.sh python scripts/qualify_bouts.py measure-noise --run output/corpus-2d --out N.json
  scripts/gate.sh python scripts/qualify_bouts.py qualify --noise N.json --out DIR
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from pose_estimation import bout_qualification as bq

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "analysis" / "bout_qualification.R"
DEV_SEED, QUAL_SEED, FREE_SEED = 20261101, 20261102, 20261103
_SKIP = re.compile(
    r"_(clinical[_a-z0-9]*|movement_phases[_a-z0-9]*|diag|metrics|kp_detail|summary|smooth|"
    r"feature_rank)\.csv$"
)


def _wrist(rows: list[dict[str, str]], side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Timestamps + stabilized wrist, NaN where the R stage masks (vis <= 0 or blank)."""
    t, x, y = [], [], []
    for r in rows:
        vis = r.get(f"body_{side}_wrist_vis") or ""
        ok = vis not in ("", "NA") and float(vis) > 0 and r[f"body_{side}_wrist_x"] != ""
        px = float(r[f"body_{side}_wrist_x"]) if ok else math.nan
        py = float(r[f"body_{side}_wrist_y"]) if ok else math.nan
        cam = [r.get(c) or "" for c in ("cam_a", "cam_b", "cam_tx", "cam_ty")]
        if all(v not in ("", "NA") for v in cam):
            a, b, tx, ty = map(float, cam)
            px, py = a * px - b * py + tx, b * px + a * py + ty
        t.append(float(r["timestamp_sec"]))
        x.append(px)
        y.append(py)
    return np.asarray(t), np.asarray(x), np.asarray(y)


def cmd_measure_noise(a: argparse.Namespace) -> int:
    paths = sorted(
        f
        for e in a.run.iterdir()
        if e.is_dir()
        for f in e.glob("*.csv")
        if not _SKIP.search(f.name)
    )
    values: dict[str, list[float]] = {"left": [], "right": []}
    excluded = 0
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = [r for r in csv.DictReader(stream) if r.get("person_idx", "0") in ("0", "")]
        for side, found in values.items():
            t, x, y = _wrist(rows, side)
            rate = 1.0 / float(np.median(np.diff(t))) if len(t) > 1 else math.nan
            proxy = bq.noise_proxy(t, x, y, rate) if len(t) >= 7 and rate > 0 else None
            if proxy is None or not proxy > 0:
                excluded += 1
                continue
            found.append(proxy)
    response = bq.proxy_response(30.0, DEV_SEED)
    pooled = values["left"] + values["right"]
    report = {
        "assets": len(paths),
        "asset_sides_eligible": len(pooled),
        "asset_sides_excluded": excluded,
        "proxy": {side: bq.summarise(v) for side, v in values.items()}
        | {"all": bq.summarise(pooled)},
        "proxy_response_30hz": response,
        "noise_sd": float(np.median(pooled)) / response,
        "noise_sd_p90": float(np.quantile(pooled, 0.9)) / response,
    }
    text = json.dumps(report, indent=1)
    print(text)
    if a.out:
        a.out.write_text(text + "\n", encoding="utf-8")
    return 0


def _request(bank: list[bq.Series], grid: list[dict[str, float | str]]) -> dict:
    def arr(v: np.ndarray) -> list[float | None]:
        return [None if not math.isfinite(e) else float(e) for e in v]

    return {
        "cells": grid,
        "prominences": list(bq.PROMINENCES),
        "peak_spacing_sec": bq.PEAK_SPACING_SEC,
        "series": [
            {
                "sid": s.sid,
                "rate": s.rate,
                "t": arr(s.t),
                "x": arr(s.x),
                "y": arr(s.y),
                "frame": [int(f) for f in s.frame],
                "clean": [arr(b.clean) for b in s.truth],
            }
            for s in bank
        ],
    }


def _run_r(
    bank: list[bq.Series], grid: list[dict[str, float | str]], work: Path, name: str
) -> dict:
    request, response = work / f"{name}_request.json", work / f"{name}_response.json"
    request.write_text(json.dumps(_request(bank, grid)), encoding="utf-8")
    subprocess.run(["Rscript", str(DRIVER), str(request), str(response)], check=True, cwd=ROOT)
    return json.loads(response.read_text(encoding="utf-8"))


def _per_cell(response: dict, k: int) -> dict[str, list[dict]]:
    return {sid: entry["cells"][k] for sid, entry in response.items()}


def _truth(response: dict) -> dict[str, list[dict]]:
    return {sid: entry["truth"] for sid, entry in response.items()}


def cmd_qualify(a: argparse.Namespace) -> int:
    noise = json.loads(a.noise.read_text(encoding="utf-8"))
    sd = float(noise["noise_sd"])
    a.out.mkdir(parents=True, exist_ok=True)
    grid = bq.cells()
    dev = bq.make_bank(DEV_SEED, a.per_stratum, sd)
    free = bq.make_bank(FREE_SEED, a.per_stratum, 0.0, scales=(0.0,), artifacts_on=False)
    r_dev = _run_r(dev, grid, a.out, "dev")
    r_free = _run_r(free, grid, a.out, "free")
    results = []
    for k, cell in enumerate(grid):
        admissible = bq.exact_on_supported(free, _per_cell(r_free, k))
        for support in bq.SUPPORTS:
            results.append(
                {
                    "order": len(results),
                    "cell": cell,
                    "support": support,
                    "admissible": admissible,
                    "dev": bq.score(dev, _per_cell(r_dev, k), _truth(r_dev), support, 0),
                }
            )
    (a.out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    ranked = bq.rank(results)
    if not ranked:
        print("qualify: no admissible cell — the method or domain changes by amendment")
        return 1
    best = ranked[0]
    k = grid.index(best["cell"])

    def peak_accuracy(p: int) -> float:
        dev_all = bq.score(dev, _per_cell(r_dev, k), _truth(r_dev), best["support"], p)["all"]
        return dev_all["peaks_exact"]

    prominence = max(range(len(bq.PROMINENCES)), key=lambda p: (peak_accuracy(p), -p))
    qual = bq.make_bank(QUAL_SEED, a.per_stratum, sd)
    r_qual = _run_r(qual, [best["cell"]], a.out, "qual")
    report = {
        "noise_sd": sd,
        "per_stratum": a.per_stratum,
        "seeds": {"dev": DEV_SEED, "free": FREE_SEED, "qual": QUAL_SEED},
        "admissible_cells": sum(r["admissible"] for r in results) // len(bq.SUPPORTS),
        "cells": len(grid),
        "chosen": best["cell"]
        | {"support": best["support"], "prominence": bq.PROMINENCES[prominence]},
        "dev_all": best["dev"]["all"],
        "runner_up": [
            {"cell": r["cell"], "support": r["support"], "dev_all": r["dev"]["all"]}
            for r in ranked[1:4]
        ],
        "qualification": bq.score(
            qual, _per_cell(r_qual, 0), _truth(r_qual), best["support"], prominence
        ),
    }
    (a.out / "qualification.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    keys = ("chosen", "dev_all", "admissible_cells")
    print(json.dumps({key: report[key] for key in keys}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("measure-noise")
    s.add_argument("--run", type=Path, default=Path("output/corpus-2d"))
    s.add_argument("--out", type=Path)
    s = sub.add_parser("qualify")
    s.add_argument("--noise", type=Path, required=True)
    s.add_argument("--per-stratum", type=int, default=4)
    s.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    return {"measure-noise": cmd_measure_noise, "qualify": cmd_qualify}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
