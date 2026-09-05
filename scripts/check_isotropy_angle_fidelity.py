#!/usr/bin/env python3
"""P16: measure that the isotropic normalisation preserves image-plane angles.

Pixel-space angle is the ground truth.  A similarity normalisation preserves it
exactly; a per-axis one does not.  This reads the PRE-FIX corpus tree, whose
coordinates are ``(x/w, y/h)``, recovers pixels by multiplying them back, and
compares three angles per observation:

* ``pixel``       — the reference, computed from the recovered pixel coordinates
* ``isotropic``   — the same pixels divided by one ``max(w, h)`` scalar
* ``anisotropic`` — the stored coordinates, i.e. what the pre-fix run published

The isotropic bound is what the fix claims; the anisotropic floor is what stops
the case passing vacuously, because a normalisation that happened to be near
isotropic would satisfy the first conjunct alone.

Reading the shipped tree is deliberate: the demonstration needs REAL landmark
geometry but no inference, so it runs before the corrected re-run is funded
rather than after it.  Recovered pixels carry the CSV's 6 dp rounding, and that
is harmless here — the reference and the isotropic value derive from the SAME
recovered pixels, so the rounding cancels and only float arithmetic remains.

Asset ids are never published.  The sample is pinned by a digest over the sorted
ids instead, which proves the membership was fixed before the numbers were seen
without putting capture identifiers into a committed artifact.

The pre-fix tree was deleted once the corrected re-run validated (D06), so this
script no longer has an input on disk and ``tests/isotropy_angle_fidelity_results
.json`` is the durable artifact (A13).  ``--tree`` therefore refuses any tree whose
report declares the isotropic token: the corrected corpus took the pre-fix tree's
path, and reading it here would compare isotropic bytes against themselves.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import sys

import numpy as np

from pose_estimation import export

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSETS_CSV = ROOT / "inventory" / "assets.csv"
DEFAULT_TREE = ROOT / "output" / "corpus-2d"
EVIDENCE = ROOT / "tests" / "isotropy_angle_fidelity_results.json"

PER_ASPECT = 4
SIDES = ("left", "right")
JOINTS = ("shoulder", "elbow", "wrist")
# A12: the fix's own claim, and the floor that keeps the case non-vacuous.
ISOTROPIC_P95_MAX_DEG = 1e-6
ANISOTROPIC_MEDIAN_MIN_DEG = 5.0


def _canonical_assets() -> list[dict[str, str]]:
    with ASSETS_CSV.open(newline="", encoding="utf-8") as stream:
        return [row for row in csv.DictReader(stream) if row["disposition"] == "canonical"]


def frozen_sample(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return the first *PER_ASPECT* landscape and portrait assets by sorted id.

    Selection is a pure function of the validated registry, so re-running it
    cannot quietly move the sample after a result is known.
    """
    ordered = sorted(rows, key=lambda row: row["asset_id"])
    by_aspect: dict[str, list[dict[str, str]]] = {"landscape": [], "portrait": []}
    for row in ordered:
        width, height = int(row["reported_width"]), int(row["reported_height"])
        if width == height:
            continue
        by_aspect["portrait" if height > width else "landscape"].append(row)
    sample = by_aspect["landscape"][:PER_ASPECT] + by_aspect["portrait"][:PER_ASPECT]
    if len(sample) != 2 * PER_ASPECT:
        raise SystemExit(
            f"the registry cannot fill the sample: "
            f"{len(by_aspect['landscape'])} landscape, {len(by_aspect['portrait'])} portrait"
        )
    return sample


def sample_digest(sample: list[dict[str, str]]) -> str:
    """Pin the sample without publishing its members."""
    joined = "\n".join(sorted(row["asset_id"] for row in sample))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _require_prefix_tree(tree: pathlib.Path) -> None:
    """Refuse a tree that already carries the fix, which would compare it to itself."""
    report = tree / "run_report.json"
    if not report.is_file():
        raise SystemExit(f"the tree has no run report: {report}")
    declared = json.loads(report.read_text(encoding="utf-8"))["configuration"].get(
        "coord_normalization"
    )
    if declared == export.COORD_NORMALIZATION:
        raise SystemExit(
            f"{tree} is already isotropic ({declared}); this measurement needs the pre-fix tree, "
            f"which D06 deleted -- read {EVIDENCE.relative_to(ROOT)} instead"
        )


def _placements(tree: pathlib.Path) -> dict[str, tuple[str, str]]:
    manifest = tree / "run_manifest.csv"
    if not manifest.is_file():
        raise SystemExit(f"the tree has no run manifest: {manifest}")
    with manifest.open(newline="", encoding="utf-8") as stream:
        return {
            row["asset_id"]: (row["event_id"], row["camera_name"]) for row in csv.DictReader(stream)
        }


def _angles(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Return the angle at *b* in degrees, one row per observation."""
    ba, bc = a - b, c - b
    norms = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.einsum("ij,ij->i", ba, bc) / norms
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _side_stack(rows: list[dict[str, str]], side: str) -> np.ndarray | None:
    """Return an (n, 3, 2) array of stored coordinates, or None when unusable."""
    columns = [(f"arm_{side}_{joint}_x", f"arm_{side}_{joint}_y") for joint in JOINTS]
    visibility = [f"arm_{side}_{joint}_vis" for joint in JOINTS]
    if any(name not in rows[0] for pair in columns for name in pair):
        return None

    def _column(name: str) -> np.ndarray:
        return np.array([float(row[name] or "nan") for row in rows], dtype=np.float64)

    stacked = np.stack([np.stack([_column(x), _column(y)], axis=1) for x, y in columns], axis=1)
    seen = np.all(np.stack([_column(name) > 0.0 for name in visibility], axis=1), axis=1)
    usable = seen & np.all(np.isfinite(stacked), axis=(1, 2))
    return stacked[usable] if usable.any() else None


def measure_asset(csv_path: pathlib.Path, width: int, height: int) -> dict[str, float]:
    """Return the per-asset deviation statistics for one landmark CSV."""
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("person_idx") == "0"]
    if not rows:
        raise SystemExit(f"no person_idx 0 rows in {csv_path.name}")

    scale = float(max(width, height))
    isotropic: list[np.ndarray] = []
    anisotropic: list[np.ndarray] = []
    for side in SIDES:
        stored = _side_stack(rows, side)
        if stored is None:
            continue
        pixels = stored * np.array([width, height], dtype=np.float64)
        reference = _angles(pixels[:, 0], pixels[:, 1], pixels[:, 2])
        scaled = pixels / scale
        isotropic.append(np.abs(_angles(scaled[:, 0], scaled[:, 1], scaled[:, 2]) - reference))
        anisotropic.append(np.abs(_angles(stored[:, 0], stored[:, 1], stored[:, 2]) - reference))

    if not isotropic:
        raise SystemExit(f"no usable arm observations in {csv_path.name}")
    iso = np.concatenate(isotropic)
    aniso = np.concatenate(anisotropic)
    iso, aniso = iso[np.isfinite(iso)], aniso[np.isfinite(aniso)]
    return {
        "observations": int(iso.size),
        "isotropic_median_deg": float(np.median(iso)),
        "isotropic_p95_deg": float(np.percentile(iso, 95)),
        "anisotropic_median_deg": float(np.median(aniso)),
        "anisotropic_p95_deg": float(np.percentile(aniso, 95)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", type=pathlib.Path, default=DEFAULT_TREE)
    parser.add_argument("--out", type=pathlib.Path, default=EVIDENCE)
    args = parser.parse_args(argv)

    _require_prefix_tree(args.tree)
    sample = frozen_sample(_canonical_assets())
    digest = sample_digest(sample)
    placements = _placements(args.tree)

    measured: list[dict[str, float]] = []
    for ordinal, row in enumerate(sample, start=1):
        placement = placements.get(row["asset_id"])
        if placement is None:
            raise SystemExit(f"sample member {ordinal} is absent from the run manifest")
        event_id, camera_name = placement
        width, height = int(row["reported_width"]), int(row["reported_height"])
        stats = measure_asset(args.tree / event_id / f"{camera_name}.csv", width, height)
        measured.append({"ordinal": ordinal, "aspect_is_portrait": height > width, **stats})

    worst_isotropic = max(item["isotropic_p95_deg"] for item in measured)
    pooled_anisotropic = float(np.median([item["anisotropic_median_deg"] for item in measured]))
    verdicts = {
        "isotropic_preserves_angles": worst_isotropic < ISOTROPIC_P95_MAX_DEG,
        "anisotropic_distorts_angles": pooled_anisotropic >= ANISOTROPIC_MEDIAN_MIN_DEG,
        "sample_spans_both_aspects": len({item["aspect_is_portrait"] for item in measured}) == 2,
    }
    payload = {
        "predicate": "P16",
        "sample_digest_sha256": digest,
        "sample_size": len(measured),
        "isotropic_p95_max_deg": worst_isotropic,
        "anisotropic_pooled_median_deg": pooled_anisotropic,
        "bounds": {
            "isotropic_p95_max_deg": ISOTROPIC_P95_MAX_DEG,
            "anisotropic_median_min_deg": ANISOTROPIC_MEDIAN_MIN_DEG,
        },
        "per_asset": measured,
        "verdicts": verdicts,
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"sample {len(measured)} assets, digest {digest[:16]}")
    print(f"  isotropic   p95 max     {worst_isotropic:.3e} deg (bound < {ISOTROPIC_P95_MAX_DEG})")
    print(
        f"  anisotropic pooled median {pooled_anisotropic:.4f} deg "
        f"(floor >= {ANISOTROPIC_MEDIAN_MIN_DEG})"
    )
    for name, ok in verdicts.items():
        print(f"  {'PASS' if ok else 'FAIL'} {name}")
    return 0 if all(verdicts.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
