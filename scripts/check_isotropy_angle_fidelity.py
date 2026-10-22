#!/usr/bin/env python3
"""P16: measure that the isotropic normalisation preserves image-plane angles.

Pixel-space angle is the ground truth.  A similarity normalisation preserves it
exactly; a per-axis one does not.  This reads the published isotropic tree,
recovers pixels by multiplying each stored coordinate by the export's own
divisor ``export.coord_scale`` (one ``max(w, h)`` scalar), and compares three
angles per observation:

* ``pixel``       — the reference, computed from the recovered pixel coordinates
* ``isotropic``   — the same pixels divided by ``max(w, h)``, the shipped form
* ``anisotropic`` — the same pixels divided by ``(w, h)``, the pre-fix form

The isotropic bound is what the fix claims; the anisotropic floor is what stops
the case passing vacuously, because a normalisation that happened to be near
isotropic would satisfy the first conjunct alone.  Every arm derives from the
same recovered pixels, so the CSV's 6 dp rounding cancels and only float
arithmetic remains.  The pre-fix tree is gone (D06); the per-axis arm is the
pre-fix export's arithmetic applied to real landmark geometry, which is all the
non-vacuity floor needs.  A pre-fix tree is refused: multiplying its ``(x/w,
y/h)`` cells by ``max(w, h)`` recovers no pixels.

Asset ids are never published.  The sample is pinned by a digest over the sorted
ids instead, which proves the membership was fixed before the numbers were seen
without putting capture identifiers into a committed artifact.

``inputs`` binds the result to what produced it: this script's bytes, the
validated registry generation, the run tree's manifest + report, and the
sample's landmark CSVs + pose configurations (each folded into one digest in
sample order).  A rerun over moved inputs refuses to overwrite the result (rc 2,
``rm`` it first); a rerun over the same inputs that measures different numbers
fails (rc 1).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import sys
from typing import Any

import numpy as np

from pose_estimation import export, inventory

ROOT = pathlib.Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "inventory"
DEFAULT_TREE = ROOT / "output" / "corpus-2d"
EVIDENCE = ROOT / "tests" / "isotropy_angle_fidelity_results.json"

PER_ASPECT = 4
SIDES = ("left", "right")
JOINTS = ("shoulder", "elbow", "wrist")
# The body schema (`--tracking body`) first, then the pre-fix tree's hands-arms schema.
COLUMN_PREFIXES = ("body", "arm")
# A12: the fix's own claim, and the floor that keeps the case non-vacuous.
ISOTROPIC_P95_MAX_DEG = 1e-6
ANISOTROPIC_MEDIAN_MIN_DEG = 5.0


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_assets(inventory_dir: pathlib.Path) -> list[dict[str, str]]:
    with (inventory_dir / inventory.ASSETS_FILENAME).open(newline="", encoding="utf-8") as stream:
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


def fold_digests(paths: list[pathlib.Path]) -> str:
    """One digest over an ordered file list: content-bound, name-free."""
    return hashlib.sha256("\n".join(_sha256(path) for path in paths).encode("ascii")).hexdigest()


def _require_isotropic_tree(tree: pathlib.Path) -> pathlib.Path:
    """Refuse a tree whose coordinates are not the isotropic form pixels recover from."""
    report = tree / "run_report.json"
    if not report.is_file():
        raise SystemExit(f"the tree has no run report: {report}")
    declared = json.loads(report.read_text(encoding="utf-8"))["configuration"].get(
        "coord_normalization"
    )
    if declared != export.COORD_NORMALIZATION:
        raise SystemExit(
            f"{tree} declares {declared!r}, not {export.COORD_NORMALIZATION!r}; "
            f"max(w, h) recovers pixels from the isotropic form alone"
        )
    return report


def _placements(manifest: pathlib.Path) -> dict[str, tuple[str, str]]:
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
    prefix = next(
        (p for p in COLUMN_PREFIXES if f"{p}_{side}_{JOINTS[0]}_x" in rows[0]), COLUMN_PREFIXES[0]
    )
    columns = [(f"{prefix}_{side}_{joint}_x", f"{prefix}_{side}_{joint}_y") for joint in JOINTS]
    visibility = [f"{prefix}_{side}_{joint}_vis" for joint in JOINTS]
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

    scale = export.coord_scale(height, width)
    isotropic: list[np.ndarray] = []
    anisotropic: list[np.ndarray] = []
    for side in SIDES:
        stored = _side_stack(rows, side)
        if stored is None:
            continue
        pixels = stored * scale
        reference = _angles(pixels[:, 0], pixels[:, 1], pixels[:, 2])
        scaled = pixels / scale
        per_axis = pixels / np.array([width, height], dtype=np.float64)
        isotropic.append(np.abs(_angles(scaled[:, 0], scaled[:, 1], scaled[:, 2]) - reference))
        anisotropic.append(
            np.abs(_angles(per_axis[:, 0], per_axis[:, 1], per_axis[:, 2]) - reference)
        )

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


def stale_inputs(recorded: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Name every binding whose recorded digest differs from the current one."""
    return sorted(
        key for key in current.keys() | recorded.keys() if recorded.get(key) != current.get(key)
    )


def measure(tree: pathlib.Path, inventory_dir: pathlib.Path) -> dict[str, Any]:
    """Return the P16 payload for *tree*, its ``inputs`` block included."""
    report = _require_isotropic_tree(tree)
    generation = inventory.validate_generation(inventory_dir)["generation"]
    sample = frozen_sample(_canonical_assets(inventory_dir))
    manifest = tree / "run_manifest.csv"
    placements = _placements(manifest)

    measured: list[dict[str, Any]] = []
    landmarks: list[pathlib.Path] = []
    configs: list[pathlib.Path] = []
    for ordinal, row in enumerate(sample, start=1):
        placement = placements.get(row["asset_id"])
        if placement is None:
            raise SystemExit(f"sample member {ordinal} is absent from the run manifest")
        event_id, camera_name = placement
        width, height = int(row["reported_width"]), int(row["reported_height"])
        landmarks.append(tree / event_id / f"{camera_name}.csv")
        configs.append(tree / event_id / "pose_config.json")
        stats = measure_asset(landmarks[-1], width, height)
        measured.append({"ordinal": ordinal, "aspect_is_portrait": height > width, **stats})

    worst_isotropic = max(item["isotropic_p95_deg"] for item in measured)
    pooled_anisotropic = float(np.median([item["anisotropic_median_deg"] for item in measured]))
    return {
        "predicate": "P16",
        "inputs": {
            "script_sha256": _sha256(pathlib.Path(__file__).resolve()),
            "registry_generation": dict(sorted(generation.items())),
            "run_manifest_sha256": _sha256(manifest),
            "run_report_sha256": _sha256(report),
            "sample_landmarks_sha256": fold_digests(landmarks),
            "sample_pose_configs_sha256": fold_digests(configs),
        },
        "sample_digest_sha256": sample_digest(sample),
        "sample_size": len(measured),
        "isotropic_p95_max_deg": worst_isotropic,
        "anisotropic_pooled_median_deg": pooled_anisotropic,
        "bounds": {
            "isotropic_p95_max_deg": ISOTROPIC_P95_MAX_DEG,
            "anisotropic_median_min_deg": ANISOTROPIC_MEDIAN_MIN_DEG,
        },
        "per_asset": measured,
        "verdicts": {
            "isotropic_preserves_angles": worst_isotropic < ISOTROPIC_P95_MAX_DEG,
            "anisotropic_distorts_angles": pooled_anisotropic >= ANISOTROPIC_MEDIAN_MIN_DEG,
            "sample_spans_both_aspects": len({item["aspect_is_portrait"] for item in measured})
            == 2,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", type=pathlib.Path, default=DEFAULT_TREE)
    parser.add_argument("--inventory", type=pathlib.Path, default=INVENTORY)
    parser.add_argument("--out", type=pathlib.Path, default=EVIDENCE)
    args = parser.parse_args(argv)

    payload = measure(args.tree, args.inventory)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out.is_file():
        recorded = json.loads(args.out.read_text(encoding="utf-8"))
        moved = stale_inputs(recorded.get("inputs") or {}, payload["inputs"])
        if moved:
            print(f"REFUSED: {args.out.name} was measured against other inputs: {', '.join(moved)}")
            print(f"  rm {args.out} to regenerate")
            return 2
        if args.out.read_text(encoding="utf-8") != rendered:
            print(f"FAIL: same inputs, different result than {args.out.name}")
            return 1
    else:
        args.out.write_text(rendered, encoding="utf-8")

    verdicts = payload["verdicts"]
    print(f"sample {payload['sample_size']} assets, digest {payload['sample_digest_sha256'][:16]}")
    print(
        f"  isotropic   p95 max     {payload['isotropic_p95_max_deg']:.3e} deg "
        f"(bound < {ISOTROPIC_P95_MAX_DEG})"
    )
    print(
        f"  anisotropic pooled median {payload['anisotropic_pooled_median_deg']:.4f} deg "
        f"(floor >= {ANISOTROPIC_MEDIAN_MIN_DEG})"
    )
    for name, ok in verdicts.items():
        print(f"  {'PASS' if ok else 'FAIL'} {name}")
    return 0 if all(verdicts.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
