#!/usr/bin/env python
"""Measure the three candidate sync-accept policies over the published sidecar.

P17b as frozen reads "qualifies a pair on agreement between the two estimators,
never on either alone".  Taken strictly that is a policy nobody has priced: the
audio estimator accepts 210 of 246 within-family pairs and the visual
corroborator 74, so an agreement rule discards most of the corpus.  The
opposite reading trusts audio alone, and the gross-error evidence that motivated
P17b bounds only the *visual* estimator.

This script prices all three readings on the same measurement, so the ruling
rests on connectivity and closure rather than on a reading of the sentence.
Prints redaction-safe aggregates only.

Usage: probe_sync_policy.py [--measurements DIR] [--inventory DIR]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pose_estimation import inventory, measure

FRAME_S = 1.0 / 29.97
# One frame at 30 Hz. Two estimators sharing no signal are called to agree at
# the resolution the pipeline can actually act on, not at an arbitrary epsilon.
AGREE_TOLERANCE_S = FRAME_S


def _float(cell: str) -> float | None:
    try:
        return float(cell)
    except (TypeError, ValueError):
        return None


def load_rows(measurements: pathlib.Path, inventory_dir: pathlib.Path) -> list[dict[str, str]]:
    sidecar = measure.validate(measurements, inventory_dir=inventory_dir)
    return list(measure.load_axis(sidecar, "sync").values())


def load_families(inventory_dir: pathlib.Path) -> dict[str, dict[str, str]]:
    """Return canonical families as ``capture_id -> {asset_id: view}``."""
    text = (inventory_dir / inventory.ASSETS_FILENAME).read_text(encoding="utf-8")
    families: dict[str, dict[str, str]] = {}
    for row in csv.DictReader(text.splitlines(), lineterminator="\n"):
        if row["disposition"] == inventory.CANONICAL:
            families.setdefault(row["capture_id"], {})[row["asset_id"]] = row["view"]
    return families


def _spans(members: tuple[str, ...], edges: set[frozenset[str]]) -> bool:
    """True when *edges* join every member of *members* into one component."""
    if len(members) < 2:
        return False
    seen = {members[0]}
    frontier = [members[0]]
    while frontier:
        current = frontier.pop()
        for other in members:
            if other not in seen and frozenset((current, other)) in edges:
                seen.add(other)
                frontier.append(other)
    return len(seen) == len(members)


def view_recoverable(members: dict[str, str], edges: set[frozenset[str]]) -> bool:
    """P38's rule: one camera per view, joined by cross-view accepted pairs.

    This is the calibration-relevant question and it is *not* "every asset
    aligns".  A family holding two files of one view needs only one of them, so
    recoverability quantifies over one-asset-per-view selections and succeeds if
    any selection spans.  Same-view pairs are excluded from the graph: aligning
    two files of one camera contributes no cross-view geometry.
    """
    by_view: dict[str, list[str]] = {}
    for asset, view in sorted(members.items()):
        by_view.setdefault(view, []).append(asset)
    if len(by_view) < 2:
        return False
    cross = {
        edge
        for edge in edges
        if edge <= members.keys() and len({members[asset] for asset in edge}) == 2
    }
    return any(
        _spans(selection, cross)
        for selection in itertools.product(*(by_view[view] for view in sorted(by_view)))
    )


def all_assets_connected(members: dict[str, str], edges: set[frozenset[str]]) -> bool:
    """Stricter than P38: every asset of the family joined, same-view included."""
    return _spans(tuple(sorted(members)), edges)


def closure_residuals(
    families: dict[str, dict[str, str]], offsets: dict[frozenset[str], float], directed: dict
) -> list[float]:
    """Return |r| for every triangle whose three edges are all accepted.

    Closure certifies self-consistency and never accuracy: acoustic propagation
    delay is an exact cocycle around a triangle, so it cancels identically.
    """
    residuals: list[float] = []
    for members in families.values():
        for triangle in itertools.combinations(sorted(members), 3):
            edges = [frozenset(pair) for pair in itertools.combinations(triangle, 2)]
            if not all(edge in offsets for edge in edges):
                continue
            a, b, c = triangle
            residuals.append(abs(directed[(a, b)] + directed[(b, c)] - directed[(a, c)]))
    return residuals


def evaluate(
    name: str, accepted: list[dict[str, str]], families: dict[str, dict[str, str]], column: str
) -> dict[str, object]:
    edges = {frozenset((row["asset_a"], row["asset_b"])) for row in accepted}
    offsets: dict[frozenset[str], float] = {}
    directed: dict[tuple[str, str], float] = {}
    for row in accepted:
        value = _float(row[column])
        if value is None:
            continue
        key = frozenset((row["asset_a"], row["asset_b"]))
        offsets[key] = value
        directed[(row["asset_a"], row["asset_b"])] = value
        directed[(row["asset_b"], row["asset_a"])] = -value
    multi = {key: value for key, value in families.items() if len(value) > 1}
    multiview = {key: value for key, value in families.items() if len(set(value.values())) > 1}
    residuals = closure_residuals(multi, offsets, directed)
    return {
        "policy": name,
        "pairs_accepted": len(accepted),
        # P38's statistic. The strict column beside it is a different question
        # and the two must never be quoted as one number.
        "families_view_recoverable": sum(view_recoverable(v, edges) for v in multiview.values()),
        "families_multiview": len(multiview),
        "families_all_assets_connected": sum(
            all_assets_connected(value, edges) for value in multi.values()
        ),
        "families_multi_asset": len(multi),
        "triangles_closed": len(residuals),
        "closure_median_ms": round(1000 * statistics.median(residuals), 3) if residuals else None,
        "closure_max_ms": round(1000 * max(residuals), 3) if residuals else None,
        "closure_within_one_frame": sum(value <= FRAME_S for value in residuals),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", default="measurements")
    parser.add_argument("--inventory", default="inventory")
    arguments = parser.parse_args(argv)
    inventory_dir = pathlib.Path(arguments.inventory)
    rows = load_rows(pathlib.Path(arguments.measurements), inventory_dir)
    families = load_families(inventory_dir)

    audio_ok = [row for row in rows if row["status_audio"] == "ok"]
    visual_ok = [row for row in rows if row["status_visual"] == "ok"]
    both = [row for row in rows if row["status_audio"] == "ok" and row["status_visual"] == "ok"]
    visual_only = [row for row in rows if row["status_visual"] == "ok" != row["status_audio"]]

    deltas = []
    for row in both:
        audio, visual = _float(row["offset_audio_s"]), _float(row["offset_visual_s"])
        if audio is not None and visual is not None:
            deltas.append(abs(audio - visual))
    agreeing = [
        row
        for row in both
        if (a := _float(row["offset_audio_s"])) is not None
        and (v := _float(row["offset_visual_s"])) is not None
        and abs(a - v) <= AGREE_TOLERANCE_S
    ]

    contradicted = {(row["asset_a"], row["asset_b"]) for row in both if row not in agreeing}
    contradicted_removed = [
        row for row in audio_ok if (row["asset_a"], row["asset_b"]) not in contradicted
    ]

    gross = []
    for row in visual_only:
        audio, visual = _float(row["offset_audio_s"]), _float(row["offset_visual_s"])
        if audio is not None and visual is not None:
            gross.append(abs(audio - visual))

    report = {
        "pairs": len(rows),
        "audio_ok": len(audio_ok),
        "visual_ok": len(visual_ok),
        "both_ok": len(both),
        "visual_only": len(visual_only),
        "cross_modality_on_both": {
            "n": len(deltas),
            "median_ms": round(1000 * statistics.median(deltas), 3) if deltas else None,
            "p95_ms": round(1000 * sorted(deltas)[int(0.95 * len(deltas))], 3) if deltas else None,
            "max_ms": round(1000 * max(deltas), 3) if deltas else None,
            "within_one_frame": sum(value <= FRAME_S for value in deltas),
            "under_10ms": sum(value <= 0.010 for value in deltas),
        },
        # The visual estimator's held-out control rate is 0/200, and this is the
        # stratum that shows a clean control rate does not bound gross error.
        "visual_only_disagreement": {
            "n": len(gross),
            "max_s": round(max(gross), 3) if gross else None,
            "over_one_second": sum(value > 1.0 for value in gross),
        },
        "policies": [
            evaluate("audio_alone", audio_ok, families, "offset_audio_s"),
            evaluate("agreement_required", agreeing, families, "offset_audio_s"),
            evaluate("visual_alone", visual_ok, families, "offset_visual_s"),
            # The ruled policy: audio estimates, the corroborator holds a veto
            # where it spoke and no vote where it did not.  A pair both accept
            # and disagree on is two independent instruments contradicting each
            # other, and neither is preferred, so it is refused rather than
            # resolved.
            evaluate("audio_corroborated", contradicted_removed, families, "offset_audio_s"),
        ],
        "contradicted_pairs": len(audio_ok) - len(contradicted_removed),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
