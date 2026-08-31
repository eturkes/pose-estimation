#!/usr/bin/env python3
"""Compare event-level synchronization solvers over the published M2 artifacts.

The probe validates every upstream generation before reading rows. It prints
redaction-safe aggregates only.

Usage: probe_alignment_solver.py [--qualification DIR] [--sessions DIR]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import pathlib
import statistics
import sys
import tempfile
import wave
from typing import Any

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pose_estimation import inventory, qualify
from pose_estimation.measure import audio_offset

FRAME_S = 1.0 / 30.0


def _read_rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _distribution(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max": ordered[-1],
    }


def _tally(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _rank_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None

    def ranks(values: list[float]) -> np.ndarray:
        order = sorted(range(len(values)), key=values.__getitem__)
        result = np.empty(len(values), dtype=np.float64)
        start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and values[order[stop]] == values[order[start]]:
                stop += 1
            result[order[start:stop]] = (start + stop - 1) / 2
            start = stop
        return result

    return float(np.corrcoef(ranks(left), ranks(right))[0, 1])


def _sign_counts(values: list[float], tolerance: float = 1e-12) -> dict[str, int]:
    return {
        "negative": sum(bool(value < -tolerance) for value in values),
        "zero": sum(bool(abs(value) <= tolerance) for value in values),
        "positive": sum(bool(value > tolerance) for value in values),
    }


def _rebase(offsets: dict[str, float], reference: str) -> dict[str, float]:
    origin = offsets[reference]
    return {asset: value - origin for asset, value in offsets.items()}


def _nearest_frame(offset_s: float, frame_rate_hz: float) -> int:
    return round(offset_s * frame_rate_hz)


def _members(placements: list[dict[str, str]]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for row in placements:
        if row["placement"] == "placed":
            grouped.setdefault(row["event_id"], []).append(row["asset_id"])
    return {event_id: tuple(sorted(asset_ids)) for event_id, asset_ids in grouped.items()}


def _view_reference(members: tuple[str, ...], camera_names: dict[str, str]) -> str:
    for camera in ("cam-above", "cam-left", "cam-right"):
        matches = [asset for asset in members if camera_names[asset] == camera]
        if matches:
            return min(matches)
    return members[0]


def _degree_reference(
    members: tuple[str, ...], edges: list[tuple[str, str, float, dict[str, str]]]
) -> tuple[str, bool]:
    degree = dict.fromkeys(members, 0)
    for first, second, _, _ in edges:
        degree[first] += 1
        degree[second] += 1
    maximum = max(degree.values())
    candidates = [member for member in members if degree[member] == maximum]
    return min(candidates), len(candidates) == 1


def _pair_map(rows: list[dict[str, str]]) -> dict[frozenset[str], dict[str, str]]:
    return {frozenset((row["asset_a"], row["asset_b"])): row for row in rows}


def _accepted_edges(
    members: tuple[str, ...], pairs: dict[frozenset[str], dict[str, str]]
) -> list[tuple[str, str, float, dict[str, str]]]:
    edges: list[tuple[str, str, float, dict[str, str]]] = []
    for first, second in itertools.combinations(members, 2):
        row = pairs.get(frozenset((first, second)))
        if row is None or row["status"] not in qualify.QUALIFIED_PAIR_STATUSES:
            continue
        a, b = row["asset_a"], row["asset_b"]
        edges.append((a, b, float(row["offset_s"]), row))
    return edges


def _directed(edges: list[tuple[str, str, float, dict[str, str]]]) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], float] = {}
    for first, second, offset, _ in edges:
        values[(first, second)] = offset
        values[(second, first)] = -offset
    return values


def _component_count(
    members: tuple[str, ...], edges: list[tuple[str, str, float, dict[str, str]]]
) -> int:
    adjacency = {member: set() for member in members}
    for first, second, _, _ in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    unseen = set(members)
    components = 0
    while unseen:
        components += 1
        frontier = [unseen.pop()]
        while frontier:
            current = frontier.pop()
            reached = adjacency[current] & unseen
            unseen -= reached
            frontier.extend(reached)
    return components


def _bfs_offsets(
    members: tuple[str, ...], edges: list[tuple[str, str, float, dict[str, str]]]
) -> dict[str, float] | None:
    if not members:
        return None
    directed = _directed(edges)
    solved = {members[0]: 0.0}
    frontier = [members[0]]
    while frontier:
        current = frontier.pop(0)
        for other in members:
            if other in solved or (current, other) not in directed:
                continue
            solved[other] = solved[current] + directed[(current, other)]
            frontier.append(other)
    return solved if len(solved) == len(members) else None


def _least_squares_system(
    members: tuple[str, ...],
    edges: list[tuple[str, str, float, dict[str, str]]],
    reference: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    unknown = [member for member in members if member != reference]
    columns = {member: index for index, member in enumerate(unknown)}
    matrix = np.zeros((len(edges), len(unknown)), dtype=np.float64)
    values = np.empty(len(edges), dtype=np.float64)
    for index, (first, second, offset, _) in enumerate(edges):
        if first != reference:
            matrix[index, columns[first]] = -1.0
        if second != reference:
            matrix[index, columns[second]] = 1.0
        values[index] = offset
    return unknown, matrix, values


def _least_squares_offsets(
    members: tuple[str, ...],
    edges: list[tuple[str, str, float, dict[str, str]]],
    reference: str,
    weights: list[float] | None = None,
) -> dict[str, float] | None:
    unknown, matrix, values = _least_squares_system(members, edges, reference)
    if not unknown:
        return {reference: 0.0}
    if weights is not None:
        scale = np.sqrt(np.asarray(weights, dtype=np.float64))
        matrix = matrix * scale[:, None]
        values = values * scale
    solution, _, rank, _ = np.linalg.lstsq(matrix, values, rcond=None)
    if rank != len(unknown):
        return None
    return {reference: 0.0, **dict(zip(unknown, solution, strict=True))}


def _least_squares_standard_errors(
    members: tuple[str, ...],
    edges: list[tuple[str, str, float, dict[str, str]]],
    reference: str,
) -> tuple[int, dict[str, float]] | None:
    unknown, matrix, values = _least_squares_system(members, edges, reference)
    degrees_of_freedom = len(edges) - len(unknown)
    if not unknown or degrees_of_freedom <= 0:
        return None
    solution, _, rank, _ = np.linalg.lstsq(matrix, values, rcond=None)
    if rank != len(unknown):
        return None
    residual = values - matrix @ solution
    variance = float(residual @ residual) / degrees_of_freedom
    covariance = variance * np.linalg.inv(matrix.T @ matrix)
    errors = np.sqrt(np.diag(covariance))
    return degrees_of_freedom, dict(zip(unknown, errors, strict=True))


def _write_wav(path: pathlib.Path, samples: np.ndarray, rate: int) -> None:
    pcm = np.clip(np.rint(samples * 24_000), -32_768, 32_767).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(pcm.tobytes())


def _sign_oracle() -> dict[str, Any]:
    rate = audio_offset.TARGET_RATE
    known_lead_s = 0.375
    lead_samples = round(known_lead_s * rate)
    clip_samples = round(20.0 * rate)
    rng = np.random.default_rng(20260830)
    shared = rng.normal(0.0, 0.25, clip_samples + lead_samples).astype(np.float32)
    earlier = shared[:clip_samples]
    later = shared[lead_samples : lead_samples + clip_samples]
    with tempfile.TemporaryDirectory(prefix="m2u5-sign-") as temporary:
        root = pathlib.Path(temporary)
        cache = root / "cache"
        later_path = root / "later-reference.wav"
        earlier_path = root / "earlier-camera.wav"
        _write_wav(later_path, later, rate)
        _write_wav(earlier_path, earlier, rate)
        audio_offset.ensure_cached(later_path, cache, "later")
        audio_offset.ensure_cached(earlier_path, cache, "earlier")
        forward, _, _, _ = audio_offset.estimate(cache, "later", "earlier")
        reverse, _, _, _ = audio_offset.estimate(cache, "earlier", "later")
    tolerance = 2.0 / audio_offset.COARSE_RATE
    if forward.status != "ok" or reverse.status != "ok":
        raise AssertionError(f"Synthetic sign oracle abstained: {forward.status}, {reverse.status}")
    if abs(forward.lag_s - known_lead_s) > tolerance:
        raise AssertionError("The forward synthetic lag misses its constructed truth.")
    if abs(reverse.lag_s + known_lead_s) > tolerance:
        raise AssertionError("The reverse synthetic lag misses its constructed truth.")
    edge = [("later", "earlier", forward.lag_s, {})]
    solved = _least_squares_offsets(("later", "earlier"), edge, "later")
    if solved is None or solved["earlier"] <= 0:
        raise AssertionError("The composed solver sign contradicts the synthetic lag.")
    return {
        "constructed_earlier_start_s": known_lead_s,
        "estimator_t_earlier_minus_t_later_s": forward.lag_s,
        "reverse_estimator_s": reverse.lag_s,
        "antisymmetry_error_s": abs(forward.lag_s + reverse.lag_s),
        "solver_reference_later_offset_s": solved["later"],
        "solver_earlier_camera_offset_s": solved["earlier"],
        "timestamp_transform": "t_reference = t_camera - offset_s",
        "assertion_tolerance_s": tolerance,
    }


def _load(
    arguments: argparse.Namespace,
) -> tuple[
    list[dict[str, str]],
    dict[str, tuple[str, ...]],
    dict[frozenset[str], dict[str, str]],
    dict[str, str],
    dict[str, float],
]:
    qualify.validate_generation(
        arguments.qualification,
        sessions_dir=arguments.sessions,
        inventory_dir=arguments.inventory,
        measurements_dir=arguments.measurements,
    )
    events = _read_rows(arguments.sessions / "events.csv")
    placements = _read_rows(arguments.sessions / "placements.csv")
    members = _members(placements)
    pairs = _pair_map(_read_rows(arguments.qualification / qualify.PAIRS_QC_FILENAME))
    camera_names = {
        row["asset_id"]: row["camera_name"] for row in placements if row["placement"] == "placed"
    }
    inventory_rows = _read_rows(arguments.inventory / inventory.ASSETS_FILENAME)
    frame_rates = {
        row["asset_id"]: float(row["reported_avg_fps"])
        for row in inventory_rows
        if row["disposition"] == inventory.CANONICAL
    }
    if len(events) != len(members):
        raise ValueError("Every event must have at least one placed camera.")
    if set(camera_names) - frame_rates.keys() or any(
        not np.isfinite(rate) or rate <= 0 for rate in frame_rates.values()
    ):
        raise ValueError("Every placed camera must carry a finite positive header frame rate.")
    return events, members, pairs, camera_names, frame_rates


def evaluate(arguments: argparse.Namespace) -> dict[str, Any]:
    events, members_by_event, pairs, camera_names, frame_rates = _load(arguments)
    edges_by_event: dict[str, list[tuple[str, str, float, dict[str, str]]]] = {}
    bfs_by_event: dict[str, dict[str, float]] = {}
    ls_by_event: dict[str, dict[str, float]] = {}
    weighted_by_event: dict[str, dict[str, float]] = {}
    solved_by_size: dict[str, int] = {}
    failed_by_size: dict[str, int] = {}
    for event in events:
        event_id = event["event_id"]
        members = members_by_event[event_id]
        edges = _accepted_edges(members, pairs)
        edges_by_event[event_id] = edges
        bfs = _bfs_offsets(members, edges)
        ls = _least_squares_offsets(members, edges, members[0])
        weighted = _least_squares_offsets(
            members,
            edges,
            members[0],
            [float(edge[3]["peak_rms"]) for edge in edges],
        )
        target = solved_by_size if bfs is not None else failed_by_size
        target[str(len(members))] = target.get(str(len(members)), 0) + 1
        if len({bfs is None, ls is None, weighted is None}) != 1:
            raise AssertionError("The solvers disagree on connectivity.")
        if bfs is not None and ls is not None and weighted is not None:
            bfs_by_event[event_id] = bfs
            ls_by_event[event_id] = ls
            weighted_by_event[event_id] = weighted

    bfs_values = [value for solved in bfs_by_event.values() for value in solved.values()]
    differences = [
        abs(ls_by_event[event_id][asset] - value)
        for event_id, solved in bfs_by_event.items()
        for asset, value in solved.items()
    ]
    weighted_differences = [
        abs(weighted_by_event[event_id][asset] - value)
        for event_id, solved in ls_by_event.items()
        for asset, value in solved.items()
    ]
    both_estimators = [
        row
        for row in pairs.values()
        if row["status_audio"] == "ok" and row["status_visual"] == "ok"
    ]
    cross_modal_error = [
        abs(float(row["offset_s"]) - float(row["offset_visual_s"])) for row in both_estimators
    ]
    solver_weights = [
        float(edge[3]["peak_rms"]) for edges in edges_by_event.values() for edge in edges
    ]

    uncertainty_counts = {
        "one_camera_gauge_only": 0,
        "two_camera_connected_df0": 0,
        "two_camera_unconnected": 0,
        "three_camera_closed_triangle_df1": 0,
        "three_camera_connected_tree_df0": 0,
        "three_camera_unconnected": 0,
    }
    standard_errors: list[float] = []
    closure_residuals: list[float] = []
    for event in events:
        event_id = event["event_id"]
        members = members_by_event[event_id]
        edges = edges_by_event[event_id]
        solved = ls_by_event.get(event_id)
        if len(members) == 1:
            uncertainty_counts["one_camera_gauge_only"] += 1
        elif len(members) == 2:
            key = "two_camera_connected_df0" if solved is not None else "two_camera_unconnected"
            uncertainty_counts[key] += 1
        elif solved is None:
            uncertainty_counts["three_camera_unconnected"] += 1
        elif len(edges) == 2:
            uncertainty_counts["three_camera_connected_tree_df0"] += 1
        elif len(edges) == 3:
            uncertainty_counts["three_camera_closed_triangle_df1"] += 1
            estimate = _least_squares_standard_errors(members, edges, members[0])
            if estimate is None or estimate[0] != 1:
                raise AssertionError("A closed three-camera event must have one residual degree.")
            standard_errors.extend(estimate[1].values())
            directed = _directed(edges)
            first, second, third = members
            closure_residuals.append(
                abs(
                    directed[(first, second)] + directed[(second, third)] - directed[(first, third)]
                )
            )
        else:
            raise AssertionError("An event has an unsupported accepted-edge topology.")

    edge_drop_movements = {"lowest_id": [], "view_priority": []}
    edge_drop_event_maxima = {"lowest_id": [], "view_priority": []}
    edge_drop_events_over_frame = {"lowest_id": 0, "view_priority": 0}
    edge_drop_solves_over_frame = {"lowest_id": 0, "view_priority": 0}
    triangle_events = 0
    for event in events:
        event_id = event["event_id"]
        members = members_by_event[event_id]
        edges = edges_by_event[event_id]
        if len(members) != 3 or len(edges) != 3:
            continue
        triangle_events += 1
        references = {
            "lowest_id": members[0],
            "view_priority": _view_reference(members, camera_names),
        }
        for name, reference in references.items():
            baseline = _least_squares_offsets(members, edges, reference)
            if baseline is None:
                raise AssertionError("A closed triangle must solve before edge removal.")
            event_movements: list[float] = []
            for removed in range(len(edges)):
                reduced = edges[:removed] + edges[removed + 1 :]
                perturbed = _least_squares_offsets(members, reduced, reference)
                if perturbed is None:
                    raise AssertionError("Dropping one triangle edge must leave a spanning tree.")
                movement = max(abs(perturbed[asset] - baseline[asset]) for asset in members)
                edge_drop_movements[name].append(movement)
                event_movements.append(movement)
                edge_drop_solves_over_frame[name] += int(movement > FRAME_S)
            event_maximum = max(event_movements)
            edge_drop_event_maxima[name].append(event_maximum)
            edge_drop_events_over_frame[name] += int(event_maximum > FRAME_S)

    failure_patterns: list[str] = []
    failed_pair_statuses: list[str] = []
    failed_accepted_edges: list[str] = []
    failed_components: list[str] = []
    for event in events:
        event_id = event["event_id"]
        if event_id in ls_by_event:
            continue
        members = members_by_event[event_id]
        rows = [pairs[frozenset(pair)] for pair in itertools.combinations(members, 2)]
        statuses = [row["status"] for row in rows]
        accepted = sum(status in qualify.QUALIFIED_PAIR_STATUSES for status in statuses)
        rejected = [status for status in statuses if status not in qualify.QUALIFIED_PAIR_STATUSES]
        failed_pair_statuses.extend(rejected)
        failed_accepted_edges.append(str(accepted))
        failed_components.append(str(_component_count(members, edges_by_event[event_id])))
        parts = [f"{status}={count}" for status, count in _tally(rejected).items()]
        failure_patterns.append(f"{len(members)}cam;accepted={accepted};" + ";".join(parts))

    signs = {name: [] for name in ("lowest_id", "latest_start", "view_priority", "highest_degree")}
    view_choices: dict[str, int] = {}
    degree_unique = 0
    for event in events:
        event_id = event["event_id"]
        members = members_by_event[event_id]
        edges = edges_by_event[event_id]
        view_reference = _view_reference(members, camera_names)
        view_name = camera_names[view_reference]
        view_choices[view_name] = view_choices.get(view_name, 0) + 1
        degree_reference, unique = _degree_reference(members, edges)
        degree_unique += int(unique)
        solved = ls_by_event.get(event_id)
        if solved is None:
            continue
        references = {
            "lowest_id": members[0],
            "latest_start": min(solved, key=lambda asset: (solved[asset], asset)),
            "view_priority": view_reference,
            "highest_degree": degree_reference,
        }
        for name, reference in references.items():
            signs[name].extend(_rebase(solved, reference).values())

    frame_differences = {
        "bfs_vs_unweighted": [],
        "unweighted_vs_peak_rms_weighted": [],
        "bfs_vs_peak_rms_weighted": [],
    }
    frame_events_changed = dict.fromkeys(frame_differences, 0)
    compared_frame_rates: list[float] = []
    for event in events:
        event_id = event["event_id"]
        if event_id not in ls_by_event:
            continue
        members = members_by_event[event_id]
        reference = _view_reference(members, camera_names)
        rebased = {
            "bfs": _rebase(bfs_by_event[event_id], reference),
            "unweighted": _rebase(ls_by_event[event_id], reference),
            "weighted": _rebase(weighted_by_event[event_id], reference),
        }
        indices = {
            name: {asset: _nearest_frame(offsets[asset], frame_rates[asset]) for asset in members}
            for name, offsets in rebased.items()
        }
        compared_frame_rates.extend(frame_rates[asset] for asset in members)
        comparisons = {
            "bfs_vs_unweighted": ("bfs", "unweighted"),
            "unweighted_vs_peak_rms_weighted": ("unweighted", "weighted"),
            "bfs_vs_peak_rms_weighted": ("bfs", "weighted"),
        }
        for label, (left, right) in comparisons.items():
            event_differences = [
                abs(indices[left][asset] - indices[right][asset]) for asset in members
            ]
            frame_differences[label].extend(event_differences)
            frame_events_changed[label] += int(any(event_differences))

    above_events = sum(
        any(camera_names[asset] == "cam-above" for asset in members)
        for members in members_by_event.values()
    )
    return {
        "q01_bfs": {
            "events_total": len(events),
            "events_solved": len(bfs_by_event),
            "events_failed": len(events) - len(bfs_by_event),
            "events_solved_by_camera_count": dict(sorted(solved_by_size.items())),
            "events_failed_by_camera_count": dict(sorted(failed_by_size.items())),
            "camera_offsets": _distribution(bfs_values),
            "camera_offsets_abs": _distribution([abs(value) for value in bfs_values]),
        },
        "q02_unweighted_least_squares": {
            "constraint": "x_b - x_a = offset_s; min(asset_id) pinned to x_ref = 0",
            "unpinned_rank_deficiency_per_connected_event": 1,
            "events_compared": len(bfs_by_event),
            "cameras_compared": len(differences),
            "cameras_changed_gt_1e-12_s": sum(bool(value > 1e-12) for value in differences),
            "absolute_difference_s": _distribution(differences),
            "absolute_difference_frames_at_30_hz": _distribution(
                [value / FRAME_S for value in differences]
            ),
        },
        "q03_peak_rms_weighted_least_squares": {
            "weight": "published raw peak_rms; no threshold normalization or squaring",
            "weights": _distribution(solver_weights),
            "events_compared": len(ls_by_event),
            "cameras_compared": len(weighted_differences),
            "cameras_changed_gt_1e-12_s": sum(
                bool(value > 1e-12) for value in weighted_differences
            ),
            "absolute_difference_vs_unweighted_s": _distribution(weighted_differences),
            "absolute_difference_vs_unweighted_frames_at_30_hz": _distribution(
                [value / FRAME_S for value in weighted_differences]
            ),
            "cross_modal_check": {
                "pairs_both_estimators_ok": len(both_estimators),
                "spearman_peak_rms_vs_absolute_disagreement": _rank_correlation(
                    [float(row["peak_rms"]) for row in both_estimators], cross_modal_error
                ),
                "spearman_peak_ratio_vs_absolute_disagreement": _rank_correlation(
                    [float(row["peak_ratio"]) for row in both_estimators], cross_modal_error
                ),
            },
            "inverse_variance_interpretation": "unsupported",
        },
        "q04_reference_rules": {
            "events_total": len(events),
            "camera_offsets_with_solved_sign": len(signs["lowest_id"]),
            "lowest_id": {
                "defined_events": len(events),
                "signs": _sign_counts(signs["lowest_id"]),
            },
            "latest_start_from_accepted_solve": {
                "defined_events": len(ls_by_event),
                "undefined_unconnected_events": len(events) - len(ls_by_event),
                "signs": _sign_counts(signs["latest_start"]),
            },
            "view_priority_above_left_right": {
                "above_only_defined_events": above_events,
                "above_only_shortfall_events": len(events) - above_events,
                "hierarchy_defined_events": len(events),
                "hierarchy_reference_counts": dict(sorted(view_choices.items())),
                "signs": _sign_counts(signs["view_priority"]),
            },
            "highest_accepted_degree_then_lowest_id": {
                "defined_events": len(events),
                "unique_highest_degree_events": degree_unique,
                "tied_events_requiring_lowest_id": len(events) - degree_unique,
                "signs": _sign_counts(signs["highest_degree"]),
            },
        },
        "q05_sign_oracle": _sign_oracle(),
        "q06_solve_uncertainty": {
            "event_topologies": uncertainty_counts,
            "closed_triangle_closure_residual_s": _distribution(closure_residuals),
            "closed_triangle_nonreference_standard_error_s": _distribution(standard_errors),
            "closed_triangle_nonreference_standard_error_ms": _distribution(
                [1000.0 * value for value in standard_errors]
            ),
            "reference_standard_error": "0 by gauge constraint, not a measurement",
            "two_camera_statement": "one edge for one free offset gives zero residual degrees",
            "three_camera_tree_statement": "two edges for two free offsets gives zero residual degrees",
            "three_camera_triangle_statement": (
                "three edges for two free offsets gives one residual degree"
            ),
            "assumptions": "independent homoscedastic zero-mean edge errors",
            "defensible_as_per_camera_uncertainty": False,
        },
        "q07_edge_removal": {
            "threshold_s": FRAME_S,
            "triangle_events": triangle_events,
            "edge_drop_solves": 3 * triangle_events,
            "lowest_id_reference": {
                "events_over_threshold": edge_drop_events_over_frame["lowest_id"],
                "events_denominator": triangle_events,
                "solves_over_threshold": edge_drop_solves_over_frame["lowest_id"],
                "solves_denominator": 3 * triangle_events,
                "movement_s": _distribution(edge_drop_movements["lowest_id"]),
                "event_maximum_movement_s": _distribution(edge_drop_event_maxima["lowest_id"]),
            },
            "view_priority_reference": {
                "events_over_threshold": edge_drop_events_over_frame["view_priority"],
                "events_denominator": triangle_events,
                "solves_over_threshold": edge_drop_solves_over_frame["view_priority"],
                "solves_denominator": 3 * triangle_events,
                "movement_s": _distribution(edge_drop_movements["view_priority"]),
                "event_maximum_movement_s": _distribution(edge_drop_event_maxima["view_priority"]),
            },
        },
        "q08_unconnected_events": {
            "events": len(events) - len(ls_by_event),
            "events_by_camera_count": dict(sorted(failed_by_size.items())),
            "accepted_edge_count_per_event": _tally(failed_accepted_edges),
            "connected_component_count_per_event": _tally(failed_components),
            "event_failure_patterns": _tally(failure_patterns),
            "rejected_pair_statuses": _tally(failed_pair_statuses),
            "rejected_pair_status_denominator": len(failed_pair_statuses),
            "artifact_policy": (
                "retain every event and camera row; publish null offset_s for every member, "
                "graph_connected=0, sync_status=unconnected, and no partial solve"
            ),
        },
        "q09_frame_rounding": {
            "reference": "view priority: cam-above, then cam-left, then cam-right",
            "rounding": "nearest integer at each asset's reported_avg_fps; ties to even",
            "cameras_compared": len(compared_frame_rates),
            "events_compared": len(ls_by_event),
            "frame_rate_hz": _distribution(compared_frame_rates),
            "comparisons": {
                label: {
                    "cameras_changed": sum(difference > 0 for difference in differences),
                    "camera_denominator": len(differences),
                    "events_changed": frame_events_changed[label],
                    "event_denominator": len(ls_by_event),
                    "absolute_frame_index_difference": _distribution(differences),
                }
                for label, differences in frame_differences.items()
            },
        },
        "q10_recommendation": {
            "solver": "unweighted least squares over all fused-accepted event edges",
            "gauge": "pin one reference selected by cam-above > cam-left > cam-right",
            "sign": (
                "x_b-x_a=pair offset; camera offset positive means earlier start; "
                "t_reference=t_camera-offset_s"
            ),
            "unconnected": "all camera offsets null; event retained with explicit failure",
            "rejected_bfs_cost": {
                "redundant_triangle_events_ignored": triangle_events,
                "camera_frame_indices_different": sum(
                    difference > 0 for difference in frame_differences["bfs_vs_unweighted"]
                ),
                "camera_denominator": len(frame_differences["bfs_vs_unweighted"]),
            },
            "rejected_weighted_cost": {
                "inverse_variance_interpretation": "unsupported",
                "camera_frame_indices_different": sum(
                    difference > 0
                    for difference in frame_differences["unweighted_vs_peak_rms_weighted"]
                ),
                "camera_denominator": len(frame_differences["unweighted_vs_peak_rms_weighted"]),
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=pathlib.Path, default=pathlib.Path("qualification"))
    parser.add_argument("--sessions", type=pathlib.Path, default=pathlib.Path("sessions"))
    parser.add_argument("--inventory", type=pathlib.Path, default=pathlib.Path("inventory"))
    parser.add_argument("--measurements", type=pathlib.Path, default=pathlib.Path("measurements"))
    arguments = parser.parse_args(argv)
    print(json.dumps(evaluate(arguments), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
