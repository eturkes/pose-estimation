#!/usr/bin/env python3
"""Build an independent M2.5 cameras_qc.csv oracle from frozen inputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ACCEPTED_STATUSES = frozenset({"ok_corroborated", "ok_uncorroborated"})
VIEW_ORDER = {"above": 0, "left": 1, "right": 2}
OUTPUT_FIELDS = (
    "event_id",
    "asset_id",
    "camera_name",
    "view",
    "offset_s",
    "offset_status",
    "is_reference",
    "reference_camera",
)
NOMINAL_FRAME_S = 1 / 29.97


@dataclass(frozen=True)
class Edge:
    asset_a: str
    asset_b: str
    offset_s: float


@dataclass
class EventSolution:
    event_id: str
    members: tuple[str, ...]
    reference: str
    reference_view: str
    offsets: dict[str, float]
    components: tuple[tuple[str, ...], ...]
    graph_connected: bool
    edge_residuals: tuple[float, ...]
    redundant: bool
    bfs_offsets: dict[str, float]


@dataclass
class Inputs:
    placements: list[dict[str, str]]
    events: list[dict[str, str]]
    pairs: list[dict[str, str]]
    assets: list[dict[str, str]]
    published_events: list[dict[str, str]]


def read_csv(path: Path, required: Iterable[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = sorted(set(required) - set(fieldnames))
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        return list(reader)


def shuffled(rows: list[dict[str, str]], rng: random.Random | None) -> list[dict[str, str]]:
    result = list(rows)
    if rng is not None:
        rng.shuffle(result)
    return result


def load_inputs(args: argparse.Namespace) -> Inputs:
    rng = random.Random(args.shuffle_seed) if args.shuffle_seed is not None else None
    return Inputs(
        placements=shuffled(
            read_csv(
                args.placements,
                ("asset_id", "event_id", "placement", "camera_name"),
            ),
            rng,
        ),
        events=shuffled(read_csv(args.events, ("event_id",)), rng),
        pairs=shuffled(
            read_csv(args.pairs, ("asset_a", "asset_b", "offset_s", "status")),
            rng,
        ),
        assets=shuffled(read_csv(args.assets, ("asset_id", "view")), rng),
        published_events=shuffled(
            read_csv(args.published_events, ("event_id", "offset_span_s")), rng
        ),
    )


def unique_map(
    rows: Iterable[dict[str, str]], key: str, *, label: str
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row[key]
        if not value:
            raise ValueError(f"{label}: empty {key}")
        if value in result:
            raise ValueError(f"{label}: duplicate {key}")
        result[value] = row
    return result


def finite_float(value: str, *, label: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(f"{label}: invalid float") from error
    if not math.isfinite(number):
        raise ValueError(f"{label}: non-finite float")
    return number


def connected_components(
    members: tuple[str, ...], edges: tuple[Edge, ...]
) -> tuple[tuple[str, ...], ...]:
    adjacency = {asset_id: set() for asset_id in members}
    for edge in edges:
        adjacency[edge.asset_a].add(edge.asset_b)
        adjacency[edge.asset_b].add(edge.asset_a)
    unseen = set(members)
    components: list[tuple[str, ...]] = []
    while unseen:
        start = min(unseen)
        stack = [start]
        component: set[str] = set()
        while stack:
            asset_id = stack.pop()
            if asset_id in component:
                continue
            component.add(asset_id)
            unseen.discard(asset_id)
            stack.extend(sorted(adjacency[asset_id] - component, reverse=True))
        components.append(tuple(sorted(component)))
    return tuple(sorted(components, key=lambda item: (-len(item), item)))


def solve_component(
    component: tuple[str, ...], reference: str, edges: tuple[Edge, ...]
) -> tuple[dict[str, float], tuple[float, ...], bool]:
    if len(component) == 1:
        return {reference: 0.0}, (), False
    component_set = set(component)
    component_edges = tuple(
        edge for edge in edges if edge.asset_a in component_set and edge.asset_b in component_set
    )
    columns = tuple(asset_id for asset_id in component if asset_id != reference)
    column_index = {asset_id: index for index, asset_id in enumerate(columns)}
    matrix = np.zeros((len(component_edges), len(columns)), dtype=np.float64)
    target = np.empty(len(component_edges), dtype=np.float64)
    full_incidence = np.zeros((len(component_edges), len(component)), dtype=np.float64)
    full_index = {asset_id: index for index, asset_id in enumerate(component)}
    for row_index, edge in enumerate(component_edges):
        full_incidence[row_index, full_index[edge.asset_a]] = -1.0
        full_incidence[row_index, full_index[edge.asset_b]] = 1.0
        if edge.asset_a != reference:
            matrix[row_index, column_index[edge.asset_a]] = -1.0
        if edge.asset_b != reference:
            matrix[row_index, column_index[edge.asset_b]] = 1.0
        target[row_index] = edge.offset_s
    expected_rank = len(component) - 1
    if np.linalg.matrix_rank(full_incidence) != expected_rank:
        raise ValueError("connected incidence matrix lacks rank n-1")
    values, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
    if rank != expected_rank:
        raise ValueError("gauge-fixed incidence matrix lacks full column rank")
    offsets = {reference: 0.0}
    offsets.update({asset_id: float(values[index]) for asset_id, index in column_index.items()})
    residuals = tuple(float(value) for value in matrix @ values - target)
    redundant = len(component_edges) > expected_rank
    return offsets, residuals, redundant


def breadth_first_offsets(
    component: tuple[str, ...], reference: str, edges: tuple[Edge, ...]
) -> dict[str, float]:
    component_set = set(component)
    adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for edge in edges:
        if edge.asset_a not in component_set or edge.asset_b not in component_set:
            continue
        adjacency[edge.asset_a].append((edge.asset_b, edge.offset_s))
        adjacency[edge.asset_b].append((edge.asset_a, -edge.offset_s))
    offsets = {reference: 0.0}
    queue = deque([reference])
    while queue:
        asset_id = queue.popleft()
        for neighbor, delta in sorted(adjacency[asset_id]):
            if neighbor in offsets:
                continue
            offsets[neighbor] = offsets[asset_id] + delta
            queue.append(neighbor)
    if set(offsets) != component_set:
        raise ValueError("breadth-first solve missed a connected member")
    return offsets


def build_oracle(
    inputs: Inputs,
) -> tuple[list[dict[str, str]], list[EventSolution], dict[str, Any]]:
    event_rows = unique_map(inputs.events, "event_id", label="events")
    asset_rows = unique_map(inputs.assets, "asset_id", label="assets")
    placed = [row for row in inputs.placements if row["placement"] == "placed"]
    placement_rows = unique_map(placed, "asset_id", label="placed assets")
    members_by_event: dict[str, list[str]] = defaultdict(list)
    for asset_id, row in placement_rows.items():
        event_id = row["event_id"]
        if event_id not in event_rows:
            raise ValueError("placement names an unknown event")
        if asset_id not in asset_rows:
            raise ValueError("placement names an unknown asset")
        if not row["camera_name"]:
            raise ValueError("placed asset has an empty camera name")
        members_by_event[event_id].append(asset_id)
    if set(members_by_event) != set(event_rows):
        raise ValueError("events and placed membership disagree")

    event_by_asset = {asset_id: placement_rows[asset_id]["event_id"] for asset_id in placement_rows}
    accepted_edges: list[Edge] = []
    directed_edges: dict[tuple[str, str], float] = {}
    local_edges_by_event: dict[str, list[Edge]] = defaultdict(list)
    cross_event_accepted = 0
    for row in inputs.pairs:
        if row["status"] not in ACCEPTED_STATUSES:
            continue
        asset_a, asset_b = row["asset_a"], row["asset_b"]
        if asset_a == asset_b:
            raise ValueError("accepted self-edge")
        if asset_a not in event_by_asset or asset_b not in event_by_asset:
            raise ValueError("accepted edge names an unplaced asset")
        offset_s = finite_float(row["offset_s"], label="accepted offset_s")
        edge = Edge(asset_a, asset_b, offset_s)
        accepted_edges.append(edge)
        for key, value in (((asset_a, asset_b), offset_s), ((asset_b, asset_a), -offset_s)):
            previous = directed_edges.get(key)
            if previous is not None and previous != value:
                raise ValueError("accepted directed edge conflicts with a duplicate")
            directed_edges[key] = value
        event_a, event_b = event_by_asset[asset_a], event_by_asset[asset_b]
        if event_a == event_b:
            local_edges_by_event[event_a].append(edge)
        else:
            cross_event_accepted += 1
    accepted_edges.sort(key=lambda edge: (edge.asset_a, edge.asset_b, edge.offset_s))
    for edges in local_edges_by_event.values():
        edges.sort(key=lambda edge: (edge.asset_a, edge.asset_b, edge.offset_s))

    solutions: list[EventSolution] = []
    output_rows: list[dict[str, str]] = []
    for event_id in sorted(event_rows):
        members = tuple(sorted(members_by_event[event_id]))
        cameras = [placement_rows[asset_id]["camera_name"] for asset_id in members]
        if len(cameras) != len(set(cameras)):
            raise ValueError("event repeats a camera name")
        for asset_id in members:
            if asset_rows[asset_id]["view"] not in VIEW_ORDER:
                raise ValueError("placed asset has an unsupported view")
        reference = min(
            members,
            key=lambda asset_id: (VIEW_ORDER[asset_rows[asset_id]["view"]], asset_id),
        )
        edges = tuple(local_edges_by_event[event_id])
        components = connected_components(members, edges)
        reference_component = next(item for item in components if reference in item)
        offsets, residuals, redundant = solve_component(reference_component, reference, edges)
        bfs_offsets = breadth_first_offsets(reference_component, reference, edges)
        solution = EventSolution(
            event_id=event_id,
            members=members,
            reference=reference,
            reference_view=asset_rows[reference]["view"],
            offsets=offsets,
            components=components,
            graph_connected=len(reference_component) == len(members),
            edge_residuals=residuals,
            redundant=redundant,
            bfs_offsets=bfs_offsets,
        )
        solutions.append(solution)
        reference_camera = placement_rows[reference]["camera_name"]
        for asset_id in members:
            solved = asset_id in offsets
            output_rows.append(
                {
                    "event_id": event_id,
                    "asset_id": asset_id,
                    "camera_name": placement_rows[asset_id]["camera_name"],
                    "view": asset_rows[asset_id]["view"],
                    "offset_s": format_offset(offsets[asset_id]) if solved else "",
                    "offset_status": "solved" if solved else "unreachable",
                    "is_reference": "1" if asset_id == reference else "0",
                    "reference_camera": reference_camera,
                }
            )

    antisymmetry_errors = [
        abs(value + directed_edges[(asset_b, asset_a)])
        for (asset_a, asset_b), value in directed_edges.items()
    ]
    diagnostics = {
        "accepted_edges": len(accepted_edges),
        "accepted_local_edges": sum(map(len, local_edges_by_event.values())),
        "accepted_cross_event_edges": cross_event_accepted,
        "directed_edge_entries": len(directed_edges),
        "antisymmetry_violations": sum(error != 0 for error in antisymmetry_errors),
        "antisymmetry_max_error_s": max(antisymmetry_errors, default=0.0),
    }
    return output_rows, solutions, diagnostics


def format_offset(value: float) -> str:
    if value == 0.0:
        return "0"
    return format(value, ".17g")


def distribution(values: Iterable[float]) -> dict[str, float | int]:
    data = sorted(values)
    if not data:
        return {"n": 0, "min": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "n": len(data),
        "min": data[0],
        "median": statistics.median(data),
        "p95": float(np.percentile(data, 95, method="linear")),
        "max": data[-1],
    }


def summarize(
    rows: list[dict[str, str]],
    solutions: list[EventSolution],
    diagnostics: dict[str, Any],
    published_events: list[dict[str, str]],
) -> dict[str, Any]:
    status_tally = Counter(row["offset_status"] for row in rows)
    reference_tally = Counter(solution.reference_view for solution in solutions)
    event_spans = {
        solution.event_id: max(solution.offsets.values()) - min(solution.offsets.values())
        for solution in solutions
    }
    unconnected = [solution for solution in solutions if not solution.graph_connected]
    component_patterns = Counter(
        "+".join(map(str, sorted(map(len, solution.components), reverse=True)))
        for solution in unconnected
    )
    component_sizes = Counter(
        len(component) for solution in unconnected for component in solution.components
    )
    residual_events = [solution for solution in solutions if solution.redundant]
    residuals = [value for solution in residual_events for value in solution.edge_residuals]
    bfs_differences = [
        abs(offset - solution.bfs_offsets[asset_id])
        for solution in solutions
        for asset_id, offset in solution.offsets.items()
    ]
    published = unique_map(published_events, "event_id", label="published events_qc")
    if set(published) != set(event_spans):
        raise ValueError("published events_qc event set differs")
    span_differences: list[float] = []
    published_span_empty = 0
    for event_id, span in event_spans.items():
        cell = published[event_id]["offset_span_s"]
        if not cell:
            published_span_empty += 1
            continue
        span_differences.append(abs(span - finite_float(cell, label="published offset_span_s")))
    references_per_event = Counter(row["event_id"] for row in rows if row["is_reference"] == "1")
    return {
        "rows": len(rows),
        "events": len(solutions),
        "nonempty_offset_s": sum(bool(row["offset_s"]) for row in rows),
        "unreachable": status_tally["unreachable"],
        "offset_status": dict(sorted(status_tally.items())),
        "events_with_one_reference": sum(value == 1 for value in references_per_event.values()),
        "reference_rows_zero": sum(
            row["is_reference"] == "1" and row["offset_s"] == "0" for row in rows
        ),
        "reference_views": dict(sorted(reference_tally.items())),
        "graph_connected_events": sum(solution.graph_connected for solution in solutions),
        "event_offset_span_s": distribution(event_spans.values()),
        "published_span_comparison": {
            "comparable": len(span_differences),
            "published_empty": published_span_empty,
            "different_gt_1e-9": sum(value > 1e-9 for value in span_differences),
            "max_difference_s": max(span_differences, default=0.0),
        },
        "least_squares_redundant_events": len(residual_events),
        "least_squares_edge_residual_abs_s": distribution(map(abs, residuals)),
        "least_squares_global_l2_s": math.sqrt(sum(value * value for value in residuals)),
        "bfs_comparison": {
            "solved_cameras": len(bfs_differences),
            "different_gt_nominal_frame": sum(value > NOMINAL_FRAME_S for value in bfs_differences),
            "different_gt_1e-12": sum(value > 1e-12 for value in bfs_differences),
            "max_difference_s": max(bfs_differences, default=0.0),
            "nominal_frame_s": NOMINAL_FRAME_S,
        },
        "unconnected": {
            "events": len(unconnected),
            "component_patterns": dict(sorted(component_patterns.items())),
            "component_sizes": {str(key): value for key, value in sorted(component_sizes.items())},
            "partial_multi_camera_solutions": sum(
                1 < len(solution.offsets) < len(solution.members) for solution in unconnected
            ),
        },
        **diagnostics,
    }


def write_output(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placements", type=Path, default=Path("sessions/placements.csv"))
    parser.add_argument("--events", type=Path, default=Path("sessions/events.csv"))
    parser.add_argument("--pairs", type=Path, default=Path("qualification/pairs_qc.csv"))
    parser.add_argument("--assets", type=Path, default=Path("inventory/assets.csv"))
    parser.add_argument(
        "--published-events", type=Path, default=Path("qualification/events_qc.csv")
    )
    parser.add_argument("--out", type=Path, default=Path("qualification.orc/cameras_qc.csv"))
    parser.add_argument("--shuffle-seed", type=int)
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = load_inputs(args)
    rows, solutions, diagnostics = build_oracle(inputs)
    write_output(args.out, rows)
    if args.summary:
        print(
            json.dumps(
                summarize(rows, solutions, diagnostics, inputs.published_events), sort_keys=True
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
