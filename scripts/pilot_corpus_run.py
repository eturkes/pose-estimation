#!/usr/bin/env python3
"""Stratified corpus-run pilot — the M2.8.1 precondition measurement (P17/P18).

Rehearses M2.8.2's route over a deterministic stratified event sample: the
session run path B1 fixed, then ``analysis/clinical_features.R``.  It measures
and certifies nothing about the corpus (contract D07).

Published surface is redaction-safe aggregates alone — per-asset rows are keyed
by an ordinal and carry stratum labels the qualification census already
publishes, never a filename, path, capture id or camera name.  Subprocess
output does carry identifiers, so it goes to a log file under ``--out`` and is
never echoed.

Rerun (source the accelerator env first, so pose inference reaches the NPU)::

    source /var/home/eturkes/.local/app/intel-accel/env.sh
    PYTHONPATH="$PWD/src:$PYTHONPATH" .venv/bin/python scripts/pilot_corpus_run.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pose_estimation import inventory as inventory_module
from pose_estimation import qualify as qualify_module
from pose_estimation.corpus_run import pose_config, read_pose_config, write_pose_config
from pose_estimation.multicam import (
    SessionError,
    discover_session,
    process_session,
    published_overlap,
)
from pose_estimation.sessions import tree_digest, validate_generation

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = "scripts/pilot_corpus_run.py"
# v2: the report gained `tracker` — generations under different trackers are shaped identically.
# v3: the report gained `drop_lower_body` + `drop_hips_camera`.
GENERATOR_VERSION = "v3"
CLINICAL_R = ROOT / "analysis" / "clinical_features.R"

# Contract P17's three axes. `pts_monotonic` is reported, never required: it is
# a function of the device in this corpus (hevc/iPad Air = 0 on all of them), so
# covering the codec covers it.
AXES = ("codec", "device_config", "rotation_deg")

# Mirrors the directory filter in clinical_features.R, so the pilot counts the
# files that R pass actually read.
R_EXCLUDED = re.compile(
    r"(metrics|kp_detail|diag|summary|smooth|feature_rank"
    r"|clinical[_a-z0-9]*|movement_phases[_a-z0-9]*)\.csv$"
)
GROUP_KEY = ("video", "person_idx")

#: Every key this report may carry (P13/A09).  Frozen and spelled here rather
#: than harvested from the payload: a set derived from what was emitted admits
#: whatever leaks into it, which is the self-authorising arm M2.8.3 A10 deleted.
#: A key outside this set is admissible only as a published label — a stratum
#: value keying a coverage block, an R reason code keying a drop census.
REPORT_FIELDS = frozenset(
    (
        "generator",
        "generator_version",
        "configuration",
        "population",
        "selection",
        "cfr",
        "partition",
        "throughput",
        "guard",
        "assets",
        "verdicts",
        "model",
        "tracking",
        "tracker",
        "drop_lower_body",
        "drop_hips_camera",
        "det_device",
        "pose_device",
        "det_frequency",
        "single_subject",
        "max_frames",
        "seed",
        "min_assets",
        "events",
        "reported_frames",
        "coverage",
        "corpus",
        "pilot",
        "codec",
        "device_config",
        "rotation_deg",
        "pts_monotonic",
        "frames_decoded",
        "index_fallback",
        "monotonic_forced",
        "pooled_fallback_rate",
        "per_asset_rate",
        "assets_with_fallback",
        "mean",
        "median",
        "min",
        "max",
        "groups_input",
        "groups_windowed",
        "groups_dropped",
        "groups_in_both",
        "groups_in_neither",
        "window_rows",
        "drop_reasons",
        "run_wall_s",
        "clinical_wall_s",
        "frames_per_s_incl_startup",
        "latency_ms_mean",
        "steady_frames_per_s",
        "corpus_hours_incl_startup",
        "corpus_hours_steady",
        "events_probed",
        "default_output_refused",
        "i",
        "pts_accepted",
        "cfr_fallback_rate",
        "latency_ms_p95",
        "strata_covered",
        "generation_digest_unmoved",
        "partition_total",
        "partition_disjoint",
        "group_qc_header_frozen",
        "diagnostics_complete",
    )
)


class PilotError(RuntimeError):
    """A redacted pilot failure."""


@dataclass(frozen=True)
class Asset:
    asset_id: str
    event_id: str
    camera_name: str
    codec: str
    device_config: str
    rotation_deg: int
    pts_monotonic: int
    reported_frames: int


@dataclass(frozen=True)
class AssetRun:
    asset: Asset
    frames_decoded: int
    pts_accepted: int
    index_fallback: int
    monotonic_forced: int
    latency_ms_mean: float | None
    latency_ms_p95: float | None

    @property
    def cfr_fallback_rate(self) -> float:
        total = self.frames_decoded
        return (self.index_fallback + self.monotonic_forced) / total if total else 0.0


@dataclass
class Partition:
    n_input: int = 0
    n_windowed: int = 0
    n_dropped: int = 0
    n_overlap: int = 0
    n_unaccounted: int = 0
    window_rows: int = 0
    reasons: Counter[str] = field(default_factory=Counter)
    header_frozen: bool = True

    def merge(self, other: Partition) -> None:
        self.n_input += other.n_input
        self.n_windowed += other.n_windowed
        self.n_dropped += other.n_dropped
        self.n_overlap += other.n_overlap
        self.n_unaccounted += other.n_unaccounted
        self.window_rows += other.window_rows
        self.reasons.update(other.reasons)
        self.header_frozen &= other.header_frozen


def _read_rows(path: Path, label: str) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise PilotError(f"cannot read the {label} table") from exc


def _load_assets(inventory: Path, qualification: Path, sessions: Path) -> list[Asset]:
    """Join the three published tables into the placed-asset population."""
    inventory_rows = {
        row["asset_id"]: row
        for row in _read_rows(inventory / "assets.csv", "inventory")
        if row.get("disposition") == "canonical"
    }
    qualification_rows = {
        row["asset_id"]: row for row in _read_rows(qualification / "assets_qc.csv", "qualification")
    }
    assets: list[Asset] = []
    for row in _read_rows(sessions / "placements.csv", "placements"):
        if row.get("placement") != "placed":
            continue
        source = inventory_rows.get(row["asset_id"])
        measured = qualification_rows.get(row["asset_id"])
        if source is None or measured is None:
            raise PilotError("a placed asset is missing its inventory or qualification row")
        try:
            rotation = int(source["reported_rotation_deg"])
            frames = int(source["reported_frame_count"])
            monotonic = int(measured["pts_monotonic"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PilotError("a placed asset carries an invalid stratum field") from exc
        codec = measured.get("codec", "").strip()
        device_config = measured.get("device_config", "").strip()
        if not codec or not device_config or rotation % 90 or frames <= 0:
            raise PilotError("a placed asset carries an invalid stratum field")
        assets.append(
            Asset(
                asset_id=row["asset_id"],
                event_id=row["event_id"],
                camera_name=row["camera_name"],
                codec=codec,
                device_config=device_config,
                rotation_deg=rotation,
                pts_monotonic=monotonic,
                reported_frames=frames,
            )
        )
    if not assets:
        raise PilotError("the session tree places no asset")
    return sorted(assets, key=lambda asset: asset.asset_id)


def _rank(seed: int, event_id: str, purpose: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{purpose}\0{event_id}".encode()).digest()


def _values(assets: list[Asset], axis: str) -> set[Any]:
    return {getattr(asset, axis) for asset in assets}


def _select_events(assets: list[Asset], min_assets: int, seed: int) -> list[str]:
    """Pick the event set: axis coverage first, then hash-ranked replication.

    Selection is per event because the run path is per session — a session runs
    every camera it holds, so an asset cannot be sampled alone.  Hash ranking
    rather than a duration or frame-count rule keeps the sample free of a
    length bias while staying reproducible from committed state.
    """
    by_event: dict[str, list[Asset]] = defaultdict(list)
    for asset in assets:
        by_event[asset.event_id].append(asset)
    chosen: list[str] = []

    def covered(axis: str) -> set[Any]:
        return {getattr(asset, axis) for event in chosen for asset in by_event[event]}

    for axis in AXES:
        for value in sorted(_values(assets, axis), key=str):
            if value in covered(axis):
                continue
            pool = sorted({asset.event_id for asset in assets if getattr(asset, axis) == value})
            chosen.append(min(pool, key=lambda event: _rank(seed, event, f"{axis}:{value}")))
    for event in sorted(by_event, key=lambda event: _rank(seed, event, "replication")):
        if sum(len(by_event[event]) for event in chosen) >= min_assets:
            break
        if event not in chosen:
            chosen.append(event)

    selected = [asset for event in chosen for asset in by_event[event]]
    for axis in AXES:
        if _values(selected, axis) != _values(assets, axis):
            raise PilotError(f"the selected events do not cover every {axis} value")
    return sorted(chosen)


def _run_event(event_dir: Path, out: Path, log: Path, args: argparse.Namespace) -> float:
    """Run one session through the shipped CLI. Returns wall seconds.

    The event's output tree is destroyed first, so a fresh attempt can never
    credit a diagnostics row or a clinical artifact an older run left behind:
    the pilot reads whatever is on disk after the launch, and a source that
    produces nothing this time reads as one that produced last time's output.
    """
    event_out = out / event_dir.name
    if event_out.exists():
        shutil.rmtree(event_out)
    command = [
        sys.executable,
        "-m",
        "pose_estimation.run",
        "--session-dir",
        str(event_dir),
        "--output-dir",
        str(out),
        "--headless",
        "--model",
        args.model,
        "--tracking",
        args.tracking,
        "--tracker",
        args.tracker,
        "--det-device",
        args.det_device,
        "--pose-device",
        args.pose_device,
        "--det-frequency",
        str(args.det_frequency),
    ]
    if args.single_subject:
        command.append("--single-subject")
    if args.drop_lower_body:
        command.append("--drop-lower-body")
    if args.drop_hips_camera:
        # One argv item: a token beginning with "-" must not read as another option.
        command.append(f"--drop-hips-camera={args.drop_hips_camera}")
    if args.max_frames:
        command += ["--max-frames", str(args.max_frames)]
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        code = subprocess.call(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if code != 0:
        raise PilotError(f"a session run exited {code}; its log is at {log}")
    write_pose_config(event_out, pose_config(args))
    return time.monotonic() - started


def _run_clinical(event_out: Path, log: Path) -> float:
    """Run the R clinical pass over one session's per-camera CSVs."""
    environment = os.environ.copy()
    environment.update(LC_ALL="C", LANG="C", TZ="UTC")
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        code = subprocess.call(
            ["Rscript", str(CLINICAL_R), str(event_out)],
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if code != 0:
        raise PilotError(f"the clinical pass exited {code}; its log is at {log}")
    return time.monotonic() - started


def _diagnostics(event_out: Path, assets: dict[tuple[str, str], Asset]) -> list[AssetRun]:
    runs: list[AssetRun] = []
    for path in sorted(event_out.glob("*_diag.csv")):
        rows = _read_rows(path, "diagnostics")
        if len(rows) != 1:
            raise PilotError("a diagnostics file does not carry exactly one row")
        row = rows[0]
        event_id, _, camera_name = row["video"].partition("/")
        asset = assets.get((event_id, camera_name))
        if asset is None:
            raise PilotError("a diagnostics row names no placed asset")
        latency_mean = row["latency_ms_mean"]
        latency_p95 = row["latency_ms_p95"]
        runs.append(
            AssetRun(
                asset=asset,
                frames_decoded=int(row["n_frames_decoded"]),
                pts_accepted=int(row["pts_accepted"]),
                index_fallback=int(row["index_fallback"]),
                monotonic_forced=int(row["monotonic_forced"]),
                latency_ms_mean=float(latency_mean) if latency_mean else None,
                latency_ms_p95=float(latency_p95) if latency_p95 else None,
            )
        )
    return runs


def _groups(path: Path) -> set[tuple[str, ...]]:
    """Distinct (video, person_idx) keys in a CSV, or the empty set if absent."""
    if not path.is_file():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header is None:
            raise PilotError("a produced CSV carries no header")
        try:
            columns = [header.index(name) for name in GROUP_KEY]
        except ValueError as exc:
            raise PilotError("a produced CSV carries no group key") from exc
        return {tuple(row[index] for index in columns) for row in reader if row}


def _r_constants() -> tuple[tuple[str, ...], frozenset[str]]:
    """Re-derive the group-qc header and reason codes from the R source (P15)."""
    source = CLINICAL_R.read_text(encoding="utf-8")
    reasons = re.search(r"GROUP_QC_REASONS <- c\((.*?)\)", source, re.S)
    schema = re.search(r"group_qc_schema <- function\(\) \{\s*tibble\((.*?)\)\s*\}", source, re.S)
    if reasons is None or schema is None:
        raise PilotError("the R source no longer declares the group-qc constants")
    header = tuple(re.findall(r"(\w+)\s*=\s*\w+\(\)", schema.group(1)))
    codes = frozenset(re.findall(r'"([a-z_]+)"', reasons.group(1)))
    if not header or not codes:
        raise PilotError("the R group-qc constants no longer parse")
    return header, codes


def _partition(event_out: Path, header: tuple[str, ...], codes: frozenset[str]) -> Partition:
    """Measure D05 on real output: every input group reaches exactly one outcome."""
    total = Partition()
    inputs = [
        path
        for path in sorted(event_out.glob("*.csv"))
        if not R_EXCLUDED.search(path.name) and path.name != "world3d.csv"
    ]
    if not inputs:
        raise PilotError("a session produced no landmark CSV")
    for path in inputs:
        stem = str(path)[: -len(".csv")]
        qc_path = Path(f"{stem}_clinical_group_qc.csv")
        window_path = Path(f"{stem}_clinical_windows.csv")
        if not qc_path.is_file():
            # P14: the disposition artifact publishes in both modes, always.
            raise PilotError("a landmark CSV produced no group-disposition artifact")
        dropped_rows = _read_rows(qc_path, "group disposition")
        dropped = {(row["video"], row["person_idx"]) for row in dropped_rows}
        windowed = _groups(window_path)
        given = _groups(path)
        one = Partition()
        one.n_input = len(given)
        one.n_windowed = len(windowed)
        one.n_dropped = len(dropped_rows)
        one.n_overlap = len(windowed & dropped)
        one.n_unaccounted = len(given - windowed - dropped)
        one.window_rows = len(_read_rows(window_path, "windows")) if window_path.is_file() else 0
        one.reasons.update(row["drop_reason"] for row in dropped_rows)
        with qc_path.open(newline="", encoding="utf-8") as stream:
            one.header_frozen = tuple(next(csv.reader(stream), [])) == header
        if not codes.issuperset(one.reasons):
            raise PilotError("a group disposition carries an unlisted reason code")
        total.merge(one)
    return total


def _guard_verdicts(sessions: Path, event_ids: list[str]) -> dict[str, Any]:
    """P02 on the real published tree: the unguarded default must refuse."""

    def never(**_kwargs: Any) -> None:
        raise PilotError("the containment guard let a camera run against the published tree")

    refused = 0
    for event_id in event_ids:
        session = discover_session(sessions / event_id)
        try:
            process_session(session, camera_processor=never, output_dir=None)
        except SessionError:
            refused += 1
    return {"events_probed": len(event_ids), "default_output_refused": refused}


def _stat(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "mean": round(statistics.fmean(values), 6),
        "median": round(statistics.median(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def _asset_payload(index: int, run: AssetRun) -> dict[str, Any]:
    """One measured asset, keyed by an ordinal — no identifier reaches here."""
    return {
        "i": index,
        "codec": run.asset.codec,
        "device_config": run.asset.device_config,
        "rotation_deg": run.asset.rotation_deg,
        "pts_monotonic": run.asset.pts_monotonic,
        "frames_decoded": run.frames_decoded,
        "pts_accepted": run.pts_accepted,
        "index_fallback": run.index_fallback,
        "monotonic_forced": run.monotonic_forced,
        "cfr_fallback_rate": round(run.cfr_fallback_rate, 6),
        "latency_ms_mean": run.latency_ms_mean,
        "latency_ms_p95": run.latency_ms_p95,
    }


def _coverage(population: list[Asset], selected: list[Asset]) -> dict[str, Any]:
    coverage: dict[str, Any] = {}
    for axis in (*AXES, "pts_monotonic"):
        corpus = Counter(str(getattr(asset, axis)) for asset in population)
        sample = Counter(str(getattr(asset, axis)) for asset in selected)
        coverage[axis] = {
            value: {"corpus": corpus[value], "pilot": sample[value]} for value in sorted(corpus)
        }
    return coverage


@dataclass
class Redaction:
    """Every string a report emits that neither admissibility clause carries."""

    keys: list[str] = field(default_factory=list)
    values: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    strings: int = 0

    @property
    def clean(self) -> bool:
        return not (self.keys or self.values)

    @property
    def nonvacuous(self) -> bool:
        """A report emitting no string satisfies every clause and proves nothing."""
        return self.strings > 0


def redaction_violations(
    payload: Any, allowed: frozenset[str], fields: frozenset[str] = frozenset()
) -> Redaction:
    """Walk a report under P13's composite rule (A09).  Both clauses are membership.

    A value is admissible from *allowed* — the published stratum labels, the R
    reason codes, the disposition codes and the code-authored constants the
    emitting program spells at its own call site.  A key is admissible from
    ``fields | allowed``, so a block keyed by a stratum value passes for the
    same reason that value passes elsewhere.

    No clause tests shape.  A key pattern like ``[a-z][a-z0-9_]*`` is a denylist
    wearing an allowlist's name: it admits every identifier of that shape, so
    one capture id is refused as a value and admitted as a key — same string,
    two verdicts.  This is the function the suite grades, so the oracle and the
    guard cannot disagree.
    """
    verdict = Redaction()
    admissible_keys = fields | allowed

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                verdict.strings += 1
                if key not in admissible_keys:
                    verdict.keys.append(key)
                    verdict.paths.append(path)
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{path}[{index}]")
        elif isinstance(node, str):
            verdict.strings += 1
            if node not in allowed:
                verdict.values.append(node)
                verdict.paths.append(path)

    walk(payload, "$")
    return verdict


def _assert_redacted(
    payload: Any, allowed: frozenset[str], fields: frozenset[str] = frozenset()
) -> None:
    """Refuse any report string this program did not author or publish as a label.

    Failures name the JSON *location* and the count, never the offending string:
    the guard exists because that string may be a subject token, so quoting it
    in an error would be the leak it prevents.  The location alone is what a
    caller needs, and a guard that cannot say where it tripped costs a whole run
    to diagnose.
    """
    verdict = redaction_violations(payload, allowed, fields)
    if not verdict.clean:
        raise PilotError(
            f"the report carries {len(verdict.keys)} key(s) and {len(verdict.values)} "
            f"value(s) outside the redaction allowlist; the first is at {verdict.paths[0]}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the M2.8.1 stratified corpus-run pilot.")
    parser.add_argument("--inventory", type=Path, default=ROOT / "inventory")
    parser.add_argument("--qualification", type=Path, default=ROOT / "qualification")
    parser.add_argument("--sessions", type=Path, default=ROOT / "sessions")
    parser.add_argument("--out", type=Path, default=ROOT / ".scratch" / "pilot-m2u81")
    parser.add_argument("--report", type=Path, default=None, help="Report path (default: <out>).")
    parser.add_argument("--min-assets", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="rtmw-l")
    parser.add_argument("--tracking", default="hands-arms")
    parser.add_argument("--tracker", default="subject", choices=["subject", "rtmlib"])
    parser.add_argument("--drop-lower-body", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-hips-camera", default="above")
    parser.add_argument("--det-device", default="GPU")
    parser.add_argument("--pose-device", default="NPU")
    parser.add_argument("--det-frequency", type=int, default=1)
    parser.add_argument("--single-subject", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--reuse-run",
        action="store_true",
        help="Analyse an existing --out tree instead of decoding again.",
    )
    return parser.parse_args()


def _assert_sinks_outside(sessions: Path, *sinks: Path) -> None:
    """No writable sink may overlap the published session tree, in either direction.

    Checked before the first mkdir and re-checked before the report write: the
    log directory is created before validation and the report is written after
    both digest snapshots, so a containment rule applied once in the middle
    leaves a window at each end where the run corrupts the input it is reading
    or leaves a green witness describing bytes that have since moved.
    """
    for sink in sinks:
        if published_overlap(sink, sessions) is not None:
            raise PilotError("an output sink overlaps the published session tree")


def _assert_sources_validated(inventory: Path, qualification: Path, sessions: Path) -> None:
    """Every source table proves its own generation before a cell of it is read.

    The report's stratum labels and its redaction allowlist are both built from
    qualification cells, so an edited table publishes its own labels and then
    authorises them: the guard would be checking the report against the same
    bytes that corrupted it.  Validating the two upstream generations is what
    makes the allowlist a statement about published data.
    """
    try:
        inventory_module.validate_generation(inventory)
        qualify_module.validate_generation(qualification, inventory_dir=inventory)
    except (inventory_module.InventoryError, qualify_module.QualifyError, OSError) as exc:
        raise PilotError("a source table fails its own generation check") from exc
    validate_generation(sessions, inventory_dir=inventory)


def main() -> int:
    args = _parse_args()
    report = args.report or args.out / "pilot_report.json"
    _assert_sinks_outside(args.sessions, args.out, report)
    _assert_sources_validated(args.inventory, args.qualification, args.sessions)
    assets = _load_assets(args.inventory, args.qualification, args.sessions)
    event_ids = _select_events(assets, args.min_assets, args.seed)
    by_key = {(asset.event_id, asset.camera_name): asset for asset in assets}
    selected = [asset for asset in assets if asset.event_id in set(event_ids)]
    header, codes = _r_constants()

    logs = args.out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    digest_before = tree_digest(args.sessions)
    guard = _guard_verdicts(args.sessions, event_ids)

    run_seconds = 0.0
    clinical_seconds = 0.0
    runs: list[AssetRun] = []
    partition = Partition()
    for index, event_id in enumerate(event_ids):
        event_out = args.out / event_id
        if args.reuse_run and read_pose_config(event_out) != pose_config(args):
            # The report's configuration is this invocation's; reused landmarks
            # from another one would publish under the wrong generation.
            raise PilotError("--reuse-run: an event was produced under another pose configuration")
        if not args.reuse_run:
            run_seconds += _run_event(
                args.sessions / event_id, args.out, logs / f"run-{index:02d}.log", args
            )
            clinical_seconds += _run_clinical(event_out, logs / f"clinical-{index:02d}.log")
        runs.extend(_diagnostics(event_out, by_key))
        partition.merge(_partition(event_out, header, codes))

    validate_generation(args.sessions, inventory_dir=args.inventory)
    digest_after = tree_digest(args.sessions)

    frames = sum(run.frames_decoded for run in runs)
    latencies = [run.latency_ms_mean for run in runs if run.latency_ms_mean is not None]
    corpus_frames = sum(asset.reported_frames for asset in assets)
    steady_fps = 1000.0 / statistics.fmean(latencies) if latencies else 0.0
    verdicts = {
        "strata_covered": True,  # _select_events raises otherwise
        "default_output_refused": guard["default_output_refused"] == guard["events_probed"],
        "generation_digest_unmoved": digest_before == digest_after,
        "partition_total": partition.n_input == partition.n_windowed + partition.n_dropped,
        "partition_disjoint": partition.n_overlap == 0 and partition.n_unaccounted == 0,
        "group_qc_header_frozen": partition.header_frozen,
        "diagnostics_complete": len(runs) == len(selected),
    }
    payload: dict[str, Any] = {
        "generator": GENERATOR,
        "generator_version": GENERATOR_VERSION,
        "configuration": {
            "model": args.model,
            "tracking": args.tracking,
            "tracker": args.tracker,
            "drop_lower_body": args.drop_lower_body,
            "drop_hips_camera": args.drop_hips_camera,
            "det_device": args.det_device,
            "pose_device": args.pose_device,
            "det_frequency": args.det_frequency,
            "single_subject": args.single_subject,
            "max_frames": args.max_frames,
            "seed": args.seed,
            "min_assets": args.min_assets,
        },
        "population": {
            "events": len({asset.event_id for asset in assets}),
            "assets": len(assets),
            "reported_frames": corpus_frames,
        },
        "selection": {
            "events": len(event_ids),
            "assets": len(selected),
            "reported_frames": sum(asset.reported_frames for asset in selected),
            "coverage": _coverage(assets, selected),
        },
        "cfr": {
            "assets": len(runs),
            "frames_decoded": frames,
            "index_fallback": sum(run.index_fallback for run in runs),
            "monotonic_forced": sum(run.monotonic_forced for run in runs),
            "pooled_fallback_rate": round(
                sum(run.index_fallback + run.monotonic_forced for run in runs) / frames, 8
            )
            if frames
            else 0.0,
            "per_asset_rate": _stat([run.cfr_fallback_rate for run in runs]),
            "assets_with_fallback": sum(1 for run in runs if run.cfr_fallback_rate > 0),
        },
        "partition": {
            "groups_input": partition.n_input,
            "groups_windowed": partition.n_windowed,
            "groups_dropped": partition.n_dropped,
            "groups_in_both": partition.n_overlap,
            "groups_in_neither": partition.n_unaccounted,
            "window_rows": partition.window_rows,
            "drop_reasons": dict(sorted(partition.reasons.items())),
        },
        "throughput": {
            "run_wall_s": round(run_seconds, 2),
            "clinical_wall_s": round(clinical_seconds, 2),
            "frames_per_s_incl_startup": round(frames / run_seconds, 3) if run_seconds else None,
            "latency_ms_mean": _stat(latencies),
            "steady_frames_per_s": round(steady_fps, 3),
            "corpus_hours_incl_startup": round(corpus_frames / (frames / run_seconds) / 3600, 2)
            if run_seconds and frames
            else None,
            "corpus_hours_steady": round(corpus_frames / steady_fps / 3600, 2)
            if steady_fps
            else None,
        },
        "guard": guard,
        "assets": [_asset_payload(index, run) for index, run in enumerate(runs)],
        "verdicts": verdicts,
    }
    labels = _values(assets, "rotation_deg") | _values(assets, "pts_monotonic")
    _assert_redacted(
        payload,
        frozenset(
            {
                GENERATOR,
                GENERATOR_VERSION,
                args.model,
                args.tracking,
                args.tracker,
                args.drop_hips_camera,
                args.det_device,
                args.pose_device,
            }
            | set(codes)
            | {asset.codec for asset in assets}
            | {asset.device_config for asset in assets}
            | {str(value) for value in labels}
        ),
        REPORT_FIELDS,
    )

    _assert_sinks_outside(args.sessions, report)
    report.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    # Per-asset rows print compact: the report is a reader's artifact, and an
    # indented row block costs more than it says.
    print(json.dumps({key: value for key, value in payload.items() if key != "assets"}, indent=2))
    print(json.dumps(payload["assets"]))
    return 0 if all(verdicts.values()) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PilotError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
