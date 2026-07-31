# Roadmap

Live long-horizon state only; completed trajectory belongs in git.

## M2 — real-world 3D clinical validation

**Status: BLOCKED** — no cleared, synchronized three-camera session with calibration; no external clinical baseline.

**Goal:** validate the full real-data chain (calibration → 2D tracking → 3D fusion → clinical metrics), quantify reprojection/drop/confidence/timing/stability behavior, and replace provisional thresholds with evidence-backed values.

**Current evidence boundary:** calibration, fusion, `world3d.csv`, clinical analysis, QA grading, and injected failure modes pass synthetic tests. No claim of real-capture or clinical-metric accuracy is warranted yet.

**Unblock probe:** run `UV_PROJECT_ENVIRONMENT=.venv uv run pose-estimation-run --list-sessions`. A redacted `3 cameras; calibration: present` result identifies a shape-compatible candidate only. Media decoding, synchronization checks, QA, and calibration-value inspection require explicit patient-data clearance.

**Sequence once unblocked:**

1. First cleared capture dry run → execute QA + end-to-end validation; record failure modes; recalibrate capture procedure and provisional thresholds.
2. Multiple-session study → quantify reprojection, tracking gaps, throughput, temporal stability, and inter-trial repeatability.
3. Agreement study → use a known-geometry/goniometer/reference-system baseline when available; otherwise retain the validity gap and specify the cheapest sufficient baseline protocol.
4. Derive only de-identified or synthetic regression fixtures; lock observed failures with tests.

**Acceptance:** reproducible commands + reports trace every claim to cleared inputs; thresholds have explicit evidence; clinical-validity gaps remain visible; all repository validation gates pass.

## Backlog

- 3D-aware downstream aggregation in `analysis/`.
- Cross-camera identity matching for multi-person scenes; fusion currently assumes one subject.
