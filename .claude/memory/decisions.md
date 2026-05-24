# Architectural decisions

Append-only log of decisions that future sessions must respect. Always add new entries to the **top** so the newest is read first when the file is loaded selectively.

## Entry schema

```
## YYYY-MM-DD — <short title>

**Context.** What problem prompted the decision.
**Decision.** What was chosen.
**Alternatives considered.** Briefly, what was rejected and why.
**Consequences.** Constraints this places on future work; how to reverse if needed.
**References.** Files, tests, or commits that encode the decision.
```

---

## 2026-05-24 — Clinical Pipeline E2E roadmap: 8-task plan

**Context.** User confirmed: 3-cam footage still incoming, calibration solver deferred. Highest priority is end-to-end clinical pipeline (video → CSV → clinical features → longitudinal analysis). Investigation revealed the rtmlib backend (RTMW-L, default and most capable model) has zero CSV export — completely blocking the R analysis pipeline for any rtmlib-processed footage.
**Decision.** 8-task roadmap prioritizing rtmlib CSV export as the keystone:
1. COCO-WholeBody → MediaPipe keypoint mapping layer (unblocked)
2. Wire CSV export into rtmlib process_source() (blocked by #1)
3. Test CSV schema compat with R pipeline (blocked by #2)
4. Harden R scripts for edge cases (blocked by #3)
5. E2E clinical pipeline smoke test (blocked by #3, parallel with #4)
6. Dependency update + security audit (independent)
7. Proactive refactor: main.py/run.py dedup (blocked by #2)
8. Tech notes drift audit (independent)

Session prompts written to `.claude/prompts/sessions.md` for autonomous execution.
**Alternatives considered.** (a) Unified backend-agnostic CSV schema replacing the MediaPipe schema: rejected — would break all existing R scripts and require coordinated migration. (b) rtmlib-native 133-column CSV + R script dual-schema support: rejected — doubles R script complexity for no user benefit. (c) Map rtmlib output to MediaPipe schema: chosen — R scripts work unchanged, mapping is well-defined, clinical keypoints (shoulders/elbows/wrists/hands) have direct COCO-WholeBody equivalents.
**Consequences.** rtmlib CSV export will match the existing MediaPipe schema column-for-column. Some MediaPipe-specific body keypoints (eye_inner/outer, mouth corners) will be approximated or NaN-filled when produced by rtmlib — this is acceptable since clinical features use only shoulders/elbows/wrists/hands. The mapping layer adds a small abstraction but enables the entire downstream analysis pipeline.
**References.** `.claude/prompts/sessions.md`, `.claude/memory/scratchpad.md` (roadmap notes), task list (8 tasks).

---

## 2026-05-24 — process_session() callback-based orchestration pattern

**Context.** `process_session()` was a `NotImplementedError` stub. Both `main.py` (MediaPipe) and `run.py` (rtmlib) have different per-frame processing pipelines with different state requirements (models/anchors vs PoseTracker/smoother), different output capabilities (CSV export vs latency-only), and different initialization sequences. A single `process_session()` implementation must support both backends without duplicating entry-point-specific logic.
**Decision.** `process_session(session, *, camera_processor, output_dir=None)` accepts a `Callable[..., Any]` callback. The callback is called once per camera with keyword args `source`, `output_csv`, `output_diag`, `video_name`. Each entry point constructs the callback as a closure capturing its pre-initialized state: `main.py` wraps `process_video()` with CSV/diag/metrics writer setup/teardown; `run.py` wraps `process_source()` with smoother reset. Session-level orchestration (output dir creation, camera iteration, progress reporting) stays in `process_session()`. In `run.py`, session dispatch was moved from before model setup to after it, so the PoseTracker and smoother are available for the callback.
**Alternatives considered.** (a) Backend-agnostic `process_session()` that reimplements per-frame processing internally: rejected — massive duplication of the ~500-line processing loops in main.py/run.py. (b) Entry points bypass `process_session()` and iterate cameras inline: rejected — loses the single orchestration point needed for future 3D fusion. (c) Protocol/ABC for the callback: rejected — adds type machinery for no practical benefit when the callback is always constructed adjacent to the call site.
**Consequences.** `process_session()` is now functional for per-camera 2D processing on both backends. 3D fusion (Task #4) will be added inside `process_session()` after per-camera processing, when calibration is present. The rtmlib path produces per-camera latency stats but no CSVs (matching existing single-source rtmlib behavior).
**References.** `multicam.py:349-406`, `main.py:505-578`, `run.py:914-966`, `tests/test_multicam.py:327-395`.

---

## 2026-05-24 — Jitter reduction: outlier rejection, hand smoothing increase, multi-frame detection carry

**Context.** User reported tracking jitter/instability and hand detection drops affecting both MediaPipe and rtmlib backends. Investigation identified three root causes: (1) single-frame keypoint spikes corrupting the velocity estimate in the One Euro filter, (2) hand min_cutoff=1.0 providing near-zero smoothing for slow hand movements, (3) detection-level carry-forward limited to 1 frame causing frequent crop loss.
**Decision.**
1. **Velocity-aware outlier rejection** in both `OneEuroFilter` (smoothing.py) and `_OneEuro` (run.py): before filtering, the unexpected component of displacement (beyond velocity prediction) is clamped per-keypoint to `outlier_cap` pixels (default 30, env: `POSE_BENCH_OUTLIER_CAP`). Predicted movement passes through fully. 0 disables.
2. **Hand min_cutoff lowered** from 1.0 to 0.5 in both `PoseSmoother._hand_mc` and `REGION_PARAMS` (run.py). Doubles the smoothing for slow hand movements while preserving fast-movement responsiveness via beta.
3. **Detection EMA alpha lowered** from 0.5 to 0.35 (`DEFAULT_DET_SMOOTH_ALPHA`). Gives 65% weight to the previous detection's crop, reducing crop-induced landmark jitter.
4. **Multi-frame detection carry-forward** extended from 1 to 3 frames (`DEFAULT_DET_CARRY_FRAMES`, env: `POSE_BENCH_DET_CARRY_FRAMES`). Uses `_carry_count` tracking instead of boolean `_carried`. Velocity prediction shifts carried detections per frame. Score decays 0.7× per frame.
**Alternatives considered.** (a) Kalman filter replacement for One Euro: rejected — One Euro's adaptive cutoff is better suited to the variable-speed hand movements in clinical settings; Kalman requires tuning process/measurement noise. (b) Learned neural smoother (SmoothNet, N-euro): rejected — requires pre-trained weights and adds inference latency. (c) Per-keypoint-group parameters beyond body/hands: deferred — the outlier cap handles the worst offenders (fingertip spikes) without adding region complexity.
**Consequences.** All new parameters are env-var tunable and included in `sweep_default.yaml`. Existing benchmark results remain reproducible via `POSE_BENCH_HAND_MIN_CUTOFF=1.0 POSE_BENCH_DET_SMOOTH_ALPHA=0.5 POSE_BENCH_OUTLIER_CAP=0 POSE_BENCH_DET_CARRY_FRAMES=1`. Three new tests in `test_smoothing.py` (outlier cap behavior) and one updated test in `test_detection.py` (multi-frame carry expiry).
**References.** `smoothing.py:37,69-82`, `run.py:109,153-158,174-187`, `processing.py:55,59,194-211,230-270`, `tests/test_smoothing.py:200-250`, `tests/test_detection.py:52-65`, `sweep_default.yaml`.

---

## 2026-05-24 — CLAUDE.md revision: expanded directives, full agent-write permission

**Context.** User rewrote CLAUDE.md with expanded and refined directives. Key policy change: CLAUDE.md was previously owner-approval-only; it is now fully agent-writable ("rewrite CLAUDE.md at any time"). Other changes: added directory-scoping constraint ("constrain development to the directory you are launched in and its children"), explicit commit-timing rules (commit at end of cohesive work, defer during mid-iteration), security audit scheduling, test suite guidance (permissible but warn against overtesting), KISS/UNIX/refactor guidance, expanded objectivity directive (first principles, scientific method, benchmarking), memory system must be kept up-to-date to avoid drift. Minor wording changes throughout.
**Decision.** Propagate the CLAUDE.md-is-agent-writable policy to all downstream files that previously enforced the approval gate: `.claude/INDEX.md` authoring rules, `AGENTS.md` pointer description, `.claude/prompts/kickoff.md` step 1. Record via this decision entry. Supersedes the approval constraint in the 2026-05-16 decision below.
**Alternatives considered.** None — this is a direct propagation of an owner-authored policy change.
**Consequences.** Agents may now rewrite CLAUDE.md freely when content is obsolete, better phrased, or superseded. All references to the old approval gate are updated. The 2026-05-16 decision entry remains as historical record but its approval constraint is superseded.
**References.** `/CLAUDE.md`, `.claude/INDEX.md:50`, `AGENTS.md:5`, `.claude/prompts/kickoff.md:9`.

---

## 2026-05-18 — Multi-camera (3-cam) scaffolding: Session + Calibration + Triangulation

**Context.** New 3-camera footage is incoming; user wants the codebase prepped. Confirmed: footage arrives as 3 separate video files per session; end goal is 3D triangulated pose; no calibration data yet (workflow needed); deliverable for this round is architecture scaffolding + CLI surface only (no per-view processing, no 3D fusion implementation).
**Decision.**
1. **Session model.** A *session* = one directory containing per-camera video files (`cam1.mp4`, `cam2.mp4`, …) plus an optional `session.json` manifest (camera names, per-camera frame offsets for software sync) and optional `calibration.json`. Sessions live under `videos/<session_id>/`; outputs mirror layout under `output/<session_id>/`.
2. **Calibration model.** Single JSON file holds intrinsics (`K`, `distortion`) + extrinsics (`rvec`, `tvec`) per camera, world-frame reference, and reprojection error. Schema versioned via `format_version`. Resolvable by explicit `--calibration <path>` or auto-discovered from `<session_dir>/calibration.json`.
3. **Modules.** Three new modules: `multicam.py` (Session, iter_synchronized_frames, process_session stub), `calibration.py` (load/save/validate, charuco solver stub), `triangulation.py` (DLT helpers + fuse_session_frame stub). All algorithmic work is `NotImplementedError` stubs with TODO references; only data plumbing is real this round.
4. **CLI.** Add `--session-dir` / `--sessions-dir` / `--calibration` to both `main.py` and `run.py`, mutually exclusive with `--source` / `--batch-dir`. New console script `pose-estimation-calibrate` with `verify` (working), `solve` (stub), `capture` (stub) subcommands.
5. **Generality.** Code is N-camera, not hardcoded to 3. Discovery pattern is `cam*.mp4`; `session.json` lets users override naming.
6. **Synchronization.** Default: assume cameras share frame indices (recorder pre-aligned). `session.json` `sync_offsets` allows integer-frame offset per camera relative to camera 0. Audio-cross-correlation sync is a follow-up.
7. **Backend coverage.** Both MediaPipe (`main.py`) and rtmlib (`run.py`) get the new flags. Triangulation is backend-agnostic — operates on 2D keypoint tensors regardless of producer.

**Alternatives considered.**
- *Composite/side-by-side video as the primary input form.* Rejected — user confirmed 3 separate files. Side-by-side is recoverable by adding a `--split-strategy` later if needed.
- *Hardcode N=3.* Rejected — generalising costs nothing and protects against the inevitable 4-camera setup. Defaults still target 3.
- *Skip the manifest and infer everything from filenames.* Rejected — sync offsets and custom camera names need an out-of-band channel. `session.json` is optional, so the zero-config path still works.
- *Implement charuco solve and triangulation in this pass.* Rejected — out of scope per user direction. Stubs land the surface so the wire-up is a focused follow-up.
- *Reuse `--batch-dir` semantics for sessions.* Rejected — `--batch-dir` iterates files independently; sessions need synchronized iteration. Separate flag prevents semantic overload.

**Consequences.**
- New public API surface (Session, calibration types/helpers, fuse_session_frame). `tests/test_public_api.py` tracks the additions.
- `process_session()` and `fuse_session_frame()` raise `NotImplementedError` — the CLI's new branches will surface that until the follow-up implements per-view processing and 3D fusion. Document this explicitly in `--help`.
- Calibration files may contain identifying lab info; treat as patient-adjacent data. `calibration/` top-level (if it materialises) goes in `.gitignore`. Calibrations inside `videos/<session>/` are already ignored.
- Output layout becomes session-scoped (`output/<session_id>/camN.csv`). Single-source CLI behaviour is preserved.
- Module count grows by 3; `architecture.md` module map must list them.

**References.** `src/pose_estimation/multicam.py`, `src/pose_estimation/calibration.py`, `src/pose_estimation/triangulation.py`, `src/pose_estimation/calibration_cli.py`, `.claude/tech/multicam.md`, `.claude/tech/calibration.md`, `tests/test_multicam.py`, `tests/test_calibration.py`, `pyproject.toml:[project.scripts]`.

---

## 2026-05-16 — Split project context into `CLAUDE.md` (meta) + `.claude/notes/` (tech)

**Context.** A new project-root `CLAUDE.md` introduces meta-instructions for AI agents (memory system, LLM-optimised docs, token efficiency). The existing `AGENTS.md` mixed meta with project-specific tech reference; size was growing and drift from code was accumulating.
**Decision.** Keep `CLAUDE.md` at the project root as the agent meta-instructions document. Move project-specific technical reference into `.claude/tech/*.md` for selective loading. Replace `AGENTS.md` with a pointer for tools that follow the AGENTS.md convention. Add `.claude/memory/` (decisions, lessons, scratchpad) and `.claude/prompts/kickoff.md`.
**Alternatives considered.** (a) Symlink `AGENTS.md → CLAUDE.md`: rejected — mixes meta and tech context and clutters `CLAUDE.md` (which is owner-approval-only). (b) Keep both side by side: rejected — guaranteed drift between two parallel docs. (c) Fold everything into one mega-file: rejected — defeats selective loading and token efficiency.
**Consequences.** All project tech notes live under `.claude/tech/`. `AGENTS.md` is a thin pointer; do not grow it. `CLAUDE.md` modifications require explicit user approval. When `tech/*` content drifts from code, fix it in the same change that introduced the drift.
**References.** `.claude/INDEX.md`, `AGENTS.md`, `/CLAUDE.md`, `.claude/prompts/kickoff.md`.
