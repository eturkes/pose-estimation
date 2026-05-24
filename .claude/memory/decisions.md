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
