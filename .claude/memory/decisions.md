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

## 2026-06-15 — Track `.serena/project.yml` only; adopt Scoped Commits

**Context.** CLAUDE.md gained a Headroom bullet declaring `.serena/project.yml` the tracked LSP config while `cache/`, `project.local.yml`, and `memories/` stay local, plus a bullet mandating [Scoped Commits](https://scopedcommits.com/). Serena auto-writes its own `.serena/.gitignore`, which covers only `/cache` + `/project.local.yml` and omits `memories/`.
**Decision.** Centralized the ignore in the root `.gitignore` via negation (`.serena/*` + `!.serena/project.yml`): only `project.yml` is tracked; everything else — including Serena's own `.serena/.gitignore` — is ignored. Added matching `permissions.deny` `Read()` rules for the three local paths. Documented the `<scope>: <description>` format in `tech/conventions.md`.
**Alternatives considered.** Edit `.serena/.gitignore` to add `/memories` and track it: rejected — Serena may regenerate that file and silently drop the override; the root `.gitignore` is agent-maintained, authoritative, and regeneration-proof.
**Consequences.** Future Serena state (memories, cache) never enters Git or agent reads. A regenerated `.serena/.gitignore` is harmless (already ignored). New commits use `<scope>: <description>` (scope vocab: `Tooling`, `Maintenance`, `Refactor`, `Docs`, or a module name). Reverse by dropping the two `.gitignore` lines and the three deny rules.
**References.** `.gitignore`, `.claude/settings.json`, `.claude/tech/conventions.md`, `.serena/project.yml`.

---

## 2026-06-08 — `compaction.sh` slimmed to single transcript-read mode

**Context.** `compaction.sh` carried a statusline branch (stdin-JSON parse + 60%/80% ANSI coloring) beside the manual transcript read. The live statusline is a separate script (`$HOME/.claude/statusline.sh`, per `$HOME/.claude/settings.json`), so that branch was dead code in both the repo copy and the canonical home mirror.
**Decision.** Adopted the user's slimmed `compaction.sh`: one path that reads the session transcript (via `CLAUDE_CODE_SESSION_ID`, else newest `*.jsonl`) and prints `N% used/window`; dropped the redundant `w>0` guard (the `case` always sets `w`). Verified byte-identical to the canonical mirror, `sh -n` clean, all three paths run (session-id, fallback, 1M window). Updated the INDEX.md helper row to match.
**Alternatives considered.** Keep dual-mode — the "repo-root `compaction.sh` re-added" entry below rejected stripping it ("divergence from canonical defeats the mirror"): superseded, because the canonical mirror was itself slimmed, so no divergence remains and the statusline is owned by `statusline.sh`.
**Consequences.** `compaction.sh` is a manual gauge only; any future statusline work belongs in `statusline.sh`. Keep the repo copy and `$HOME/.claude/compaction.sh` byte-identical.
**References.** `compaction.sh`, `.claude/INDEX.md` (Root-level agent helpers row); supersedes the option-(c) rejection in the 2026-06-08 "repo-root `compaction.sh` re-added" entry below.

---

## 2026-06-08 — `/session` → `/session-prompt` command rename

**Context.** The bootstrap slash command was named `/session`, overloading "session" — already used by `.claude/prompts/sessions.md` (the roadmap) and the multicam `Session` abstraction. The name also did not signal that the command emits the session-bootstrap *prompt*.
**Decision.** `git mv .claude/commands/session.md → session-prompt.md`, so the auto-discovered command (and skill) is now `/session-prompt`. Command body and frontmatter are unchanged. Repointed every live reference: `AGENTS.md`, `.claude/INDEX.md` (Layout tree + `commands/` and `prompts/` table rows), `.claude/prompts/sessions.md`. Left the prior `kickoff → /session` entry below untouched as historical record.
**Alternatives considered.** (a) Rewrite the older `kickoff → /session` entry to read `/session-prompt`: rejected — the log is append-only; this entry supersedes it for navigation, mirroring that entry's own reasoning. (b) Keep `/session` and add an alias: rejected — one canonical name avoids drift.
**Consequences.** Fresh sessions invoke `/session-prompt [TASK]`; the skill registers as `session-prompt`. The roadmap file remains `prompts/sessions.md` (plural) — no collision. `/session` mentions in the entry below are historical and resolve to `/session-prompt` going forward. Reverse by `git mv` back and re-pointing the same four references.
**References.** `.claude/commands/session-prompt.md`, `AGENTS.md`, `.claude/INDEX.md`, `.claude/prompts/sessions.md`.

---

## 2026-06-08 — kickoff prompt → `/session` slash command

**Context.** The reusable session-bootstrap prompt lived at `.claude/prompts/kickoff.md` and had to be pasted by hand with `<TASK>` manually substituted. Claude Code auto-discovers project slash commands from `.claude/commands/*.md`, so the same prompt can be a first-class command.
**Decision.** Moved the prompt into `.claude/commands/session.md` (frontmatter `description` + `argument-hint: [TASK]`), substituting `$ARGUMENTS` for `<TASK>`. `/session <TASK>` overrides the roadmap for the session; bare `/session` (empty `$ARGUMENTS`) routes to `.claude/prompts/sessions.md` and continues the **Current roadmap** from the next unfinished item, falling back to the Phase 4 maintenance cycle when all items are ✓. Deleted `kickoff.md`. Repointed the active references: `AGENTS.md`, `.claude/INDEX.md` (Layout tree + new `commands/` table), `.claude/prompts/sessions.md`.
**Alternatives considered.** (a) Keep `kickoff.md` and make the command a thin wrapper that re-reads it: rejected — two sources of one prompt invite drift. (b) Retroactively rewrite the `kickoff.md` mentions in the older entries below: rejected — the append-only log records historical state accurately; this entry supersedes them for navigation. (c) Encode the roadmap-vs-override branch as command logic: not possible — slash commands are static prompt text, so the branch is phrased as an instruction the agent evaluates against `$ARGUMENTS`.
**Consequences.** Fresh sessions start with `/session [TASK]` instead of a paste. The bootstrap prompt is now single-sourced in `session.md`; keep its steps in sync with CLAUDE.md / INDEX.md if those evolve. Older entries still cite `prompts/kickoff.md` paths — they are historical and resolve to `commands/session.md` going forward. Reverse by moving the body back to `prompts/kickoff.md` and dropping the frontmatter.
**References.** `.claude/commands/session.md`, `AGENTS.md`, `.claude/INDEX.md`, `.claude/prompts/sessions.md`.

---

## 2026-06-08 — Refactor pass: single One Euro impl, shared video_io.py, resolve_cli_sessions (partially supersedes 2026-05-24 "not worthwhile")

**Context.** Proactive refactor sweep. Reference scan found 6 definition-only constants; the documented-parallel `rtmlib_smoothing._OneEuro` proved algebraically identical to `smoothing.OneEuroFilter` (same formulas, different fp op order; missing only the scalar-confidence fast path / `_tau_d` cache / `__slots__`); and main.py/run.py had grown twin helpers AFTER the 2026-05-24 "main/run refactor not worthwhile" analysis — with real drift: `VIDEO_EXTS` had `.flv`, `VIDEO_EXTENSIONS` did not, and the session-resolution block was byte-identical 20 lines in both `_dispatch_sessions`.
**Decision.** (1) `OneEuroFilter` is the single One Euro implementation; `rtmlib_smoothing.py` imports it for `KeypointSmoother`'s per-region filters; `run.py` re-exports it for tests; the bare-default test site pins `min_cutoff=0.5` explicitly (class default is 1.0). (2) New `video_io.py`: `open_capture(source, display=None)` (unified WARNING wording), `safe_fps`, `frame_to_surface`, `collect_video_files` (returns Paths; run.py strs them), `VIDEO_EXTS` (.flv superset), FPS constants. (3) `multicam.resolve_cli_sessions()` owns --session-dir/--sessions-dir resolution + dispatch summary; main.py keeps its try/except-print, run.py keeps propagating. (4) Dead constants deleted (`EPSILON`, 4× `_COCO_*` range markers, `_HAND_MIDDLE_MCP`). (5) R: the 3×/2× `bcol`/`hcol` closures in clinical_features.R now delegate to the (previously unused) top-level `body_col`/`hand_col`. The 2026-05-24 decision still holds for the *processing loops* (semantically divergent; stay separate) — superseded only for the leaf helpers above.
**Alternatives considered.** Keeping `_OneEuro = OneEuroFilter` alias (rejected: preserves the two-implementations illusion); unifying multicam's `VIDEO_EXTENSIONS` tuple into video_io (rejected: ordered glob/error-message contract scoped to session-camera discovery); moving R `detect_tracking` into utils.R (rejected: couples two deliberately-standalone scripts for 5 trivial lines); merging `KeypointSmoother`/`PoseSmoother` (rejected: genuinely divergent track models).
**Consequences.** Smoothing changes apply to both backends in one place — tune `OneEuroFilter` once. Capture/FPS/batch fixes land in video_io only. main.py batch mode now also accepts `.flv`; open-failure messages are WARNING-style in both entry points. rtmlib path gains the scalar-confidence fast path (unused today). Reverse by re-vendoring per-entry-point copies (git history has them).
**References.** `src/pose_estimation/{video_io,rtmlib_smoothing,smoothing,multicam,main,run}.py`, `analysis/clinical_features.R`, `tests/test_{helpers,rtmw_confidence,rtmw_regions}.py`, `.claude/tech/{architecture,multicam,tests}.md`.

---

## 2026-06-08 — Single cv2 wheel via uv override-dependencies (maintenance)

**Context.** Maintenance audit found three cv2 wheels coinstalled: rtmlib declares `opencv-python` + `opencv-contrib-python` alongside our `opencv-python-headless`. All cv2 wheels unpack the same `cv2/` tree, so coinstallation file-stomps nondeterministically — runtime showed a mixed install (contrib's `ximgproc` present, headless GUI behaviour, install-order dependent).
**Decision.** `[tool.uv] override-dependencies` excludes rtmlib's two wheels with always-false markers (`sys_platform == 'never'`); cv2 ships exactly once via headless. Repair sequence after editing overrides: `uv lock && uv sync && uv sync --reinstall-package opencv-python-headless` (uninstalling the stompers removes shared files headless also owns).
**Alternatives considered.** Forking rtmlib (heavy for a metadata-only problem); switching our dep to contrib-headless to match rtmlib (still two wheels, ships unused modules).
**Consequences.** rtmlib must keep working against headless cv2 — verified: its installed source references no contrib-only modules, and the full suite (incl. charuco/aruco) is green. Any future dependency declaring another cv2 wheel needs the same override. Legacy-charuco note: `calibrateCameraCharuco*` is absent from contrib at 4.13 too — the 3B modern-API decision holds regardless of wheel.
**References.** `pyproject.toml` (`[tool.uv]`), `.claude/tech/environment.md` (single cv2 wheel policy), `.claude/tech/calibration.md` (API constraint).

---

## 2026-06-08 — ChArUco solver via modern OpenCV API in a separate charuco.py (Session 3B)

**Context.** Session 3B spec called for `cv2.aruco.calibrateCameraCharucoExtended`, but that function (and `calibrateCameraCharuco`) is contrib-only and ABSENT from opencv-python-headless ≥ 4.7 (we ship 4.13). The solver also needs cv2 + multicam, while `calibration.py` was deliberately cv2-free.
**Decision.**
1. **Modern API path**: `CharucoDetector.detectBoard` → `board.matchImagePoints` → `cv2.calibrateCamera` (intrinsics) and `cv2.stereoCalibrate(CALIB_FIX_INTRINSIC)` (pairwise extrinsics vs world cam). stereoCalibrate's (R,T) = camera-from-world directly because the world cam IS the world frame.
2. **Module split**: solver in new `charuco.py` (imports calibration/multicam/_types, acyclic); `calibration.py` stays cv2-free IO/validation.
3. **Direct-pair extrinsics only**: every camera must share board views with the world-frame camera; chains (A↔B↔C) unsupported and documented as a capture requirement.
4. **Global RMS**: per logical frame, anchor board pose via solvePnP in the world cam (fallback first detector), lift to world, reproject into all detecting cameras. No bundle-adjustment stage — synthetic accuracy (f < 2%, rot < 1°, trans < 15 mm @ 0.84 m baseline, RMS ≈ 0.4 px) did not justify it.
5. **Capture = frame-per-press**: pygame grid of live feeds; SPACE appends one frame per camera to per-camera MJPG AVIs, so frame index = press index and capture output feeds `solve` through the standard `discover_session` path with zero sync offsets.
**Alternatives considered.** (a) Depend on opencv-contrib for the legacy charuco calibrate — rejected: replaces the headless wheel, legacy API deprecated upstream. (b) solvePnP-per-frame + pose averaging for extrinsics — rejected: stereoCalibrate jointly optimises over all shared frames and yields residuals for free. (c) Free-running capture recording — rejected: frame-per-press makes synchronization inherent rather than post-hoc.
**Consequences.** Calibration sessions are ordinary `discover_session` directories (manifest + sync offsets work unchanged). Capture arrangement must give the world camera shared board views with every other camera. Board geometry constants live in `charuco.py`, not `calibration.py`.
**References.** `src/pose_estimation/charuco.py`, `src/pose_estimation/calibration_cli.py`, `tests/test_charuco.py`, `tests/test_calibration_cli.py`, `.claude/tech/calibration.md`.

---

## 2026-06-08 — world3d.csv schema + R 3D mode is an adapter, not a parallel path (Session 3C)

**Context.** Session 3C: export fused 3D to CSV and make `clinical_features.R` produce metric clinical features from it. The 2D feature path already used `dist_3d`/`angle_at_vertex` (xyz-capable) and window-speed = dist×fs.
**Decision.** (1) Single `world3d.csv` (no separate diag file): metadata `video,frame_idx,timestamp_sec,person_idx` + per-kp `_x_m,_y_m,_z_m,_confidence,_reproj_err_px,_n_views,_cheirality_ok`; kp names match the 2D schema; units m & px. Writer `export.write_world3d_csv` is duck-typed (unpacked args, no multicam import) to avoid the export↔multicam cycle; not in the package public API (matches `frame_to_rows`). `SessionFusion.frames` became `(frame_idx, timestamp_sec, world, diag)` — the exact writer row layout. (2) R side is an **adapter**: `is_world3d()` detects `_x_m` cols; `adapt_world3d()` gates each kp-frame to NA when `reproj_err_px > REPROJ_GATE_PX` (20, = fusion `max_view_reproj_px`) or `cheirality_ok==0`, drops diag cols, renames `_{xyz}_m→_{xyz}`; the existing feature path then runs unchanged in metric units. Only trunk metrics needed true 3D decomposition (`trunk_lean_angle_3d`, `trunk_lean_sagittal_3d`→new `trunk_lean_sagittal_deg`, `trunk_rotation_3d`, `posture_symmetry_3d`); lateral lean shares the 2D x–y formula. (3) Outputs get `_3d` suffix so downstream globs (`_clinical.csv`/`_clinical_windows.csv`) skip them — metre rows must not mix into normalised aggregations; 3D aggregation is out of scope.
**Alternatives considered.** Separate `world3d_diag.csv` (rejected — consumers must gate, so diagnostics belong inline with the row they qualify); a parallel R 3D feature path (rejected — duplicates ~1000 lines; the gate+rename adapter reuses everything); emitting projected 2D-style angles in 3D mode (rejected — discards the z signal that justifies fusion); no `_3d` suffix / shared filenames (rejected — silently corrupts unit semantics in every downstream aggregator).
**Consequences.** Any new per-kp 3D quality signal must be added to `make_world3d_header`, `write_world3d_csv`, and the `adapt_world3d` gate together. Vertical = −y assumes the `world_frame` camera is level. Building 3D-aware downstream aggregation later means widening those globs and deciding unit handling — deliberately deferred. `trunk_lean_sagittal_deg` is NA for all 2D inputs (out-of-plane is unmeasurable from one view).
**References.** `src/pose_estimation/{export,multicam}.py`, `analysis/clinical_features.R`, `tests/test_{multicam,r_pipeline}.py` (`TestWorld3DClinical`), `.claude/tech/{multicam,analysis,architecture,tests}.md`, `.claude/prompts/sessions.md` (3C).

---

## 2026-06-08 — 3D fusion is post-hoc CSV read-back; fuse_session_frame signature finalised

**Context.** Session 3A: implement the `fuse_session_frame` policy layer and wire it into `process_session()`. Per-camera keypoints are written to CSV by the `camera_processor` callback and never retained in memory, so the wiring needed a data source.
**Decision.** (1) Fusion reads the per-camera CSVs back (`export.read_csv_keypoints`) after the camera loop: normalised coords → pixels via *calibrated* resolution, logical frame = raw − `sync_offset`, `person_idx==0` only. (2) `fuse_session_frame(per_camera_keypoints, calibration, *, confidences, min_views=2, min_confidence=0.0, max_view_reproj_px=20.0) → (world, FusionDiagnostics)` — the stub's `SessionFrame` param was dropped (images unused; caller carries the frame index) and the return became a tuple per the roadmap spec. (3) Policy: validity mask → weighted DLT → greedy worst-view rejection (only while > min_views remain) → cheirality *flagging* (never dropping). (4) `process_session` keeps its `dict[camera → result]` return; fusion result (`SessionFusion`) is produced by public `fuse_session_outputs()` and currently only summarised to stdout — the world3d.csv writer (3C) will consume it inside `process_session`. (5) Fusion failures in `process_session` warn and continue: 2D CSVs are already on disk, and an exception would abort remaining `--sessions-dir` batches.
**Alternatives considered.** Returning keypoints from `camera_processor` (couples the callback contract of both backends, duplicates CSV content in memory); streaming lockstep multi-camera processing (major restructure, not needed for offline clinical use); raising on fusion failure (loses batch progress for a re-runnable step); reserved dict key for fusion in `process_session` results (collides with camera-name namespace).
**Consequences.** Fusion accuracy is bounded by what the CSVs hold: smoothed, normalised, 6-decimal-rounded 2D (~1e-3 px quantisation — negligible). Multi-person 3D requires cross-camera identity matching first. 3C should call `fuse_session_outputs` and write `world3d.csv` from `SessionFusion`; re-running fusion without reprocessing video is already supported. With exactly `min_views` views an outlier view cannot be dropped — downstream must gate on `reprojection_error_px`.
**References.** `src/pose_estimation/{triangulation,multicam,export,_types}.py`, `tests/test_{triangulation,multicam}.py`, `.claude/tech/multicam.md` (3D fusion section).

---

## 2026-06-08 — Read() deny-list in .claude/settings.json (CLAUDE.md directive)

**Context.** CLAUDE.md now mandates maintaining `permissions.deny` `Read()` rules so sessions can't waste context on low-value reads (one lock file ≈ 75K tokens).
**Decision.** Deny `Read()` on: env/library trees (`.venv`, `renv`), binary model dirs (`model`, `mediapipe`), patient data + pipeline outputs (`videos`, `output`, `benchmark_output`), lock files (`uv.lock`, `renv.lock`), `LICENSE`, `.git`, caches (`__pycache__`, `.pytest_cache`, `.ruff_cache`, `.Rproj.user`), rendered analysis artifacts (`analysis/*.html`, `*_files`, `*_cache`). Escape hatch for rare legitimate needs (e.g. checking a pinned version, sampling an output CSV, reading third-party source in `.venv`): Bash `rg`/`head`/`sed -n` — token-cheaper than Read anyway.
**Alternatives considered.** Denying `.claude/repomap.md` to force `rg`-only use: rejected — Grep/offset-Read of the map is part of the documented nav flow. Leaving `.venv` readable for third-party debugging: rejected — Bash sampling covers it with less token risk.
**Consequences.** Maintenance criterion when adding paths: deny anything large, binary, regenerable, or boilerplate whose full read provides little benefit; keep small, hand-edited sources readable. Update the list when gitignore gains a new data/artifact dir. A denied Read mid-task is a signal to switch to Bash sampling, never a hard blocker.
**References.** `.claude/settings.json`, `/CLAUDE.md` (200K-context bullet), `.claude/INDEX.md` (Layout).

**Context.** Working this repo cost excess tokens: no symbol index forced whole-file reads of 800–1271-line modules, and bootstrap pulled large memory/docs. User approved a four-lever fix: (1) a grep-able repo map, (2) prune memory/docs, (3) navigation-discipline docs, (4) split large files — accepting that splitting is behaviour-affecting.
**Decision.** (1) `scripts/repomap.py` generates `.claude/repomap.md` — `path:line: signature` lines (Python via stdlib `ast`, R via regex) over `git ls-files`, so `rg SYMBOL .claude/repomap.md` returns a jump target; drift-guarded by `tests/test_repomap.py` (`--check` subprocess, mirroring `test_public_api.py`). (2) Collapsed the three same-day compaction.sh entries here into one and added nav hints to `INDEX.md`/`kickoff.md`/`conventions.md`. (3) Documented the `rg → path:line → Read(offset)` workflow. (4) Split `run.py` 1271→782 by extracting two self-contained concern modules — `rtmlib_smoothing.py` (`_OneEuro`, `KeypointSmoother`, `REGION_PARAMS`, `_KP_*`) and `rtmlib_openvino.py` (`_patch_rtmlib_openvino`) — re-imported into `run.py`, leaving the public surface and console scripts unchanged.
**Alternatives considered.** (a) Also split `processing.py` (extract cropping): rejected — `get_hand_crop` shares `PALM_*_KP_IDX` with five other sites and tests, so extraction circular-imports (lessons 2026-06-04). (b) Split `clinical_features.R`: rejected — standalone, no cross-file dedup to gain; the map already navigates it. (c) Hand-maintained map: rejected — drift is this repo's top failure mode, so generator + test-guard is mandatory.
**Consequences.** Regenerate the map after adding/moving/renaming a symbol (`python scripts/repomap.py`); it indexes tracked files only, so `git add` a new module first. `rtmlib_smoothing`/`rtmlib_openvino` are the canonical homes for those symbols while `run.py` re-exports them (tests import `_OneEuro`/`REGION_PARAMS` via `pose_estimation.run`). Reverse a split by inlining the module back and dropping the import.
**References.** `scripts/repomap.py`, `.claude/repomap.md`, `tests/test_repomap.py`, `src/pose_estimation/{run,rtmlib_smoothing,rtmlib_openvino}.py`, `.claude/tech/architecture.md`, `.claude/tech/conventions.md` (Navigation).

---

## 2026-06-04 — Relocation repair: rewrite .venv absolute paths in place (offline) over uv sync

**Context.** The project was moved from `~/Documents/pro/pose-estimation` to `~/Projects/pose-estimation`. The move broke the venv's hardcoded absolute paths: the editable `.pth` (→ `ModuleNotFoundError` on `import pose_estimation`), all `bin/*` shebangs, and `activate*` `VIRTUAL_ENV`. The `output`/`videos` symlinks (relative), `.venv/bin/python` (→ system `/usr/bin/python3.13`), renv library symlinks (0 dangling), and all text config/source/docs survived the move untouched.
**Decision.** Repaired in place from inside the container with a binary-safe Python rewrite (old→new on `.venv` **text** files only — 27 files: bin scripts, activate variants, editable `_editable_impl_*.pth`, `direct_url.json`), skipping all `*.pyc`/`*.so` (path byte-length differs by 5; an in-place edit would corrupt them). Cleared regenerable caches carrying the old path (project `__pycache__`, `.ruff_cache`). Left `.venv` site-packages `.pyc` and renv `.so` (cosmetic embedded build/`co_filename` strings; recompiling gigabytes for a debug string is unjustified churn). Verified: editable import, console scripts, 252/252 pytest, renv loads `ragg`/`dplyr`/`ggplot2` at the new path.
**Alternatives considered.** (a) `uv sync` on the host (the canonical refresh): deferred to the user — CLAUDE.md reserves host commands for the user and the env note flags in-container sync as unreliable; the offline rewrite is deterministic, network-free, and the exact inverse of the move. (b) Recreate the venv from `uv.lock`: rejected — re-downloads multi-GB deps (torch/openvino/mediapipe) to fix pure path strings. (c) `sed -i` across every matching file: rejected — would corrupt `.so`/`.pyc` (length change), and the shell's `grep` is a profile function that hides dot-dir matches, making enumeration unreliable (use `find -exec grep`/Python).
**Consequences.** A later host `uv sync` reconciles cleanly (rewritten paths are already correct). If uv's editable mechanism changes (e.g. an `__editable__` finder replacing the plain `.pth`), the repair file list shifts but the principle holds (text-only rewrite, skip binaries). No code or test changes.
**References.** `.claude/tech/environment.md` (Relocation), `.claude/memory/lessons.md` (2026-06-04).

---

## 2026-06-01 — compaction.sh kept in-repo as a mirror of the global gauge (reverses the relocation)

**Context.** After the prior turn removed the repo-root `compaction.sh` (treating `$HOME/.claude/compaction.sh` as the sole canonical home), the user reversed course: re-added a repo-root `compaction.sh` and reworded CLAUDE.md back to "the supplied `compaction.sh`" (bare path ⇒ in-repo), retaining the 80% threshold. The re-added file is byte-identical to the canonical global script (`…/pro/agents/claude/compaction.sh`, symlinked at `$HOME/.claude/`): dual-mode (manual transcript read + statusline stdin-JSON), 80%/60% statusline coloring.
**Decision.** Incorporated the user's file as-is: `git add` the repo-root `compaction.sh` (verified identical to canonical, mode 100755, `sh -n` clean, runs: `30% 61K/200K`) and restored the "Root-level agent helpers" section in `INDEX.md`, corrected to 80% + dual-mode + a "byte-identical mirror of `$HOME/.claude/compaction.sh`, keep in sync" note. Authored nothing — the user-supplied content is the current canonical version, so the repo copy is a faithful mirror, not a divergent variant.
**Alternatives considered.** (a) Resurrect the older removed copy (parent of `899a10e`): rejected — it was manual-only with a stale 90% comment; the user's file is the newer 80%/dual-mode version. (b) Overwrite the user's file with my own authored version: rejected — it is already current canonical and verified; overwriting risks regressions and ignores what the user placed. (c) Strip the statusline half for a repo-only manual gauge: rejected — divergence from canonical defeats the mirror, and the manual branch already covers `sh compaction.sh`.
**Consequences.** Two copies of the gauge now exist (repo root + `$HOME/.claude/`) and must be kept in sync (noted in `INDEX.md`). The repo root again carries a shell script by design — retain it during keep-root-clean passes. No pipeline or test impact.
**References.** `/CLAUDE.md` (compaction directive: 80% + bare `compaction.sh`), `compaction.sh`, `.claude/INDEX.md` (Root-level agent helpers).

---

## 2026-06-01 — CLAUDE.md alignment audit + R-environment maintenance

**Context.** User revised CLAUDE.md (home-dir maintenance directive; "entries must add value beyond docs/code/Git — package versions are bloat"; dry-prose rule; subagents must use the largest model + TDD/multi-agent councils; Debian-Distrobox-on-openSUSE env detail with LSP/`bgcmd` tooling; agent-oriented-languages pointer) and asked for any downstream work.
**Decision.** Propagated each actionable directive to a single canonical home (no duplication, per the new value rule): `tech/environment.md` gained the two-layer host/container model + in-container LSP/`bgcmd` tooling and dropped drift-prone kernel/Python-version literals; `INDEX.md` authoring rules gained "each entry earns its place; omit version numbers (they live in lockfiles/commits)"; `tech/conventions.md` gained a "Working style (agents)" section (subagent-model rule, dry prose, red-green-refactor); pruned incidental package versions from the R-4.6 migration entry. Acted on the maintenance directive: removed the orphaned renv R-4.5 library tree (111 dangling symlinks), healed 5 dangling R-4.6 links via `renv::restore()`, reinstalled the documented R-graphics apt sysreqs a container rebuild had dropped (ragg/ggplot2 verified loading), cleaned the apt cache.
**Alternatives considered.** (a) Retroactively scrub every version string from the append-only logs: rejected — load-bearing versions (R 4.6 C-API break, dplyr 1.2+ recycle change) are the *point* of those entries; only incidental ones pruned. (b) Duplicate the subagent directive into `kickoff.md` for visibility: rejected — it already lives in CLAUDE.md, which every session reads first; duplicating violates the new value rule. (c) Delete the dangling R-4.6 links as cruft: rejected — they are recorded in `renv.lock`, so `renv::restore()` is the correct heal. (d) Author a persistent setup script for the R sysreqs: deferred — depends on the user's container-recreation cadence (offered as a follow-up).
**Consequences.** Agent docs carry the new directives at single canonical locations. The renv library is consistent with the lock (0 dangling). R-graphics sysreqs must be reinstalled after any container *recreation* (recorded as a lesson). No code or test changes; this is a docs + environment-hygiene change.
**References.** `/CLAUDE.md`, `.claude/tech/environment.md`, `.claude/tech/conventions.md`, `.claude/INDEX.md`, `.claude/memory/lessons.md` (2026-06-01 entry).

---

## 2026-05-24 — Temporal movement segmentation: velocity + aperture state machine

**Context.** Phase 2 clinical metrics (bilateral comparison, movement quality, trunk/torso) were complete. The final Phase 2 task is automated temporal segmentation to enable per-phase analysis ("is the reach phase getting smoother over sessions?") rather than whole-trial averages.
**Decision.** Implemented three functions in `clinical_features.R`: `running_median()` (sliding median filter), `classify_movement_phases()` (aperture-derivative state machine for REACH/GRASP/TRANSPORT/RELEASE classification), `segment_movements()` (speed-threshold movement detection via RLE + phase classification + per-phase feature extraction). Output: `*_movement_phases.csv` with 19 columns (per-phase: velocity, path, NJ, SAL, symmetry; per-movement: duration, n_phases, efficiency). State machine uses adaptive threshold (5% of aperture range) with min_phase_frames debounce (default 3). Falls back to REACH-only without hand data. Wired into main loop after window features.
**Alternatives considered.** (a) ML-based segmentation (HMM, LSTM): rejected — requires labelled training data, not interpretable for clinicians, and the rule-based approach matches the structured tasks (reach-grasp-transport-release) in clinical protocols. (b) Peak-detection event-based approach: rejected — less robust than the derivative-threshold state machine for noisy aperture signals. (c) Global speed-only segmentation (no phase sub-classification): rejected — clinical value comes from per-phase metrics (reach smoothness vs transport smoothness), not just movement detection. (d) Separate R script for segmentation: rejected — segmentation uses the same clinical features already computed, so co-locating avoids redundant file IO and ensures schema consistency.
**Consequences.** New output file `*_movement_phases.csv` added to each run. File exclusion filter updated to skip it during directory-mode processing. 251 total tests (2 new). Phase 2 is now complete (2A-2D all done). Phase 3 (3D pipeline) is the next major work item.
**References.** `analysis/clinical_features.R:608-860`, `tests/test_r_pipeline.py:108-230`, `.claude/tech/analysis.md` (segmentation section).

---

## 2026-05-24 — Adaptive min_cutoff: movement-phase-aware smoothing

**Context.** Session 1A added outlier rejection, lower hand min_cutoff, and multi-frame carry — but fixed min_cutoff still allowed 1-5px jitter during rest periods. The One Euro filter's beta mechanism handles fast movement well (cutoff rises with speed), but the min_cutoff floor is constant — during rest, small noise passes through unchanged.
**Decision.** Added per-keypoint adaptive min_cutoff that interpolates between `rest_cutoff` (at low velocity) and `min_cutoff` (at high velocity). Velocity tracked via EMA of per-keypoint speed (px/frame) from the filtered derivative dx_hat. Implemented directly inside OneEuroFilter.__call__ (smoothing.py) and _OneEuro.__call__ (run.py), gated by `rest_cutoff is not None and x.ndim == 2`. PoseSmoother and KeypointSmoother pass rest_cutoff from env vars to their filter constructors. Defaults: body rest_cutoff=0.05, hand rest_cutoff=0.15, rest_speed=2.0, fast_speed=10.0.
**Alternatives considered.** (a) Wrapper/mixin class around OneEuroFilter: rejected — adds indirection without benefit; the adaptation is 15 lines inside __call__ and disabled cleanly via rest_cutoff=None. (b) Kalman-style velocity regime estimator: rejected — EMA is simpler, has a single tunable (speed_alpha), and dx_hat is already a filtered derivative. (c) Global-speed adaptation (same cutoff for all keypoints): rejected — per-keypoint is strictly better (shoulder stays smooth during a wrist reach). (d) Opt-in (default disabled): rejected — adaptive mode with well-chosen defaults is strictly better; disable via env var POSE_BENCH_*_REST_CUTOFF=none.
**Consequences.** Enabled by default for both backends. 4 new env vars (POSE_BENCH_BODY_REST_CUTOFF, POSE_BENCH_HAND_REST_CUTOFF, POSE_BENCH_REST_SPEED, POSE_BENCH_FAST_SPEED). Backwards compatibility: setting rest_cutoff=none or rest_cutoff equal to min_cutoff disables the adaptation. 6 new tests in test_smoothing.py.
**References.** `smoothing.py:37-48,97-117`, `run.py:163-190,218-237`, `tests/test_smoothing.py:254-340`, `sweep_default.yaml`.

---

## 2026-05-24 — Roadmap: Stability + Clinical Metrics + 3D Pipeline

**Context.** Previous E2E roadmap complete. User reports persistent jitter/drops across all backends/modes and needs four categories of new clinical metrics (trunk/torso, movement quality, bilateral comparison, temporal segmentation). 3-cam footage ~2-4 weeks away.
**Decision.** 10-task roadmap in 4 phases:
- Phase 1 (Tracking stability): Investigate jitter + add adaptive smoothing. Highest priority — clean input is prerequisite for reliable clinical metrics.
- Phase 2 (Clinical metrics): Bilateral comparison -> movement quality scores -> trunk/torso metrics -> temporal segmentation. Ordered by complexity; segmentation blocked by bilateral + quality.
- Phase 3 (3D pipeline): fuse_session_frame() + solve_charuco() + 3D CSV/R. Implementable with synthetic data; validates against real footage when available.
- Phase 4 (Maintenance): Periodic dependency/security/tech-notes cycle.
**Alternatives considered.** (a) Wait for 3-cam footage before any new work: rejected — 2-4 week window is ideal for the substantial R-side clinical metrics work that's independent of 3D. (b) Start with temporal segmentation (highest clinical value): rejected — segmentation quality depends on having bilateral and quality metrics for per-phase characterization. (c) Unified Python+R clinical feature engine: rejected — R is the established clinical analysis language in this domain; maintaining the R pipeline is the right call.
**Consequences.** Phase 2 adds ~4 new output files (*_clinical.csv gains columns; new *_movement_phases.csv). Trunk metrics are body-mode-only. When 3D arrives (Phase 3), clinical_features.R gains a 3D analysis path (*_clinical_3d.csv). The segmentation state machine is rule-based (interpretable for clinicians) rather than ML-based.
**References.** `.claude/prompts/sessions.md` (session prompts), `.claude/memory/scratchpad.md` (plan rationale).

---

## 2026-05-24 — main.py/run.py refactor: analysis shows not worthwhile

**Context.** Roadmap task #7 proposed extracting shared patterns (CLI args, video loop, pygame setup, batch iteration, progress reporting) from main.py (827 lines) and run.py (1193 lines).
**Decision.** After line-by-line analysis, skip the refactor. Only ~15 lines are truly duplicated; they're embedded in semantically divergent loops with different return types (bool vs latency list), state management (MediaPipe internals vs rtmlib tracker), and CSV handling. Extracting shared code would add indirection without meaningful dedup.
**Alternatives considered.** (a) Extract a `VideoCaptureCommon` module for generic parts: rejected — the generic parts (progress string formatting) are too sparse to justify a new module. (b) Unify around a base class: rejected — different return types and initialization sequences prevent a clean interface.
**Consequences.** The two entry points remain self-contained. If drift becomes a problem, the progress-reporting string formatting (~15 lines) is the best extraction candidate.
**References.** `.claude/memory/scratchpad.md` (analysis notes).

---

## 2026-05-24 — Path traversal fix in session.json manifest parsing

**Context.** Security audit identified two path traversal vulnerabilities in multicam.py where `session.json` camera file references and calibration paths were resolved without validating they stay within the session directory.
**Decision.** Added `_safe_resolve(base, ref)` helper that resolves relative paths and rejects any result that escapes the base directory. Applied to both camera file resolution (line 240) and calibration path resolution (line 281).
**Alternatives considered.** (a) Restrict to filename-only (no slashes): rejected — legitimate use cases include subdirectory references like `raw/cam1.mp4`. (b) Symlink-aware check with `os.path.realpath`: rejected — `pathlib.Path.resolve()` already follows symlinks, and the string-prefix check is sufficient.
**Consequences.** Manifest files with `../` traversal references now raise `SessionError`. Two new tests guard the behavior.
**References.** `multicam.py:56-62`, `tests/test_multicam.py:166-193`.

---

## 2026-05-24 — R environment migrated to R 4.6.0

**Context.** R was upgraded from 4.5 to 4.6.0 on the host. The renv lockfile (targeting R 4.5) used package versions with C API calls (`Rf_findVar`, `Rf_allocSExp`) removed in R 4.6.
**Decision.** Install all packages at latest CRAN versions (R 4.6-compatible) via `install.packages()`, then `renv::snapshot()` to update the lockfile. Updated Matrix, renv, and all tidyverse packages to current CRAN releases. System deps installed: libfontconfig1-dev, libfreetype6-dev, libx11-dev, libharfbuzz-dev, libfribidi-dev, libpng-dev, libtiff-dev, libjpeg-dev, libwebp-dev.
**Alternatives considered.** (a) Pin R 4.5 in the project: rejected — R 4.6.0 is the system version and we should track it. (b) Use binary packages from Posit PPM: rejected — not available for Debian Trixie/openSUSE.
**Consequences.** renv.lock now targets R 4.6.0. Future `renv::restore()` on R 4.6+ hosts should work directly. R 4.5 compatibility is not guaranteed (package versions may use R 4.6 features).
**References.** `renv.lock`, `renv/activate.R`.

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
