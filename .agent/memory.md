# Memory

Sole memory store (CLAUDE.md). Carried-forward lessons + a condensed decision log.
Append durable items; prune what code, `.claude/tech/*.md`, or git already preserve. Full ADR rationale lives in git history.

## Lessons (append-only; positive-form rules — see AGENTS.md pink-elephant note)

Schema per entry: brief symptom → **Rule** (positive) → **Where** (files/tests that encode it).

**2026-06-23 — Merge an unrelated-history collaborator fork by *content*, never `git merge` — their history carries patient-data blobs forever.**
A parallel fork (separate `git init`, 2 commits, author `bernys`) tracked a patient `video.MP4` in its root; a real `git merge --allow-unrelated-histories` would embed that 8 MB blob in our objects permanently (later deletion can't purge history). **Rule:** reconcile such a fork as a content merge onto our `main` with a `Co-authored-by:` trailer for credit — pin the fork commit (minimal per-file diff of the fork's baseline tree vs our commits; here `20c36a0`), attribute each differing file as *ours* (0 diff at the fork base ⇒ keep ours, e.g. our post-fork probe work) vs *theirs* (>0 ⇒ 3-way merge), and pull in only source — never their tracked data/outputs or their own agent tooling. Clean their incorrect practices on the way in (CLI-arg paths over hardcoded ones, drop dead stubs/commented code). **Where:** this merge (`analysis/{data_extraction,arthrose_diag}.R` + `run.py` live-camera CSV export); `.gitignore` case-folded video-extension globs (so a stray root clip can't be committed).

**2026-06-16 — A shared test-helper module needs its dir on *both* pytest `pythonpath` and `ty.environment.root`.**
A sibling helper (`tests/synthetic_session.py`) importable at test runtime stayed invisible to `ty` (10 `unresolved-import`) — the two tools resolve modules independently. **Rule:** to share a test-helper module, add its dir to pytest `pythonpath` AND mirror it into `[tool.ty.environment].root` in the same change; verify with `uv run pytest` *and* `uv run ty check`; keep the two roots in sync. **Where:** `pyproject.toml`, `tests/synthetic_session.py`, `tests/conftest.py`.

**2026-06-15 — `np.errstate` cannot silence numpy's `warnings.warn`; all-NaN-column reductions fail under `filterwarnings=error`.**
`world3d.csv` carries a column per full-skeleton keypoint, so keypoints never fused (legs/face in arm mode) are all-NaN; `nanmedian`/`nanmean`/`nanpercentile` emit a `RuntimeWarning` via `warnings.warn`, which `np.errstate` (IEEE FP flags only) cannot mask. **Rule:** wrap `nan*` reductions in `with warnings.catch_warnings(): warnings.simplefilter("ignore", RuntimeWarning)` after confirming the all-NaN input is expected; reserve `np.errstate` for genuine FP-flag noise. **Where:** `validation.py` (`_temporal_jitter_mm`), `pyproject.toml` (`filterwarnings`).

**2026-06-15 — Shared components: fixture shape must match each backend's real output.**
`BoneLengthSmoother` is shared by MediaPipe (3D landmarks) and rtmlib (2D keypoints); a fixture that built rtmlib keypoints as `(133,3)` hid an `IndexError` on the real `(N,2)` output (constraints are on by default). **Rule:** build each backend's fixtures to its real runtime shape (rtmlib→2D, MediaPipe→3D) and prefer whole-row/vectorized ops (`landmarks[d]-landmarks[p]`) over hardcoded axis indices. 2D bone-length constraints are approximate under foreshortening; `--no-constraints` disables them. **Where:** `constraints.py` (`BoneLengthSmoother.update`), `tests/test_rtmw_constraints.py`, `tests/test_constraints.py`.

**2026-06-08 — `Read()` deny rules also block Bash commands naming those paths.**
The permission engine maps `Read()` denies onto any Bash command whose text references a denied path; the old "sample via Bash" escape hatch is gone. **Rule:** probe deny-listed trees with command text free of denied paths — functional checks (`uv run pytest --version` proves shebang health) and interpreter introspection (`uv run python -c "import pkg; ..."` walking from `pkg.__file__`). **Where:** `.claude/settings.json` (canonical deny statement).

**2026-06-08 — Synthetic ChArUco rendering: identity pose faces the camera; supersample the warp; diversify poses.**
Three failures building synthetic calibration tests: zero detections (boards <25 px/square), warp mush (plain `warpPerspective` aliases marker interiors), and a 16 mm stereo error (an Rx(π) "face camera" flip renders the mirrored back — OpenCV planar +z points INTO the board; a narrow pose cloud weakly constrains intrinsics, and the fx error couples into stereo translation). **Rule:** place boards for ≳25 px/square; render via `getPerspectiveTransform` + 3× supersampled `warpPerspective` then `INTER_AREA`; use identity-orientation-plus-tilts (identity already faces the camera; `texture_px = obj_m/square_size*px_per_square`, +y down). When accuracy misses tolerance *identically* across encode-quality runs, widen the pose cloud (translation AND tilt) rather than loosening tolerance. **Where:** `tests/test_charuco.py`, `.claude/tech/calibration.md`.

**2026-06-04 — When judging a split seam, audit every free name in the block, not a hand-picked list.**
A `processing.py` helper extraction failed late: `get_hand_crop` references `PALM_WRIST_KP_IDX`/`PALM_FINGER_KP_IDX` shared by five other sites → circular import. **Rule:** before extracting a block, enumerate *every* free name it references (walk the AST or `rg` each global) and confirm each moves with the block or stays importable one-directionally; any name shared with the source module's other sites is coupling — split only when the seam is acyclic. **Where:** `src/pose_estimation/processing.py` (`PALM_*_KP_IDX`).

**2026-06-04 — Repair venv absolute paths after a project move; never byte-edit `.pyc`/`.so`.**
A uv `.venv` hardcodes the project path in `bin/*` shebangs, `activate*`, and the editable `.pth`; moving the dir breaks imports + console scripts. **Rule:** after a move, rewrite old→new in `.venv` **text** files only (shebangs, `activate*`, editable `.pth`, `direct_url.json`) and clear `__pycache__`/`.ruff_cache`; skip `*.pyc`/`*.so` (in-place edit corrupts binaries). Enumerate with `find -exec grep` or Python — the shell's `grep` is a profile function that prunes dot-dirs, so `grep -r .venv` finds nothing. Verify with an import, a console script, `pytest`, `Rscript -e 'renv::project()'`. Canonical alternative: `uv sync` on the host. **Where:** `.claude/tech/environment.md` (Relocation).

**2026-06-01 — Reinstall R graphics sysreqs after container recreation.**
apt packages live in the container, not the project; recreating it drops them while project-local renv `.so` files survive but can't find runtime libs (e.g. `ragg` → `libwebpmux.so.3`). **Rule:** after a container rebuild, reinstall the R-graphics sysreqs, then verify with `Rscript -e 'library(ragg); library(ggplot2)'`: `sudo apt install -y libfontconfig1-dev libfreetype6-dev libx11-dev libharfbuzz-dev libfribidi-dev libpng-dev libtiff-dev libjpeg-dev libwebp-dev`. **Where:** `analysis/*.R` (graphics via `ragg`/`ggplot2`).

**2026-05-24 — Always validate resolved paths stay within the expected directory.**
`(base / user_ref).resolve()` in `multicam.py` accepted `../` traversal from session.json manifest fields. **Rule:** after resolving a relative path, check it stays within its base (`str(resolved).startswith(str(base_resolved) + "/")`) before use; *also* validate manifest *label* fields that become path components without passing through `_safe_resolve` (session_id → output dir, camera name → CSV filename — reject separators, `.`/`..`, control chars) regardless of whether a sibling `file` is set (2026-06-22 codex-review: the name check previously ran only when `file` was absent). **Where:** `multicam.py:_safe_resolve` + `_safe_name_component`, `tests/test_multicam.py:test_*_path_traversal_*` + `test_*_rejects_traversal_*`.

**2026-05-24 — R 4.6 broke the C API: always `renv::snapshot()` after upgrading R.**
A lockfile from R 4.5 failed to compile on R 4.6 (`Rf_findVar`, `Rf_allocSExp` removed). **Rule:** after upgrading R, install packages at latest CRAN versions first, then `renv::snapshot()`; use `renv::record("pkg@version")` to fix individual entries. **Where:** `renv.lock` (R version), `.claude/tech/environment.md`.

**2026-05-24 — R `if_else()` requires vector-length arguments in dplyr 1.2+.**
dplyr 1.2+ delegates to `vctrs` with strict length matching; a scalar condition with vector branches throws `vctrs::stop_recycle_incompatible_size`. **Rule:** use base R `if()`/`else` for scalar conditions inside `mutate()`; reserve `if_else()` for element-wise vectorized branching. **Where:** `analysis/longitudinal.R:65-72`.

**2026-05-16 — Treat hand-maintained tech docs as drifting unless guarded.**
An unguarded markdown doc (then `AGENTS.md`) silently rotted — listed a moved test, omitted new modules/scripts. **Rule:** extract tech notes from *code* at audit time, not from prior doc versions; prefer `path:line` references an agent detects when stale; when a commit adds a module, public-API export, or top-level script, also touch the matching `.claude/tech/*.md`. **Where:** `.claude/tech/architecture.md` (module map), `.claude/tech/tests.md` (inventory), `tests/test_public_api.py` (asserts package surface).

## Decision log (condensed; newest first — full rationale in git)

- 2026-06-25 — In-container CPU/GPU/NPU OpenVINO is the **primary** run path; host-side `.venv-host` only for host-OS launches. Verified: with the machine-local accel env sourced, both the `PYTHONPATH` accel runtime and the `.venv` pip wheel enumerate `['CPU','GPU','NPU']` and a generic compile+infer selftest passes per device — **device access is gated by that sourced env (driver farm: `LD_LIBRARY_PATH`/ICD/Level-Zero), not by which `openvino` imports.** `PYTHONPATH` order keeps the accel build in front of the pip wheel, so the pip dep stays harmlessly — `CLAUDE.local.md`'s "never pip install openvino" is a `sys.path`-precedence guard, not anti-shadowing. Project-model per-device coverage (NPU op support) = `scripts/npu_compat.py` + runtime NPU→CPU fallback, not a blanket guarantee. Machine-local specifics stay in git-ignored `CLAUDE.local.md`. **Where:** `environment.md` (Devices/inference), `.envrc`, `roadmap.md`.
- 2026-06-22 — Made the footage gate deny-safe + single-sourced the quality gates (the deferred genericization half): read-only `pose-estimation-run --list-sessions` probe (in-source `videos/` default keeps the command deny-literal-free, no frame decode, identifiers redacted to ordinal + camera count + calibration presence); detail in `roadmap.md` M2 Gate + `.claude/tech/entrypoints.md`. (`4777c52` `7f9b452`; codex-review redaction/hardening `8db22ba` `d96e317` `2b95f96` `2efec45`)
- 2026-06-22 — Split instruction files by audience × scope: `AGENTS.md` cross-agent contract (Claude loads it via top-of-file `@AGENTS.md` import, Codex natively), `CLAUDE.md` the Claude-Code delta, `~/.claude/CLAUDE.md` machine-global; all kept project-generic (state lives in `.agent/`). (`292e5d1`)
- 2026-06-22 — Adopted multi-mode session flow (MODE auto-detected from active-milestone status) + restructured roadmap into M<m>/M<m>.<u> ledger with commit trace-keys; flow specified in `/session-prompt`. (`c12ef18` `a9ba1c4`)
- 2026-06-19 — Consolidated agent state into `.agent/` (memory.md + roadmap.md + context.sh); retired `.claude/{INDEX,prompts,memory}/` and the repomap (Serena LSP supersedes it for Python; R navigated by grep). Tech notes kept under `.claude/tech/`.
- 2026-06-16 — Human-facing prose scope: only explicitly human-facing text avoids LLM-isms and uses hyphens over en/em-dashes; code/comments stay agent-optimized.
- 2026-06-16 — Failure-mode suite (1D) asserts the report field that genuinely moves under each fault, calibrated against outlier rejection.
- 2026-06-16 — `.serena/` tracked via its own committed `.gitignore` (commit as is).
- 2026-06-15 — Capture/QA gate (`qa_check` / `--qa-only`) + `docs/capture_protocol.md` + anonymization (de-identified derived artifacts only; no patient imagery).
- 2026-06-15 — Validation verdict (1B): graded PASS/WARN/FAIL on self-consistency surrogates, perf/symmetry informational; `THRESHOLDS` single-source in `validation.py`.
- 2026-06-15 — Validation harness (1A): orchestrate existing blocks (no reimplementation); baseline-optional agreement leg.
- 2026-06-15 — Default-deny `.gitignore` for calibration + credentials; history verified clean.
- 2026-06-08 — Refactor pass: single One Euro impl, shared `video_io.py`, `resolve_cli_sessions`.
- 2026-06-08 — Single cv2 wheel via uv `override-dependencies` (avoids opencv-python vs -contrib clash).
- 2026-06-08 — ChArUco solver uses the modern OpenCV API in a separate `charuco.py`.
- 2026-06-08 — `world3d.csv` schema fixed; R 3D mode is an adapter over the 2D path, not a parallel pipeline.
- 2026-06-08 — 3D fusion is post-hoc CSV read-back; `fuse_session_frame` signature finalized.
- 2026-06-08 — `Read()` deny-list in `.claude/settings.json` over data/cache/lock paths (CLAUDE.md directive).
- 2026-06-04 — Relocation repair: rewrite `.venv` absolute paths in place (offline) over `uv sync`.
- 2026-05-24 — Temporal movement segmentation: velocity + aperture state machine.
- 2026-05-24 — Adaptive `min_cutoff`: movement-phase-aware One Euro smoothing.
- 2026-05-24 — `main.py`/`run.py` left separate — analysis showed a merge not worthwhile.
- 2026-05-24 — Path-traversal fix in session.json manifest parsing (`_safe_resolve`).
- 2026-05-24 — R environment migrated to R 4.6.0; renv re-snapshotted at latest CRAN (apt sysreqs in `environment.md`).
- 2026-05-24 — `process_session()` callback-based orchestration pattern.
- 2026-05-24 — Jitter reduction: outlier rejection + increased hand smoothing + multi-frame detection carry.
- 2026-05-18 — Multi-camera (3-cam) scaffolding: `Session` + `Calibration` + `Triangulation`.
