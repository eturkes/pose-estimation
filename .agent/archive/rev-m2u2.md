# rev-m2u2 — correctness / spec / claim-soundness lens

Fill every `verdict` + `evidence` cell. `verdict` ∈ `clean` | `defect` | `claim-gap` | `n/a`. `evidence` = what you ran or the `file:line` that decides. Rewrite in place after each batch of 3.

| id | probe | verdict | evidence |
| -- | ----- | ------- | -------- |
| R01 | `EVENT_ID_PATTERN` accepts exactly what `plan` emits, and rejects every `capture_id`. Sweep the live registry's 188 families plus hostile synthetic ones (task token with a digit, uppercase, empty task). | defect | Live 188-family sweep: 193/193 emitted IDs match and 0 equal a `capture_id`. Synthetic `cap2`, `CAP`, and empty tasks make `plan` emit three IDs rejected by `EVENT_ID_PATTERN` (`src/pose_estimation/sessions.py:90,308`). Red→green: `tests/test_sessions_review.py::test_plan_rejects_event_ids_outside_published_grammar`. |
| R02 | The `raise` in `plan` for a non-conflict family holding two assets in one view: reachable? Correct? Does it match what `inventory` guarantees about `view_conflict`? | clean | `CaptureRecord.view_conflict` is exactly `len(assets) > len(distinct views)` and `capture_row` publishes it (`src/pose_estimation/inventory.py:259,783`). All 188 live rows matched that derivation. The raise is unreachable from producer output, reachable from direct/semantically forged tables, and correctly fails closed before duplicate camera/link names collide (`sessions.py:296-299` at reviewed base). |
| R03 | `grammar.get(capture_id, row["grammar_version"])` — when do `captures.csv` and `assets.csv` disagree on `grammar_version`, and what should win? | defect | Producer output cannot disagree: all 382 asset rows and 188 capture rows are `v1`. Direct `plan()` or an internally re-digested corrupt registry can disagree; reviewed code silently chose `captures.csv` (`sessions.py:249,319`). Neither should win: a mixed qualification is inconsistent and must raise. Red→green: `tests/test_sessions_review.py::test_plan_rejects_cross_table_grammar_mismatch`. |
| R04 | `resolve_source` containment: the `real_target != real_root and not startswith(real_root + sep)` test. Attack it — sibling directory with the root as a name prefix, symlinked root, `..` after realpath, corpus root itself as a target. | defect | Sibling-prefix escape and symlink escape → `source_path_unsafe`; symlinked root + inside file → accepted; lexical `..` → rejected; target resolving to the root directory → `source_missing`. Edge defect: corpus root `/` falsely rejects every contained file because `real_root + os.sep == "//"` (`sessions.py:211-225`). Red→green: `tests/test_sessions_review.py::test_resolve_source_accepts_file_below_filesystem_root`. |
| R05 | `decode_source_path` is claimed a **total inverse** of `inventory._printable_path`. Prove or break it by round-tripping a byte-exhaustive path corpus (every byte 0x01-0xFF in a name, plus surrogate-escaped names). | clean | `os.fsencode(decode_source_path(_printable_path(os.fsdecode(...))))` reproduced every byte 0x01-0xFF individually and in one mixed path; explicit backslash, U+0080, U+DC80/U+DCFF, newline, and DEL cases also passed. 0 failures (`src/pose_estimation/sessions.py:155`). |
| R06 | `_relative_parts` — any hostile relative path it admits that it should not, or vice versa. | clean | Accepted 6/6 legal POSIX-relative cases, including literal backslash, `C:` component, control byte, and nested path. Rejected 10/10 empty/absolute/doubled/trailing/`.`/`..`/NUL cases. The behavior matches contract A04 exactly (`src/pose_estimation/sessions.py:183-195`). |
| R07 | `run()`'s only path-relation guard is `out.resolve() == inventory_path.resolve()`. What happens when `--out` is *inside* `--inventory`, `--inventory` inside `--out`, or `--out` inside `--corpus`? | defect | All three relations were accepted (`sessions.py:495-506`). `out ⊂ inventory` and `out ⊂ corpus` published successfully. Worse, an owned `out` containing the input registry returned success, then the retiring-tree cleanup deleted that registry. Red→green: three overlap regressions in `tests/test_sessions_review.py`; worktree fix rejects ancestor or descendant overlap with either input root. |
| R08 | Publication failure atomicity: trace every failure point in `run()`'s `try/finally` and say exactly what survives. Pay attention to the case where `out.rename(retiring)` succeeds and `staging.rename(out)` then fails. | n/a | Explicit dispatch exclusion: `rev2-m2u2` owns atomicity. I ran no failure-injection probe and added no duplicate test. MAIN must merge that peer's verdict for this row's substantive result. |
| R09 | `validate_generation` versus contract P09: one probe per tamper class, including the two-argument registry-drift form. Name any class that passes when it should fail. | defect | Caught: edited events/placements digest, digest-field edit, non-object/invalid/missing marker, removed session directory, removed symlink, and two-argument registry drift. Incorrectly accepted four semantic marker edits: changed/removed `inventory` and changed/removed `generator_version` (`sessions.py:536-575`). Red→green: `test_validate_generation_rejects_semantic_marker_edits`; worktree adds a parsed-body self-digest, while key-order/whitespace reserialization remains valid. |
| R10 | `tree_digest`'s real scope versus what `docs/technical/sessions.md` claims it covers. Quote both. | claim-gap | Docs: “covers each entry name, each link target, and the bytes of each regular file” (`sessions.md:110`). Code: loops only `for directory in ... out_dir.iterdir() if p.is_dir()` then each immediate entry (`sessions.py:406,414-415`). Probe: a root file and an empty child directory leave the digest unchanged; a child entry changes it. A12 accepts this scope, but the docs omitted both exclusions. |
| R11 | Re-derive every number in `docs/technical/sessions.md` and in commit `bb780b6`'s body from the committed registry: 193/382/379/3, 58/84/51, 186/7, 380-of-382, 0.57 s. Report each as reproduced or not. | clean | Reproduced: 193 events / 382 assets / 379 placed / 3 held out; 58/84/51 by 1/2/3 cameras; 186 `family` / 7 `unresolved`; 380/382 uppercase `.MOV` (`docs/technical/sessions.md:60,92`). Fresh-process generation: 0.476-0.556 s, median 0.549 s across 5 runs, consistent with the 0.57 s commit measurement. |
| R12 | Cross-document consistency for the shipped surface: `entrypoints.md`, `architecture.md`, `multicam.md`, `README.md`, `pyproject.toml` scripts. Flag every statement the code does not support. | defect | Seven divergences: README + entrypoints say 7 scripts while `pyproject.toml` declares 8; inventory text sits under the sessions heading; placements omit required `grammar_version`; manifest `subject_ordinal` is string instead of numeric; data-boundary prose falsely assigns source paths to manifests/tables; corpus wording says content is opened/read; architecture's package API list omits exported `SessionFusion`. `multicam.md`, script mapping, and default `sessions/` behavior otherwise match code. Red→green schema tests + worktree docs repair all seven. |
| R13 | Project `CLAUDE.md` Authoring + Engineering conformance on every durable file the unit touched. Human-facing docs go to the ASD-STE100 register; everything else agent-optimized. Comment budget: does each comment buy a `why` a fresh agent would otherwise re-derive? | defect | `sessions.md` has 0 `LONG`/`FILLER` failures (3 advisory findings); `.agent`/pyproject changes conform. Durable defects: false source-path/content claims, unsynced staging read exclusion, and a 12-line `run.py` comment that duplicates docs/roadmap provenance. Worktree repairs those. Full touched-doc scanner still reports 40 findings / 13 failing `LONG|FILLER`: README 1/1, architecture 11/2, entrypoints 8/3, multicam 17/7, sessions 3/0; these remaining failures are pre-existing portions of touched files, not M2.2's inserted prose. |
| R14 | `sessions` exports nothing through `pose_estimation/__init__.py`. Is that right, given `test_public_api.py` pins the surface and `multicam`'s `Session` is exported? Argue the boundary. | clean | Correct boundary. `multicam.Session` is a runtime pipeline domain type; `sessions.Event`/`Placement` and generator functions form an explicit tooling-module API. Inventory follows the same convention. Docs import `validate_generation` from `pose_estimation.sessions` (`sessions.md:119`), while README defines the stable package root via `__init__.py` (`README.md:239`). Top-level re-export would lock a second session vocabulary without a consumer need. |

## Defects (one section per accepted finding: severity, `file:line`, divergence, impact, acceptance check, red test path)

### D01 — `plan` can publish an event ID outside its own grammar

- Severity: medium.
- Surface: `src/pose_estimation/sessions.py:90,308`.
- Divergence: public `plan()` interpolates unvalidated `capture_id` input, despite P03 and the module claim that every event key has the advertised grammar.
- Impact: direct `plan()` callers can receive invalid event keys. `run()` is protected by upstream inventory validation, so the committed registry is unaffected.
- Acceptance: reject every constructed `event_id` that fails `EVENT_ID_PATTERN`; keep the live 193-event sweep green.
- Red test: `tests/test_sessions_review.py::test_plan_rejects_event_ids_outside_published_grammar` (3 failures before the worktree-only guard; 3 passes after it).

### D02 — mixed grammar versions are silently resolved

- Severity: medium.
- Surface: `src/pose_estimation/sessions.py:249,319` at reviewed base.
- Divergence: public `plan()` lets the capture row override disagreeing asset rows instead of rejecting an internally inconsistent registry qualification.
- Impact: direct callers or a semantically forged but digest-consistent registry can publish a misleading grammar version. Producer-generated registries cannot reach the branch today.
- Acceptance: require one capture grammar version and the same value on every member asset before constructing an event.
- Red test: `tests/test_sessions_review.py::test_plan_rejects_cross_table_grammar_mismatch` (failed before the worktree-only consistency guard; passes after it).

### D03 — filesystem-root corpus fails containment

- Severity: low.
- Surface: `src/pose_estimation/sessions.py:211-225` at reviewed base.
- Divergence: the prefix test forms `//` when `corpus_root == "/"`, so a regular file below the root is classified outside it.
- Impact: a valid absolute corpus root cannot be `/`; default and live roots are unaffected.
- Acceptance: use component-aware containment that treats `/` as every absolute path's ancestor while preserving sibling-prefix and symlink-escape rejection.
- Red test: `tests/test_sessions_review.py::test_resolve_source_accepts_file_below_filesystem_root` (failed before the worktree-only `commonpath` guard; passes after it).

### D04 — overlapping roots can delete an input registry

- Severity: high.
- Surface: `src/pose_estimation/sessions.py:495-506` at reviewed base.
- Divergence: `run()` rejects only equal output and registry paths. It accepts ancestor/descendant overlap with both input roots.
- Impact: when the input registry is inside an owned output, replacement returns success and deletes that registry with the retiring tree. Output inside the corpus also mutates the source boundary; an output ancestor could retire corpus data.
- Acceptance: after `inventory.validate_generation` and before ownership/publication, reject any ancestor-or-descendant overlap between `out` and either input root; preserve all pre-existing bytes.
- Red tests: `test_run_rejects_output_inside_inventory`, `test_run_rejects_output_inside_corpus`, and `test_run_preserves_inventory_nested_under_output` in `tests/test_sessions_review.py` (3 failures before the worktree-only guard; 3 passes after it).

### D05 — semantic `generation.json` tampering validates

- Severity: high.
- Surface: `src/pose_estimation/sessions.py:536-575` at reviewed base.
- Divergence: A11 says every semantic edit raises, but one-argument validation ignores `inventory` and `generator_version` entirely.
- Impact: a consumer accepts altered or absent provenance and generator identity while reporting the generation valid. Passing `inventory_dir` catches only the altered upstream block, not an altered generator identity.
- Acceptance: bind the parsed marker body to a deterministic digest while excluding only its own digest field; reject changed/removed body fields and accept key-order or whitespace-only reserialization.
- Red tests: `test_validate_generation_rejects_semantic_marker_edits` (4 failures before the worktree-only parsed-body digest; 4 passes after it) and `test_validate_generation_accepts_marker_reserialization`.

### D06 — placements omit the required grammar qualification

- Severity: medium.
- Surface: `src/pose_estimation/sessions.py:80-88,142-152,327-341` at reviewed base; `docs/technical/sessions.md:108`.
- Divergence: contract amendment A06 requires `grammar_version` in `placements.csv`, but the column, dataclass field, population, and schema prose omit it.
- Impact: held-out and placed asset outcomes cannot be qualified or joined by grammar version at the placement grain without reopening the upstream registry.
- Acceptance: copy each asset row's `grammar_version` into every placement and document the column.
- Red test: `tests/test_sessions_review.py::test_placements_publish_asset_grammar_version` (failed before the worktree-only schema field; passes after it).

### D07 — manifest subject ordinal has the wrong JSON type

- Severity: medium.
- Surface: `src/pose_estimation/sessions.py:128,313,345-379` at reviewed base; `docs/technical/sessions.md:72`.
- Divergence: the contract's manifest schema uses numeric `subject_ordinal`, while `Event` and the emitted JSON use a string.
- Impact: schema consumers see a type different from the authoritative example; CSV text remains unaffected.
- Acceptance: parse the registry cell into `int` at the event boundary, emit JSON number, and render CSV text explicitly.
- Red test: `tests/test_sessions_review.py::test_manifest_publishes_numeric_subject_ordinal` (failed before the worktree-only type correction; passes after it).

### D08 — tree-digest documentation overstates coverage

- Severity: low; judgment-only claim gap.
- Surface: `docs/technical/sessions.md:110`; `src/pose_estimation/sessions.py:406,414-415` at reviewed base.
- Divergence: prose says every entry name is covered; accepted A12 behavior excludes all root files and cannot represent an empty child-directory name.
- Impact: an operator can infer tamper coverage that validation intentionally does not provide.
- Acceptance: state the child-directory/immediate-entry scope and both exclusions; retain the accepted implementation.

### D09 — shipped surface documentation drifted in five places

- Severity: low; judgment-only except D06-D07.
- Surface: `README.md:42`; `docs/technical/entrypoints.md:3,97-110`; `docs/technical/sessions.md:3,72,108,130`; `.gitignore:53-54`; `docs/technical/architecture.md:34-38`.
- Divergence: 7-vs-8 script counts; inventory prose under the sessions heading; false source-path claims; corpus-read wording; incomplete `SessionFusion` API list. D06-D07 are the schema-bearing members.
- Impact: operators receive contradictory defaults and sensitivity explanations; agents inherit a stale public-surface map.
- Acceptance: align counts/headings/schema/types/data-boundary language with code and add `Read(sessions.*/)` beside the matching ignore rule. Worktree Markdown passed a Marksman positive control with no diagnostics published for the four edited files.

### D10 — touched human-doc set still fails the declared text register

- Severity: low; deterministic authoring-conformance finding.
- Surface: pre-existing portions of `README.md`, `docs/technical/architecture.md`, `docs/technical/entrypoints.md`, and `docs/technical/multicam.md`.
- Divergence: `.scratch/steq.py` reports 40 findings, including 13 failing `LONG`/`FILLER` sentences. Per file findings/failures: 1/1, 11/2, 8/3, 17/7; `sessions.md` is 3/0.
- Impact: the unit touched files whose full human-facing surface does not meet the project-wide ASD-STE100 claim. M2.2's inserted prose adds no failing sentence, so this is inherited debt rather than a product behavior defect.
- Acceptance: the committed text-register gate proposed in `.agent/polish.md` reaches 0 `LONG`/`FILLER` over the declared inventory. Keep this outside the M2.2 spine unless MAIN expands scope.

## REV-M2U2-DONE-1

## REV-M2U2-DONE-2

Persisted by MAIN from the reviewer's messages: the agent saturated at 99% 237K/240K before it wrote this section itself.

| id | probe | verdict | evidence |
| -- | ----- | ------- | -------- |
| R08 | Residual atomicity after MAIN's first rollback fix: is the ordinary second-rename failure the only losing schedule? | defect | The ordinary case was fixed — injected failure gave 3 renames including the restore, and the old tree stayed valid. The crash state still lost data. Setup: a dead-pid complete `staging` plus a valid dead-pid `retiring`, with `out` absent (the SIGKILL-between-renames shape). The new run built its own staging, `_sweep_orphans` deleted both dead generations, then the injected `staging.rename(out)` failure ran; the except block saw an empty `out` and called `retiring.rename(out)` on a path this run never created, so `FileNotFoundError` replaced the injected `OSError`. End state: `out` absent, dead staging gone, dead retiring gone — no recoverable generation. Fix directed: move the orphan sweep after a successful `staging.rename(out)` and guard the rollback with `retiring.exists()`. |
| R08 | Fault-injection re-grade on the final bytes. | clean | Initial `staging→out` failure preserves the original error, leaves `out` absent, debris 0. `out→retiring` failure preserves the valid prior `out`, debris 0. A rollback-rename double failure leaves the prior generation valid under `retiring` with `out` absent, the rollback error surfacing while the swap stays the exception context. A post-swap sweep failure leaves the new `out` and the prior `retiring` both valid. No additional data-loss defect. |
| R13 | Durable evidence pointers in attached state and commit bodies. | defect | `.agent/roadmap.md:58` pointed at `.scratch/agents/contract-m2u2.md` and `.agent/polish.md:37` cited `Contract §10`, both inside a gitignored, cleanup-bound tree; commit bodies `7af7ce7` (`.scratch/probe_m2u2_guards.py`, `.scratch/agents/rulings-m2u2.md`) and `d5b29b6` (`.scratch/probe_m2u2_digest.py`) name evidence a clean checkout cannot follow. `bb780b6` and `7af7ce7` also record a gate still in flight. Directed: archive the contract and rulings, repoint both files, and record the decisive committed-state gate in the close commit. MAIN did all three; a follow-up audit then caught `.agent/archive/rulings-m2u2.md:3` still naming the scratch contract, which MAIN also repointed. |
| R13 | Project `CLAUDE.md` register conformance, corrected scope. | claim-gap | `docs/technical/conventions.md:29-39` puts `docs/technical/**` in the internal agent register, so the 52-finding / 18-failure ASD scan is not a conformance gate there and MAIN's earlier ruling stands. Human-register signal across the touched set is README alone: 1 inherited `LONG` at line 268, with M2.2's own README edits clean. Technical-doc findings survive only where they are accuracy or redundancy defects, never sentence-length ones. |
| R13 | Final pre-close diagnostics across every durable file the unit touched. | clean | Marksman and Serena returned 0 warnings for `README.md`, all four touched files under `docs/technical/`, `.agent/roadmap.md`, `.agent/memory.md`, `.agent/polish.md`, and both archived M2.2 files. An archive-wide `.scratch/` search returned 0 matches after the pointer correction, with durable archive references present as the positive control. |
