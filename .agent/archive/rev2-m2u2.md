# rev2-m2u2 — adversarial input / determinism / atomicity / redaction lens

Fill every `verdict` + `evidence` cell. `verdict` ∈ `clean` | `defect` | `claim-gap` | `n/a`. `evidence` = the command you ran and its result. Rewrite in place after each batch of 3.

| id | probe | verdict | evidence |
| -- | ----- | ------- | -------- |
| S01 | Determinism sweep on the live registry: shuffled `iterdir` order, four locales (`C`, `C.utf8`, `en_US.utf8`, `tr_TR.utf8` — Turkish dotless-i is the case-folding trap), changed `PYTHONHASHSEED`, changed `TZ`, and a different `--out` name. Compare CSV bytes, manifest bytes, directory names, entry names, link targets. | clean | `uv run --no-sync python .probe/batch1.py s01` imported this worktree's module and compared 4 runs: 4/4 snapshots equal across 193 directory names + 572 entries; CSV, generation, manifest, entry-name, and link-target bytes matched under all four locale/hash-seed/TZ/shuffle/output-name variants. |
| S02 | Two generators publishing into one `--out` concurrently. Staging is `<out>.staging.<pid>`, so name collision is avoided — what actually happens to the tree, and does either run's `finally` delete the other's work? | clean | After rebasing to `d5b29b6`, `uv run --no-sync python .probe/batch2.py s02` forced 3 schedules: initial-absent = 1 success + 1 handled `OSError`; existing = 2 successes; owner held after retirement while peer published = 1 success + 1 handled `OSError`. Every final tree validated against the live registry, discovered 193 events, and left 0 siblings. The new orphan sweep preserved every live-PID sibling; the failing owner's `finally` did not delete the peer's tree. |
| S03 | Interrupted publication: SIGKILL the generator during `_build`, during `out.rename(retiring)`, and between the two renames. Report what is on disk each time, whether `discover_sessions` sees a partial child, and whether a rerun recovers. | defect | Before `d5b29b6`, `uv run --no-sync python .probe/batch1.py s03` left dead-PID debris; MAIN accepted D03. Rebased rerun retained the allowed complete-or-missing active-root states, restored 193 events, and reduced both sibling counts to 0. Follow-on hostile sibling probe `pytest -q ...::test_run_sweeps_orphan_with_oversized_pid_suffix` failed: a 100-digit PID suffix raises uncaught `OverflowError` before publication. |
| S04 | Path-relation attacks: `--out` inside `--corpus`, `--out` == `--corpus`, `--out` inside `--inventory`, `--inventory` inside `--out`, `--out` a symlink to elsewhere. | defect | On `d5b29b6`, `uv run --no-sync python .probe/batch2.py s04` showed: nested output under corpus or inventory publishes; an owned `out == corpus` succeeds, deletes the source, validates internally, then discovery raises; an owned output enclosing inventory succeeds and deletes that registry; a symlink output becomes a real generated directory, leaves its target untouched, and leaves 1 retiring symlink. Unowned equality/enclosure alone refused safely. `uv run --no-sync pytest -q tests/test_sessions_adversarial.py` = 2 passed / 5 failed red path-safety cases. |
| S05 | Hostile listed targets: a registry row naming a symlink inside the corpus, a dangling symlink, a FIFO, a directory, a file removed between registry publication and generation. Each must hold out with the right reason, never crash and never place. | defect | Under superseding A14, `uv run --no-sync python .probe/batch3.py s05` placed and discovered an in-corpus symlink resolving to an in-corpus regular file; rejected an escaping symlink as `source_path_unsafe`; and rejected dangling symlink/FIFO/directory/pre-run removal as `source_missing`, always before publication. Defect: deterministic removal immediately after `resolve_source()` returned still published 1 event/1 `ok` placement; `validate_generation()` passed, the camera target was non-regular, and discovery raised `SessionError`. Red test failed as expected. |
| S06 | Malformed registry rows: duplicate `asset_id`, duplicate `(capture_id, view)` in a non-conflict family, an unrecognized `disposition` string, a missing column, an empty `capture_id`. | defect | `uv run --no-sync python .probe/batch3.py s06`: duplicate `asset_id` succeeded as a valid 2-row ledger with only 1 unique id/camera name; unrecognized disposition became `excluded_asset`; missing `source_path` escaped as `KeyError`; a canonical asset missing its capture row succeeded. Duplicate non-conflict view and empty `capture_id` correctly raised. Follow-on zero-row probe found missing assets/captures headers both publish because `_read_table()` discards fieldnames and per-row validation never runs. Six red cases cover the class. |
| S07 | `view_conflict` edge cases: set to `1` with a single asset, set to `0` with duplicate views, missing column, a non-`0/1` value. | defect | `uv run --no-sync python .probe/batch3.py s07`: `1` with one asset published an `unresolved` event; `2` silently normalized to `0`/`family`; missing column raised raw `KeyError`. Only duplicate views with `0` correctly raised `SessionsError`. Three targeted red tests failed. |
| S08 | Cell hostility: control characters, a leading `=`/`+`/`-`/`@` (spreadsheet formula injection), a quote, a newline, and a very long path — through `decode_source_path` into `events.csv`/`placements.csv` and into console output. | defect | `uv run --no-sync python .probe/batch4.py s08`: escaped control/quote/newline source path round-tripped, published, stayed absent from CSV/manifest/summary, and preserved raw link text; a 1,929-character path published validly; NUL failed safely. Hostile registry cells remained active: 4 formula-leading `task` cells in `events.csv`, 4 formula-leading `asset_id` cells in `placements.csv`, and 8 manifest values; quote/newline CSV row counts stayed correct. A malformed row with a newline + ANSI escape in `asset_id` returned 2 but echoed the raw id as 2 stderr lines with 1 escape control. Three red tests (including S09) failed. |
| S09 | Redaction sweep over one full live run: console stdout+stderr, every `session.json` value, every `events.csv` cell. Zero corpus filenames, zero subject directory names. Count and report violations. | defect | `uv run --no-sync python .probe/s09_refined.py` ran the live 382-asset tree: status 0, 4 stdout lines, 0 stderr, 0 source-path/filename console matches, 0 filename matches across 193 manifests + 193 event rows. Every manifest and every event row copied `subject_ordinal` equal to its associated source directory: 193 + 193 direct violations. Registry check: 379/379 canonical rows have `source_path` first component exactly equal to `subject_ordinal`. |
| S10 | Extension handling: `.MOV`, `.mov`, `.Mov`, `.mp4`, `.flv`, no suffix, two suffixes (`a.mov.txt`), a suffix that is only a dot. Which reach the tree, which hold out, and does `_find_glob_for_name` resolve every emitted name? | clean | `uv run --no-sync python .probe/batch4.py s10`: `.MOV`, `.mov`, `.Mov`, `.mp4` placed as 4 discoverable events with lowercase `.mov/.mp4` link suffixes and 4/4 `_find_glob_for_name` resolutions. `.flv`, no suffix, `.mov.txt`, and trailing dot held out as `extension_not_discoverable`. All 8 placements conserved; generation validated; discovery returned 4. |
| S11 | False-negative hunt on `validate_generation`: find any mutation of the published tree that it accepts. Try replacing a link target's text with an equivalent path, swapping two session directories' contents, adding an empty directory, adding a file to the root, and changing a manifest's key order. | defect | At `7af7ce7`, `uv run --no-sync python .probe/batch1.py s11` accepted 4/6 mutations: equivalent link text, empty directory, root file, and external event-directory symlink; it rejected directory swap + manifest key reorder, and both red tests failed. MAIN accepted D01/D02. Rebased at `d5b29b6`, the identical probe rejected 6/6 and both regressions passed. |
| S12 | Degenerate registries: empty, quarantined-only, excluded-only, one canonical asset, a 100-asset conflicted family (the new overflow guard), a family whose every asset holds out. | clean | On `d5b29b6`, `uv run --no-sync python .probe/s12.py` produced valid generations for empty (0/0), quarantined-only (0 events/1 `quarantined_stem`), excluded-only (0/1 `excluded_asset`), one canonical (1 event/1 placed), and all-undiscoverable-extension (0/3 `extension_not_discoverable`) registries; discovery counts matched 0/0/0/1/0. The 100-asset conflicted family raised `SessionsError` before creating `out`. |
| S13 | Idempotency at corpus scale: rerun over the populated live tree, compare against a fresh oracle byte for byte; inject a stale directory, a stale root file, and a stale link, then rerun and prove each is gone. | clean | `uv run --no-sync python .probe/batch2.py s13`: independent live generations matched across all 768 snapshot entries; a plain replacement rerun stayed byte-identical. After injecting one stale event directory, one root file, and one link, rerun removed 3/3, restored exact oracle equality, and passed two-argument validation against the live registry. |
| S14 | `pose-estimation-run --list-sessions --sessions-dir <out>` and `multicam.discover_sessions(<out>)` against the live tree and against each degenerate tree from S12. Exit codes, counts, exceptions, redaction. | defect | `uv run --no-sync python .probe/batch5.py`: live direct/CLI counts = 193/193, exit 0; one-canonical = 1/1, exit 0. Both outputs had 0 session/camera identifier violations. Valid empty, quarantined-only, excluded-only, and all-extension-held-out trees each returned direct count 0 but CLI exit 1 with no reported count. The rejected 100-conflict case had no tree; direct discovery raised and CLI exited 1 as expected. Red zero-generation test failed. |

## Defects (one section per accepted finding: severity, `file:line`, divergence, impact, acceptance check, red test path)

### D01 — raw symlink target text normalizes before hashing

- Severity: low.
- Surface: `src/pose_estimation/sessions.py:418`.
- Divergence: A12 requires the exact target text. `Path.readlink()` returns a normalized `Path`; adding a leading `./` changes `os.readlink()` bytes but leaves `tree_digest()` unchanged.
- Impact: `validate_generation()` certifies a tree whose symlink target text differs from the published generation, weakening its exact-artifact guarantee.
- Acceptance check: rewrite one target from `T` to `./T`; `validate_generation()` must raise `SessionsError` while both paths resolve equivalently.
- Red test: `tests/test_sessions_adversarial.py::test_validate_generation_binds_exact_symlink_target_text`.

### D02 — event-directory symlink escapes the validated root

- Severity: medium (scope interpretation uncertain; A12 names child directories, not symlinked directories).
- Surface: `src/pose_estimation/sessions.py:414`.
- Divergence: `Path.is_dir()` follows a top-level symlink, then the digest hashes only its external target's entries. Replacing an emitted event directory with a relative symlink to a byte-identical external copy validates.
- Impact: a validated event becomes externally located and mutable; `multicam.discover_sessions()` follows the same directory symlink after the boundary check.
- Acceptance check: replace one event directory with a symlink to a byte-identical out-of-tree directory; `validate_generation()` must raise `SessionsError`.
- Red test: `tests/test_sessions_adversarial.py::test_validate_generation_rejects_external_event_directory_symlink`.

### D03 — SIGKILL debris survives successful regeneration

- Severity: medium.
- Surface: `src/pose_estimation/sessions.py:520-533` at `7af7ce7`; fixed by `_sweep_orphans()` at current line 510.
- Divergence: PID-scoped names prevent concurrency collision, but pre-fix reruns clear only their own PID's names. SIGKILL bypasses `finally`, so dead staging/retiring trees persist indefinitely.
- Impact: patient-adjacent partial and complete tree copies accumulate outside the active root; the between-renames case leaves two copies until a later run.
- Acceptance check: rerun after each build/retire/between-renames SIGKILL; remove every dead-PID sibling, preserve every live-PID sibling, restore a valid active tree.
- Regression test: `tests/test_sessions_adversarial.py::test_run_sweeps_only_dead_publication_siblings` (passes at `d5b29b6`; pre-fix behavior is red).

### D04 — output path overlap can delete the corpus or registry

- Severity: high.
- Surface: `src/pose_estimation/sessions.py:565-567`.
- Divergence: `run()` rejects only resolved equality with `--inventory`; it accepts overlap with `--corpus`, ancestor/descendant overlap with either input, and a symlink output. Ownership-marker shape then licenses renaming and deleting an input tree.
- Impact: `out == corpus` deletes the listed source and returns an internally valid but undiscoverable tree; an owned output enclosing inventory deletes the registry. Nested outputs mutate inputs. A symlink output silently changes node identity and leaves a retiring symlink.
- Acceptance check: require pairwise-disjoint resolved input/output roots and reject a symlink output before ownership or publication; leave every input and symlink byte-for-byte intact.
- Red tests: `tests/test_sessions_adversarial.py::test_run_rejects_owned_output_equal_to_corpus`, `::test_run_rejects_registry_nested_under_owned_output`, `::test_run_rejects_output_nested_under_input`, `::test_run_rejects_symlink_output`.

### D05 — source can disappear after validation and publish as `ok`

- Severity: medium.
- Surface: `src/pose_estimation/sessions.py:282`, `:468-486`.
- Divergence: `resolve_source()` proves the listed target once during planning. `_build()` later emits the camera link without rechecking it, so a corpus change between those steps violates P16 at publication.
- Impact: `run()` returns success and `validate_generation()` certifies 1 placed camera whose target is absent; `discover_sessions()` then raises.
- Acceptance check: remove a listed regular file immediately after `resolve_source()` returns; publication must raise `SessionsError`, leave no new `out`, and clean staging.
- Red test: `tests/test_sessions_adversarial.py::test_run_rejects_source_removed_after_validation`.

### D06 — digest-valid malformed rows bypass the consumer boundary

- Severity: high.
- Surface: `src/pose_estimation/sessions.py:241-304`.
- Divergence: `_read_table()` validates no header, enum, uniqueness, or asset-to-capture relation. `plan()` overwrites duplicate ids through dicts, maps every unrecognized non-canonical disposition to `excluded_asset`, lets missing family rows fall back, and leaks `KeyError` for absent columns.
- Impact: a corrupt but internally hashed registry can publish duplicate placement identity and false exclusions, or crash outside the documented status-2 domain boundary.
- Acceptance check: reject duplicate `asset_id`, unrecognized disposition, missing required columns (including zero-row tables), and canonical assets without exactly one capture row as `SessionsError`; publish nothing.
- Red tests: `tests/test_sessions_adversarial.py::test_plan_rejects_duplicate_asset_ids`, `::test_plan_rejects_unrecognized_disposition`, `::test_plan_maps_missing_required_column_to_sessions_error`, `::test_plan_rejects_canonical_asset_without_capture_row`, `::test_run_rejects_missing_header_on_empty_table`.

### D07 — malformed `view_conflict` semantics silently normalize

- Severity: medium.
- Surface: `src/pose_estimation/sessions.py:257`.
- Divergence: only the literal `"1"` is true; every other present value becomes false, while a missing value leaks `KeyError`. A single-asset family carrying `1` is accepted despite the registry definition `view_conflict = n_assets != n_views`.
- Impact: corrupt capture metadata changes event grain and `take_resolution` while the generated tree validates as authoritative.
- Acceptance check: require the exact `0|1` domain and consistency with family members; map a missing field to `SessionsError`. Keep the existing duplicate-view/zero rejection.
- Red tests: `tests/test_sessions_adversarial.py::test_plan_rejects_single_asset_view_conflict`, `::test_plan_rejects_nonbinary_view_conflict`, `::test_plan_maps_missing_view_conflict_to_sessions_error`.

### D08 — hostile registry cells reach spreadsheet and terminal interpreters

- Severity: medium.
- Surface: `src/pose_estimation/sessions.py:288`, `:364-421`.
- Divergence: unconstrained registry strings are copied verbatim into CSV/JSON. The canonical-path error also interpolates raw `asset_id` into stderr. CSV quoting preserves rows but does not neutralize formula prefixes; terminal output preserves newlines and ANSI escapes.
- Impact: opening a generated CSV can evaluate a hostile formula; an error can forge log lines or inject terminal controls. Manifests preserve the same hostile values, though JSON does not execute them.
- Acceptance check: reject cells outside their registry grammar before rendering, and never echo an untrusted raw cell in console errors. Formula-leading ids/tasks and control-bearing ids must publish nothing and emit one control-free error line.
- Red tests: `tests/test_sessions_adversarial.py::test_plan_rejects_formula_prefixed_registry_cells`, `::test_plan_rejects_trailing_newline_cell`, `::test_main_redacts_hostile_registry_cell_from_error_console`.

### D09 — `subject_ordinal` reproduces the source directory identity

- Severity: high (patient-adjacent redaction boundary).
- Surface: `src/pose_estimation/sessions.py:364-419`.
- Divergence: P13 requires zero subject-directory names in manifests and `events.csv`, yet `subject_ordinal` is copied verbatim. For this corpus, every canonical source's first directory component equals that field.
- Impact: all 193 event artifacts expose the exact source-directory subject identity, preserving a direct linkage that the redaction predicate claims absent.
- Acceptance check: keep the required subject field only in a representation that cannot equal/reconstruct the raw directory label, or explicitly supersede P13. A source rooted at directory `D` must emit no manifest/event string equal to `D`.
- Red test: `tests/test_sessions_adversarial.py::test_generation_redacts_source_subject_directory_name`.

### D10 — list mode rejects a valid zero-event generation

- Severity: low.
- Surface: `src/pose_estimation/multicam.py:227`, called by `src/pose_estimation/run.py:692`.
- Divergence: A03 makes an empty registry and zero-event tree valid, and direct discovery returns `[]`. Shared processing dispatch converts that exact count into `SessionError`, so read-only `--list-sessions` cannot report the valid zero.
- Impact: empty, quarantined-only, excluded-only, and all-extension-held-out generations look operationally broken despite validating and preserving every placement.
- Acceptance check: list mode must exit 0 and print `Discovered sessions: 0 session(s)` for a valid zero-event generated tree; processing mode may retain its empty-input error.
- Red test: `tests/test_sessions_adversarial.py::test_list_sessions_reports_valid_empty_generation`.

### D11 — oversized orphan PID escapes the handled error domain

- Severity: low.
- Surface: `src/pose_estimation/sessions.py:523` at `d5b29b6` (pending implementation line 670 has the same catch set).
- Divergence: `_sweep_orphans()` parses an arbitrary sibling suffix with unbounded `int()` and passes it to `os.kill()`, but catches `ValueError` and `ProcessLookupError` only. A value outside C `long` raises `OverflowError`.
- Impact: one crafted sibling name prevents every generation and escapes `main()`'s documented status-2 domain until manually removed.
- Acceptance check: treat a numerically unrepresentable PID as dead/malformed debris, remove it, and continue publication; keep live-PID preservation unchanged.
- Red test: `tests/test_sessions_adversarial.py::test_run_sweeps_orphan_with_oversized_pid_suffix`.

## REV2-M2U2-DONE-1

## REV2-M2U2-DONE-2

| id | probe | verdict | evidence |
| -- | ----- | ------- | -------- |
| R08 | Publication atomicity under SIGKILL, concurrent publishers, failed ownership transfer, and hostile orphan names. | defect | `uv run --no-sync python .probe/batch1.py s03` killed build, retirement, and between-renames phases: active root was always complete or absent, never partial; rerun restored 193 events. `uv run --no-sync python .probe/batch2.py s02` forced initial-absent, dual-replacement, and retired-owner/peer-publish schedules: every survivor validated and discovered 193; no `finally` deleted the peer. Pre-fix dead-PID debris became D03; follow-on 100-digit PID raised `OverflowError` as D11. MAIN fixed both in `src/pose_estimation/sessions.py:655-758`: dead-only sweep, live/own-PID preservation, rollback only into an empty root, peer-owned root preservation, and `(ValueError, OverflowError, ProcessLookupError)` handling. |
| R09 | Registry cell alphabets: formula prefixes, ANSI/control bytes, and the regex terminal-newline edge. | defect | `uv run --no-sync python .probe/batch4.py s08` measured 4 formula-leading `events.csv` cells, 4 `placements.csv` cells, 8 manifest values, and a two-line/one-escape stderr injection. The pending guard still admitted a final newline because `Pattern.match()` + `$` stops before it; `test_plan_rejects_trailing_newline_cell` pinned that bypass. MAIN moved both alphabet call sites to `fullmatch`, changed exported `EVENT_ID_PATTERN` to `\Z`, and kept refusal messages value-free in `src/pose_estimation/sessions.py:123-141,331-347`. |
| R10 | Zero-row tables with short headers. | defect | Populated missing-column rows leaked `KeyError`; the phase-2 control re-digested empty tables after dropping `source_path` or `view_conflict`, and both runs exited 0 with a published empty tree. MAIN moved header validation into `_read_table(path, columns)` before row materialization; `src/pose_estimation/sessions.py:293-306,735-736` now raises `SessionsError` even when no row exists. Red cases: `test_run_rejects_missing_header_on_empty_table[assets|captures]`. |
| R11 | Non-ASCII numeric `subject_ordinal`. | defect | `"²".isdigit()` is true but `int("²")` raises uncaught `ValueError`; Arabic-Indic `"٢"` silently becomes integer 2. MAIN replaced `isdigit()` with ASCII `[0-9]+` under the shared full-cell validator in `src/pose_estimation/sessions.py:130-134,344-353`. Both representations were red before the fix and are pinned with the existing nonnumeric control. |
| R12 | Redaction and zero-session exit semantics. | claim-gap | Relation-aware live sweep found 0 corpus filenames in console, 193 manifests, or 193 event rows. It also measured `subject_ordinal == source directory` for 379/379 canonical assets. MAIN rejected D09 as a generator defect: that ordinal is the published low-entropy pseudonym already embedded in every `capture_id`, `event_id`, and event directory; the claim was narrowed in `docs/technical/sessions.md:168`. MAIN rejected D10: `--list-sessions` answers whether at least one session exists, so valid zero-event trees correctly exit 1 at `src/pose_estimation/multicam.py:227` / `run.py:692`. |
| R13 | Output-root overlap and symlink semantics. | defect | D04's two-way overlap failures were accepted: output equal to, inside, or enclosing either input could delete or pollute it. MAIN now resolves the publication target, checks ancestor/descendant overlap against corpus + registry before ownership, and publishes through a symlinked `--out` without replacing the link in `src/pose_estimation/sessions.py:628-652,703-758`. This supersedes review test `test_run_rejects_symlink_output`; the correct invariant is link identity preserved + target replaced atomically + zero debris. |
| R14 | Rebase delta for any later reuse of `wt/rev2-m2u2`. | n/a | Rebase from `d5b29b6` must absorb: `_read_table` header checks; `_validate_tables` alphabets/uniqueness/domain/family checks; derived `view_conflict`; `grammar_version` in placements; numeric manifest ordinal; link-time `resolve_source`; two-way disjoint roots; publication through real symlink target; `_remove` for link debris; failed-swap rollback; post-build orphan sweep; own-PID skip; filesystem-root `_is_within`; and the four phase-2 fixes above. Surfaces: `src/pose_estimation/sessions.py:81-141,258-758`, `src/pose_estimation/run.py:671-700`, `docs/technical/sessions.md:48-168`. MAIN reported 110 focused session cases passing. Final directive prohibited a local rebase or suite rerun, so no post-fix worktree gate is claimed. |
| R15 | Uncontested terminal probes: extension matrix, corpus-scale idempotency, live listing, and validation false negatives. | clean | S10 placed `.MOV/.mov/.Mov/.mp4` as lowercase discoverable links and held out `.flv`/no suffix/`.mov.txt`/trailing dot; 8 placements conserved. S13 matched two live generations across 768 entries, matched a plain rerun, removed directory/root-file/link injections 3/3, and passed registry-bound validation. S14 live/one-event direct + CLI counts were 193/193 and 1/1 with zero identifier output; zero-event exit 1 is R12's documented contract. Rebased S11 rejected all 6 mutation classes. Relevant boundaries: `src/pose_estimation/sessions.py:468-627,761-810`, `src/pose_estimation/multicam.py:173-246`, `src/pose_estimation/run.py:671-700`. |
