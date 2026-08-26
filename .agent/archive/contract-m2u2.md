# Contract — M2.2 session materialization + discovery

Tier `kernel`. MAIN authors `src/pose_estimation/sessions.py`, its CLI, and `docs/technical/sessions.md`. Every downstream artifact decides against this file.

## 1. Ruling: the instance grain

**A recording event is one performance of one task-side by one subject, captured by 1-3 hand-started cameras. Its key is `event_id`, and no `event_id` can ever equal a `capture_id`.**

```
event_id = f"{capture_id}_run-{run_index:02d}"      e.g. s02-cap-l_run-01
```

Evidence for the shape, not preference:

- Prior art (Pose2Sim, OpenCap, Anipose, EasyMocap, FreeMoCap, MMPose) puts the recording event below any participant or visit grouping and never treats a semantic family as proof that clips are one event → a family key must not be the event key.
- BIDS defines `run-<index>` as one uninterrupted acquisition, and requires `run-1`, `run-2`, … whenever acquisitions share every other entity. That is exactly "the subject performed this task-side again". Adopt the vocabulary rather than invent `-e<k>`.
- The project's standing constraint — calibration may never bind to `capture_id` — becomes structurally unrepresentable, because every event key carries `_run-` and every family key does not.

Every family gets a run index, singletons included, so a later-discovered repeat never collides with an unsuffixed key. `run_index` is **not chronological and carries no provenance claim**; BIDS does not make the index order chronological either. Assignment order is the registry's own `source_path` code-point order, which makes it deterministic.

Naming note: `session.json`'s `session_id` field carries the `event_id` string, because that is the consumer's field name and the schema does not change. Our own vocabulary says `event_id` everywhere else. `session_id` stays reserved for a future visit/day grouping that this corpus does not carry.

## 2. Ruling: take resolution for `view_conflict` families

**A run groups only assets we can assert were acquired during one performance. Where the registry proves more than one take and nothing identifies which view belongs to which, no multi-camera event is asserted.**

- 186 families have `n_assets == n_views` → one run, all views, `take_resolution = "family"`.
- 2 families carry `view_conflict = 1` (`s02-cap-l`, 3 assets; `s14-nut-r`, 4 assets) → each asset becomes its own single-camera run, `take_resolution = "unresolved"`. 2 families → 7 runs.

Evidence: no published pipeline infers same-take membership from filename, file order, duration, frame count, or creation-time proximity; take membership is declared at acquisition or by an operator, and temporal alignment is a separate later step from a decoded shared signal. MAIN measured the registry directly and the header facts separate neither conflict. Unequal frame counts are compatible with one event after offset estimation, so they are evidence of neither sameness nor difference.

Do not read a conflicted family's run count as a performance count. `take_resolution = "unresolved"` means the true grouping is unknown and the per-asset runs understate it. M2.3 may resolve these families by decode; that is a later ruling, not this unit's.

## 3. Ruling: partial-view policy

**Every canonical asset reaches the tree. Qualification is recorded, never enforced by omission.**

- One-view families (51) materialize as one-camera runs. `discover_session` accepts them, so nothing downstream needs a special case.
- Two-view families (85) and three-view families (52) materialize whole.
- The 3 quarantined assets are held out — quarantine is a stem-grammar verdict, and an asset with no parsed `(subject, task, side)` has no family to join.
- 0 excluded assets today; the code must still route an `excluded` disposition to hold-out.

Expected tree: **193 runs = 58 one-camera / 84 two-camera / 51 three-camera**; 379 placed + 3 held out = 382.

## 4. Deliverable surface

| artifact | role |
| -------- | ---- |
| `src/pose_estimation/sessions.py` | generator + `validate_generation` consumer boundary |
| `pose-estimation-sessions` console script | `= pose_estimation.sessions:main`, matching the six existing entry points |
| `docs/technical/sessions.md` | schema owner, mirroring `docs/technical/inventory.md` |
| `tests/test_sessions.py` | the unit's suite |
| `.gitignore` `sessions/` | the tree is patient-adjacent |

CLI: `pose-estimation-sessions --inventory inventory --out sessions [--strict]`. It reads the registry and **never walks the corpus**.

Emitted set, all under `--out`:

- `<event_id>/session.json` + one `<view>.<lowercase-ext>` symlink per camera.
- `events.csv` — `event_id, capture_id, subject_ordinal, task, side, run_index, take_resolution, n_cameras, views, view_conflict, grammar_version, generator_version`. Sorted by `event_id`.
- `placements.csv` — one row per discovered asset, all 382 — `asset_id, capture_id, disposition, placement, placement_reason, event_id, camera_name`. Sorted by `asset_id`.
- `generation.json` — digests of `events.csv`, `placements.csv`, a tree digest, and **the upstream `inventory/generation` digests**, so the tree is bound to the registry version that produced it.

`session.json` shape (extra keys are ignored by `_load_manifest`, so no schema change):

```json
{"format_version": 1, "session_id": "<event_id>", "capture_id": "…", "run_index": 1,
 "subject_ordinal": 2, "task": "cap", "side": "l", "take_resolution": "family",
 "n_cameras": 3, "grammar_version": "v1", "generator_version": "v1",
 "cameras": [{"name": "above", "sync_offset": 0, "asset_id": "a-…", "view": "above",
              "content_sha256": "…"}]}
```

No camera entry carries `file:`. `_safe_resolve()` resolves through a symlink and then rejects it, so an explicit `file:` ref fails on exactly the tree we emit.

## 5. Testable predicates

- **P01 no silent drop.** `placements.csv` has exactly one row per `assets.csv` row, joined by `asset_id`, 382 of 382. A placed row names an `event_id` + `camera_name` that exist in the tree; a held-out row leaves both empty.
- **P02 placement conservation.** `sum(placed) == symlink count in the tree == sum of `events.csv:n_cameras``.
- **P03 grain type safety.** No `event_id` equals any `capture_id`; every `event_id` matches `^s\d{2}-[a-z]+-[lr]_run-\d{2}$`; `capture_id` and `run_index` parse back out unambiguously.
- **P04 conflict policy.** Every asset of a `view_conflict` family sits in its own single-camera run with `take_resolution = "unresolved"`; no multi-camera run mixes them.
- **P05 discovery.** `multicam.discover_sessions(out)` returns exactly the `events.csv` row set, raising nothing; every `Session.session_id` equals its `event_id`; every camera resolves to an existing regular file.
- **P06 `--list-sessions`.** `pose-estimation-run --list-sessions --sessions-dir <out>` exits 0 and reports the same count.
- **P07 determinism.** Two runs from one registry produce byte-identical `events.csv`, `placements.csv`, `generation.json`, byte-identical manifests, and identical directory names, entry names and symlink targets — under a shuffled `iterdir`, four locales, a changed `PYTHONHASHSEED`, a changed timezone, and a different `--out` name.
- **P08 idempotency + stale removal.** A second run over a populated tree restores exact equality with a fresh oracle, and an entry injected between runs is gone afterwards.
- **P09 consumer boundary.** `sessions.validate_generation(out)` returns the generation block for a valid set and raises `SessionsError` for each tamper class: edited `events.csv`, edited `placements.csv`, edited `generation.json`, a removed session directory, a removed symlink, and a registry whose `inventory/generation` digests no longer match the ones recorded.
- **P10 registry gate.** The generator calls `inventory.validate_generation()` before reading a row and propagates `InventoryError`.
- **P11 path decode.** The `source_path` reverse decode is exact. **The live corpus contains zero escaped cells**, so the corpus provides no coverage for this and synthetic escaped paths are mandatory: `\\`, `\x0a`, `\xc2\x80` (U+0080) and `\x80` (a non-UTF-8 byte) must round-trip to distinct paths.
- **P12 symlink extension.** Every emitted symlink name ends in a lowercase member of `VIDEO_EXTENSIONS`, so `_find_glob_for_name` resolves it; 380 of 382 sources end `.MOV`.
- **P13 redaction.** No console line, no CSV cell outside `placements.csv`/`events.csv`, and no manifest value carries a corpus filename or a subject directory name. The tree's own names are `capture_id`-derived pseudonyms.
- **P14 publication.** The tree is built in a sibling temporary directory and swapped in by two renames, so `discover_sessions(out)` never observes a partially built child. Temporary directories are siblings of `--out`, never children.

## 6. Invariant surfaces

`src/pose_estimation/multicam.py` (unchanged by this unit), `inventory/` schema, `.gitignore`, `.claude/settings.json` deny list, `docs/technical/entrypoints.md`.

## 7. Gate identity

`uv run --no-sync ruff check && uv run --no-sync ruff format --check && uv run --no-sync ty check && uv run --no-sync pytest` in the primary tree, accelerator env sourced first. Baseline at dispatch: `1a26bdc`, 734 passed / 0 skipped.

## 8. Probe-corpus seed

Synthetic registries: empty; one canonical asset; a one-view family; a two-view family; a three-view family; a `view_conflict` family with 2 assets in one view; a family with 3 assets in one view; a quarantined-only registry; an `excluded` asset; escaped `source_path` cells (P11); a source extension that is not `.mov`; a registry whose `generation` block is tampered.

## 9. Amendments A01-A09 (from `map-m2u2`; these override §4-§5 where they differ)

- **A01 `--corpus` is required.** `assets.csv` stores corpus-**relative** paths and deliberately omits the root, so the CLI is `pose-estimation-sessions --inventory inventory --corpus videos/3-cam --out sessions [--strict]`.
- **A02 Symlink targets are relative**, written as the relative path from the session directory to the corpus file. An absolute target would bake this machine's checkout path into the tree, and the container and the host see the same checkout at different absolute paths.
- **A03 Camera name = `cam-<view>`, link name = `cam-<view><lowercase-ext>`.** The `cam` prefix keeps the tree discoverable through the glob path as well as the manifest path, and makes both paths yield identical camera names. `CAMERA_GLOB` is `cam*`.
- **A04 Listed-path validation, not rediscovery.** After decoding `source_path`, reject an absolute path, an empty / `.` / `..` component, a NUL, and a malformed escape; then require the resolved target to stay under `--corpus` and to be a regular file. The generator may `stat`/`open` only listed targets and must never call `iterdir`/`glob`/`rglob` on the corpus — pin that with instrumentation in the suite.
- **A05 Root ownership.** `generation.json` is the root marker. Refuse a non-empty `--out` that carries no valid marker, so the tool never adopts or deletes a directory it does not own.
- **A06 Grammar qualification stays a field, not a name component.** `grammar_version` rides `events.csv`, `placements.csv` and every manifest, and `generation.json` binds the tree to the upstream `inventory/generation` digests — which change on any grammar migration. Embedding the grammar in the directory name buys nothing over that binding.
- **A07 `--list-sessions` default root moves from `videos` to `sessions`.** The old default names a non-recursive raw-media root that never held sessions.
- **A08 Emit no calibration reference** — no manifest `calibration` key, no `calibration.json`. M2.6 binds calibration to `event_id`, and an unowned default file would bind it silently.
- **A09 `sync_offset: 0` is an unmeasured placeholder.** Materialization and listing make no synchronization, physical-take, calibration or 3D claim. M2.5 owns alignment.

Additional predicates from the amendments: **P15** no corpus directory listing occurs (instrumented); **P16** every symlink target is relative and resolves to a regular file under `--corpus`; **P17** a non-empty unowned `--out` is refused; **P18** the manifest-without-`file` path and the glob path both accept an out-of-tree symlink target, pinned as a regression so the `.agent/polish.md` symlink-consistency item cannot silently break this tree.

## 10. Deferred (polish, not this unit)

- Mutation campaign over `sessions.py` predicates, mirroring `scripts/run_inventory_mutations.py`.
- `sync_offset` beyond 0 — M2.5 owns it.
- Any use of decode evidence to resolve the 2 conflicted families — M2.3 owns it.
