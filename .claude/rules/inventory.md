---
paths:
  - "src/pose_estimation/inventory.py"
  - "src/pose_estimation/video_io.py"
  - "scripts/run_inventory_mutations.py"
  - "scripts/check_inventory_determinism.py"
  - "tests/test_inventory*.py"
  - "docs/technical/inventory.md"
---

# Corpus registry (`pose-estimation-inventory`)

Family identity for the whole project. Schema owner = `docs/technical/inventory.md`.

- `capture_id` = `s{subject_ordinal:02d}-{task}-{side}` names a task-side **FAMILY** and carries no take component; a retake stays an asset inside that family with `repeat >= 1` and raises `view_conflict`. **A `view_conflict` family holds more than one physical take, so nothing may bind calibration or a session to it.** Full identity = the pair `(grammar_version, capture_id)` — a grammar migration can move membership while the readable key survives. Measured: 2 of 188 families conflict, and the corpus holds exactly one repeat-marked asset, inside one of them.
- `capture_id` is a low-entropy stable pseudonym: it supports linkage and does not resist enumeration. `asset_id` = blake2b-64 over the corpus-relative POSIX path (`surrogateescape`, so it is total over non-UTF-8 names); it is **not unique by construction**, and the guarantee is the explicit collision check that refuses to publish. A keyed HMAC stays declined: a lost key makes every downstream identifier unreproducible; clean-base determinism takes priority over opacity.
- `inventory/` is gitignored and patient-adjacent: `assets.csv` carries source paths, so `--out` is as sensitive as the corpus and the tool sets no file mode. `census.json` holds aggregates alone and is the only artifact whose numbers may be quoted.
- **Every consumer calls `inventory.validate_generation(out_dir)` before reading a row.** `generation` carries three digests — both CSVs and the census over its own remaining fields — so a half-published set, an edited CSV and an edited census all fail.

## Two committed gates back the registry's claims; rerun both before quoting any of them

- `scripts/run_inventory_mutations.py` mutates 72 predicates across `inventory.py` + `video_io.py` and demands a committed red per mutant — **71 killed, `M028` alone surviving as a ruled equivalent**, because both validation orders raise `InventoryError` and no contract pins precedence between two simultaneously corrupt tables. The catalogue self-validates — each patch must match exactly one occurrence and change bytes — so a stale mutant fails loudly instead of scoring a silent no-op kill.
- `scripts/check_inventory_determinism.py` runs 20 sweeps proving the three artifacts are a function of corpus bytes alone (hash seed, four locale settings, shuffled `iterdir`, path spelling, `--out` name, timezone, `umask`, `-O`), then puts 13 tamper classes through the consumer boundary and checks the **exception class**, so a leaked `OSError` reads as a failure rather than a pass.
- Both stream to `tests/inventory_*_results.json` per unit and refuse to append to a file measured against different source digests.
- **A new predicate earns a mutant in the same commit**, and a new test file must join the runner's `TEST_COMMAND` or the oracle cannot see it.
- **`check_inventory_determinism.py` binds to a spelled-out 4-entry `SOURCE_FILES`** — checker, `__init__.py`, `inventory.py`, `video_io.py` — records `source_sha256` per file, and exits 2 rather than overwrite a result measured against other bytes (`schema_version` 2). No head SHA: the regenerating run always precedes the commit carrying its output, so a recorded SHA names the parent state. Two cases in `tests/test_inventory_review.py` grade it — one recomputes every digest from the worktree, one grades `stale_source_mismatches` on a clean and a stale map. **Neither reaches `main`**, so the rc=2 wrapper and its `REFUSED` line are unpinned; the helper detects the mismatch, nothing proves the checker then refuses (→ `.agent/deferred.md`, checker firing seeds). **`run_inventory_mutations.py:817` still records `tested_head`**; that drop rides the `.agent/deferred.md` row, which already forces the regeneration a schema change needs.
