---
paths:
  - "tests/**/*.py"
  - "tests/fixtures/**"
  - "scripts/make_calibration_qc_fixtures.py"
  - "scripts/regenerate_r_clinical_goldens.py"
  - ".gitignore"
---

# Committed fixtures, goldens + media

## The gitignore trap that eats a whole deliverable

**Four `.gitignore` rows are slash-free component names and therefore match at ANY depth**: `inventory:60`, `sessions:78`, `qualification:85`, `calibration_qc:94`. A committed test fixture placed under one of those names — `tests/fixtures/calibration_qc/`, or an inner `inputs/qualification/` — is silently uncommittable, and `git add` reports success while committing nothing.

- **Verify every intended fixture path with `git check-ignore -v` BEFORE building the tree**, root and inner directories alike; the failure is invisible at every later step.
- Marker FILES are safe by construction: `qualification.json` is not a component named `qualification`, and the `*.*/` siblings carry a trailing slash so they match directories alone.
- Layout that clears all four: root `tests/fixtures/calibration_qc_set/`, upstream tree `inputs/upstream/`, golden `expected/published/`.

## Media fixtures (PyAV) — three ordering rules

Every audio-bearing fixture in `tests/` is muxed by PyAV, and these decide whether the file a test writes is the file it meant.

- **Set `layout` (and every other codec-context attribute) immediately after `add_stream`, before the first `mux` call.** Muxing opens every codec in the container, and an opened codec context ignores a later attribute write — silently. A video-only `mux` earlier in the same function is enough to freeze the audio stream's layout at its default.
- **An audio frame carries no `pts`; the encoder assigns it.** Supply `sample_rate`, `format` and `layout` on the frame and let the encoder do the timing. The video path is the opposite (`frame.pts` set explicitly, `tests/test_measure_cache.py`), so the two must not be written from one template.
- **A declared stream with no packet never reaches the container.** `add_stream` alone produces a file whose header the probe reads and whose `streams.audio` is empty — indistinguishable from a real no-audio asset. A fixture meaning "has audio" must mux at least one encoded packet and flush the encoder.
- **Synthetic rotated fixtures are buildable in-container.** PyAV cannot write a display matrix and `ffmpeg` is absent, so stamp an ISO-14496-12 `tkhd` matrix into a PyAV-muxed clip: the matrix sits 48 bytes into the `tkhd` box as nine big-endian int32 (`{a,b,u,c,d,v,x,y,w}`, 16.16 fixed except `w` at 2.30), and ffmpeg reads the angle from `a,b,c,d` alone, so the translation terms do not matter. Verified to reproduce the corpus behaviour exactly at 0/90/180/270.

## Goldens

- **A golden built only from well-formed input pins nothing about the failure path it exists for.** The group-disposition goldens were header-only on three healthy 2D datasets and on `world3d`, so `2d_drop` was added with groups `((0,91),(1,3),(2,20))` — one healthy plus one per short-input drop reason, truncating the healthy trajectory so the drop reason is the only difference. Its golden carries 2 rows over `too_few_frames` + `shorter_than_window`, and the three `2d_drop` goldens together pin D05 on committed bytes: 3 groups = 1 windowed + 2 dropped, disjoint. Existing goldens stayed byte-identical because healthy datasets keep the default single group.
- **Golden-regeneration tests cannot prove an artifact's absence.** `regenerate()` copies a filename whitelist out of a staging directory it then deletes, so an unexpected output never reaches the golden directory. Assert absence by running the producer into a preserved directory and listing it, with a positive control proving the run happened.
- **Budget a published artifact by its enumerators, not by its writer.** **Seven places enumerate the artifact set and the producer names none of them**: `_expected_outputs` in `scripts/regenerate_r_clinical_goldens.py`, `_DATASETS` + `_BASE_WIDTHS` in `tests/test_r_clinical_goldens.py`, `_EXPECTED_GOLDENS` in `tests/test_r_timebase_truth.py`, the sorted `produced ==` list in `tests/test_r_pipeline.py::test_world3d_outputs_not_rescanned`, the `group_qc` name tuple in `tests/test_corpus_run_preconditions.py:858`, and — whenever a `src/` module moves with it — the `source_digests` in `tests/qualify_determinism_results.json` + `tests/calibration_qc_determinism_results.json`. Wiring one new artifact moved all of them.
- **A deliverable-first seed leaves its suite wiring FAIL on purpose.** A case asserting `rc == 0` would be red until the artifact lands, and a skip guard breaks the zero-skip invariant the full gate's A32 reconciliation enforces (`tests/a32_gate.py`). The test lands in the same commit as the artifact it grades; seeding a passing stub instead would encode the defect. Its red witness is therefore the pre-commit targeted run on the unfixed tree, never a red commit (→ `gates.md`).
- **A validator graded only against an absent artifact is graded on its error paths.** M2.7.2's checker called `shutil.copytree(inputs, dest)` onto a live `TemporaryDirectory` — missing `dirs_exist_ok=True` — and its three replay predicates were already FAIL for missing fixtures, so the `FileExistsError` stayed invisible until a teammate's first live replay. **Grade a new validator against a hand-made minimal POSITIVE before funding production behind it**; an all-`unknown` seed proves only that the failure path prints.
- The generator imports the `tests/test_qualify` + `tests/test_sessions` builders (`_publish`, `_write_media`, `_uniform`, `_canonical`) rather than re-implementing registry, session and media construction — those builders carry the PyAV ordering rules above, which a second copy would have to re-learn.
