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

- **A `.gitignore` row with no leading or middle slash matches at any depth** — the publisher and
  data trees (`inventory`, `sessions`, `qualification`, `measurements`, `calibration_qc`, `cohort`,
  `output`, `videos`) plus the directory rows `calibration/`, `model/`, `build/`, `dist/`. A fixture
  under one of those names (`tests/fixtures/calibration_qc/`, an inner `inputs/qualification/`) is
  silently uncommittable: `git add` reports success and commits nothing.
- **Verify every intended fixture path with `git check-ignore -v` BEFORE building the tree**, root
  and inner directories alike; the failure is invisible at every later step.
- Marker FILES are safe (`qualification.json` is no component named `qualification`; the `*.*/`
  siblings match directories alone). Layout that clears them: root
  `tests/fixtures/calibration_qc_set/`, upstream tree `inputs/upstream/`, golden
  `expected/published/`.

## Media fixtures (PyAV) — ordering rules

Every audio-bearing fixture in `tests/` is muxed by PyAV; these decide whether the file a test
writes is the file it meant.

- **Set `layout` (and every codec-context attribute) right after `add_stream`, before the first
  `mux`.** Muxing opens every codec in the container and an opened context ignores later writes,
  silently — a video-only `mux` earlier in the function freezes the audio layout at its default.
- **An audio frame carries no `pts`; the encoder assigns it** → supply `sample_rate`, `format`,
  `layout` on the frame. The video path is the opposite (`frame.pts` set explicitly,
  `tests/test_measure_cache.py`) → never write both from one template.
- **A declared stream with no packet never reaches the container**: `add_stream` alone yields a
  file whose `streams.audio` is empty, indistinguishable from a real no-audio asset → a "has audio"
  fixture muxes at least one encoded packet and flushes the encoder.
- **Rotated fixtures are buildable in-container**: PyAV cannot write a display matrix and `ffmpeg`
  is absent → stamp an ISO-14496-12 `tkhd` matrix into a PyAV clip, 48 bytes into the box, nine
  big-endian int32 (`{a,b,u,c,d,v,x,y,w}`, 16.16 fixed except `w` at 2.30); ffmpeg reads the angle
  from `a,b,c,d` alone. Reproduces the corpus behaviour at 0/90/180/270.

## Goldens + validators

- **A golden built only from well-formed input pins nothing about the failure path it exists
  for.** `2d_drop` (groups `((0,91),(1,3),(2,20))`: one healthy + one per short-input drop reason,
  the healthy trajectory truncated so the reason is the only difference) pins D05 on committed
  bytes: 3 groups = 1 windowed + 2 dropped (`too_few_frames`, `shorter_than_window`).
- **Golden-regeneration tests cannot prove an artifact's absence**: `regenerate()` copies a
  filename whitelist out of a staging dir it then deletes → assert absence by running the producer
  into a preserved dir and listing it, with a positive control proving the run happened.
- **Budget a published artifact by its enumerators, not its writer.** Seven places enumerate the
  R artifact set and the producer names none: `_expected_outputs` in
  `scripts/regenerate_r_clinical_goldens.py`, `_DATASETS` + `_BASE_WIDTHS` in
  `tests/test_r_clinical_goldens.py`, `_EXPECTED_GOLDENS` in `tests/test_r_timebase_truth.py`, the
  sorted `produced ==` list in `tests/test_r_pipeline.py::test_world3d_outputs_not_rescanned`, the
  `group_qc` name tuple in `tests/test_corpus_run_preconditions.py`, and — when a `src/` module
  moves with it — the campaign digests (`determinism.md`).
- **A deliverable-first seed leaves its suite wiring FAIL on purpose**: an `rc == 0` case is red
  until the artifact lands, and a skip guard breaks A32's zero-skip invariant → the test lands in
  the artifact's commit; its red witness is the pre-commit targeted run (→ `gates.md`).
- **Grade a new validator against a hand-made minimal POSITIVE before funding production behind
  it**: an absent-artifact check reaches error paths alone. A live minimal input must also
  exercise construction, including `shutil.copytree` destination handling.
- The generator imports the `tests/test_qualify` + `tests/test_sessions` builders (`_publish`,
  `_write_media`, `_uniform`, `_canonical`) rather than re-implementing them — they carry the PyAV
  ordering rules above.
