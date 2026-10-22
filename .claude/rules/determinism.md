---
paths:
  - "scripts/check_*_determinism.py"
  - "tests/*_determinism_results.json"
  - "src/pose_estimation/__init__.py"
  - "src/pose_estimation/calibration_qc.py"
  - "src/pose_estimation/cohort.py"
  - "src/pose_estimation/corpus_run.py"
  - "src/pose_estimation/inventory.py"
  - "src/pose_estimation/multicam.py"
  - "src/pose_estimation/qualify.py"
  - "src/pose_estimation/sessions.py"
  - "src/pose_estimation/video_io.py"
  - "src/pose_estimation/view_agreement.py"
  - "src/pose_estimation/measure/**"
  - "scripts/probe_bias_transfer.py"
  - "scripts/probe_calibration_bias.py"
  - "scripts/probe_sync_policy.py"
  - "tests/test_cohort.py"
  - "tests/test_sessions.py"
  - "tests/test_measure.py"
  - "tests/test_qualify.py"
  - "tests/goldens/r_clinical/2d_idx_*"
---

# Determinism campaigns — regeneration order

- **A campaign refuses to regenerate over a stale result → `rm` its JSON first.** Each
  `scripts/check_*_determinism.py` compares recorded source digests against current bytes and
  exits rather than overwrite (the stale-green barrier). The calibration-QC campaign's field is
  `source_digests` → read the key before asserting on it.
- **Find every affected campaign before regenerating: `rg -ln '<file>' tests/*_results.json`.**
  One source can sit in several campaigns (`qualify.py` → qualify, calibration-QC + cohort);
  regenerate every dependent campaign, including comment-only source changes.
- **The cohort campaign holds one `source_digest` over its whole tuple**, so its failure names no
  file → read `scripts/check_cohort_determinism.py` `SOURCES` (`cohort.py`, `corpus_run.py`,
  `inventory.py`, `qualify.py`, `sessions.py`, `view_agreement.py`, `tests/test_cohort.py`,
  `tests/test_sessions.py`, the `2d_idx` frame / window / bout R goldens) before any gate whose
  diff touches one.
- Order: land every source + test edit → `rm tests/<name>_determinism_results.json` → rerun each
  affected campaign (`scripts/gate.sh python scripts/check_<name>_determinism.py`) → commit
  (regeneration = fixpoint, `evidence.md`).
- `calibration_qc` + its campaign bind `scripts/probe_bias_transfer.py` +
  `probe_calibration_bias.py` → reuse those probes by import (as `scripts/probe_crossview_arm.py`
  does), never by edit, unless republishing `calibration_qc/` is the intent.
