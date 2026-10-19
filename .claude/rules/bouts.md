---
paths:
  - "analysis/clinical_features.R"
  - "analysis/bout_qualification.R"
  - "scripts/qualify_bouts.py"
  - "src/pose_estimation/bout_qualification.py"
  - "tests/test_m4u5a_bouts.py"
---

# Wrist-motion bouts — the qualified detector (contract `.agent/archive/contract-m4u5a.md`)

- **The `BOUT_*` constants in `clinical_features.R` = the qualification's chosen cell** (A01:
  threshold 0.05, Q 0.95, min 0.3 s, merge 0.2 s, noise k 4, level bounds, extended SPARC
  support, prominence 0.1). Any change to detection, a constant, the generator or the scoring
  reruns the qualification before a corpus run consumes it, and the new cell lands as an
  amendment — a hand-tuned constant is an unqualified one.
- Regenerate (repo root, ~30 min CPU, Rscript on PATH):
  `scripts/gate.sh python scripts/qualify_bouts.py measure-noise --run output/corpus-2d --out N.json`
  then `scripts/gate.sh python scripts/qualify_bouts.py qualify --noise N.json --per-stratum 4 --out DIR`
  → `DIR/qualification.json` (`chosen`, `qualification` strata), `results.json` (every cell).
  Seeds live in `qualify_bouts.py`; the qualification bank is never used for selection.
- **The relative threshold needs its noise floor.** Without `BOUT_NOISE_K` a still hand's tracker
  noise becomes bouts (the probe's k = 0 cells started bouts ~1.4 s early on noise); with it every
  stationary synthetic series yields 0 bouts.
- **The golden reach moves for the whole series**, so the 4 `2d_*_clinical_bouts.csv` goldens are
  header-only and `2d_*_clinical_bout_qc.csv` carry `no_bout` / `no_usable_wrist` rows; bout
  values are pinned by `tests/test_m4u5a_bouts.py`, not by goldens.
- Firing (planted in `clinical_features.R`, reverted): NA-as-zero speed with no split → P03
  2 failed; SPARC over position → P01 1 failed; NJ over dropped slots → P03 1 failed;
  normaliser Q 1.0 → P04's own mutant assertion.
