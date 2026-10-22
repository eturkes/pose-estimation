---
paths:
  - "src/pose_estimation/measure/*.py"
  - "scripts/run_m2u5_mutations.py"
  - "scripts/run_measure_mutations.py"
  - "scripts/check_m2u5_determinism.py"
  - "scripts/probe_sync_policy.py"
  - "scripts/probe_rigidity_saturation.py"
  - "scripts/probe_ransac_threshold.py"
  - "tests/test_measure*.py"
  - "tests/test_m2u5*.py"
---

# Measurement sidecar (`measure/`)

Publication contract → `publishers.md`. Alphabet + validation-predicate law lives there too.

- **`scripts/run_m2u5_mutations.py` mutates the primary tree's `src/` in place.** It restores under `try/finally` and verifies the restored digests, so an interrupted run is safe, but a *concurrent* reader sees mutated source. Never run it beside anything that reads or edits `src/`, and re-read a surprising source line after it exits. Same shape as `scripts/run_inventory_mutations.py`.
- **Published row order is contract-bearing, and two tables answer to different rules.** `_canonical(rows, key)` sorts `pairs_qc`/`cameras_qc`/`events_qc` at the publish site so their order is a function of the rows rather than of a loader's return order. `assets_qc.csv` is deliberately exempt: it publishes in registry order, which groups a capture's assets for a reader, while `asset_id` is a content hash whose ordering means nothing — `check_m2u5_determinism.py`'s D09 pins that instead. Permuting a loader's already-canonical return is fault injection past a validated postcondition, so the campaign leaves `load_assets` unwrapped by design.

## Rigidity — the gate constant and the population it selects

- **A gate constant must never double as an instrument parameter.** P21 accepted `drift_p95 ≤ 4 px` while the retired producer used that same constant as the MAGSAC inlier threshold, so no residual above the gate could ever be reported and 210/286 assets "failed" a gate judged against a quantity it pinned. `residual_p95` tracks the threshold monotonically over 8× while inliers grow 6% — it measures where the cut falls, not the scene. `measure/rigidity.py` now separates `RANSAC_THRESHOLD_PX = 8.0` from `DRIFT_P95_GATE_PX = 20.0`, pinned by `test_r2_gate_is_independent_of_ransac_threshold`.
- **A population count carries the instrument setting that produced it.** Eligible 298 is not a corpus property: holding the gate at 20 px and sweeping RANSAC 4→48 px moves eligibility 298→311, plateauing at 32 px (`scripts/probe_rigidity_saturation.py`). Every recovered asset lands in `camera_motion`, so rigid 280 and families 71/137 are threshold-invariant — a loosened instrument buys assets that fail the gate, never extrinsics.
- **`(device_config, view)` is the coarsest key naming a stable geometry**; the view token alone is not one. **iPad(5)/16.6 `above` = 89 assets, 23% of the corpus, an unstable camera** — `valid_fraction` median 0.212 against 1.000 in every other cell, `decode_status` ok on 89/89, highest quiet-border motion energy, and the landing site of 27 of the visual spike's 28 independent flags. `left`-vs-`right` handedness is unresolved by anything in this corpus.

## Sync — two closure statistics, three family rules

- **`events_qc.csv` groups by event** and accepts on R6's fused verdict (30 triangles, 5.403/30.286 ms). **`scripts/probe_sync_policy.py` groups by capture family** and accepts on audio alone (35 triangles, 4.451/30.286 ms — the P38 figure). Same corpus, same estimator, different populations. Always name the population beside the number.
- **"Families connected" names three different statistics, all over the same 137 multiview families.** `families_view_recoverable` = P38's, quantifying over *one camera per view*: a family counts when some one-asset-per-view selection is spanned by **cross-view** accepted pairs, which is what M2.6 consumes — **122** audio, 34 visual. `families_all_assets_connected` = every asset joined, same-view edges counted — **121** audio, 34 visual. `families_closure_gated` = the visual spike's: two views recover on one accepted cross-view pair, three views only when some one-asset-per-view triple has **all three** of its pairs accepted — **111** audio, **26** visual. All three ship in `probe_sync_policy.py`. Check which rule a number came from before opening an investigation.
- **Spanning three views takes two edges; closing them takes three, and that one edge is the whole 34 → 26.** The corpus splits 85 two-view + 52 three-view families. Measured over `visual_alone`: two-view 17 connected / 17 closure-gated, three-view 17 / **9** — and that 9 is the visual `triangles_closed` itself. The spike's rule was never a third connectivity definition, it was connectivity on two views and closure on three. `families_by_view_count` ships the split beside the totals, so the mechanism re-derives rather than being transcribed. A gap of one family between a port and its spike is a rule difference (the audio case, 122 vs 121); a gap of eight is this.
- **Two unrelated 26/137s exist.** `visual_alone` `families_closure_gated` = 26 and `agreement_required` `families_view_recoverable` = 26 measure different things on different pair sets. Quote the policy *and* the rule beside the count, never the count.
- Realized reference-time residual over sampled frames is median **6.31 ms**, p95 21.88, max 32.71 — inside one 33.3 ms frame. **Always separate the sync residual from a geometry verdict**, or a geometric null gets misread as an alignment defect and re-opens a closed unit.
- `drift_ppm`/`drift_se` carry no status column, so their emptiness (114/246 rows) never says which of `short_overlap`, `insufficient_windows`, `degenerate_regression` or `global_abstention` caused it. `measure.DRIFT_STATUSES` already names the four tokens. **Ruled won't-fix on the deferred row's own second arm**: `drift_ppm`, `status_drift` and `DRIFT_STATUSES` return zero hits across `src/pose_estimation/cohort.py` + `analysis/` against 7 inside `measure/`, so no consumer reads the reason, P36's column list stays frozen, and adding `status_drift` would re-measure the corpus for a field nobody reads. Reverses on a consumer that needs the abstention reason.
- `SUPPORTED_VERSIONS` is one module-wide frozenset, so a schema change on one axis invalidates every axis — over-invalidation is a granularity defect, not a correctness one (`.agent/archive/polish.md`). **Ruled not now**: the payoff arrives only when a sidecar axis schema next changes, and the conservative behaviour is sound. Revisit at the first proposal to move one axis's columns.
