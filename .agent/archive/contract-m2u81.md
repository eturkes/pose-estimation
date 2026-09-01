# Contract — M2.8.1, corpus-run preconditions + instrumented pilot

Tier `kernel`. Baseline `dac21d8`. Gate identity + invariant surfaces + predicates P01-P18 +
design decisions D01-D07 + negative-control seed §7. Frozen at write; amendments append to §8.

## 1. Why this unit exists

M2.8.2 funds a ~6.5 h resumable run over 379 real assets. The 2D path has only ever run over 12
preliminary clips from the retired `videos/initial/` tree. Every defect below is **silent** — it
moves a denominator or destroys an input with no error — so it must close before the run is funded.

## 2. Measured defect census — corrects the M2.8 plan on two counts

The plan named "two measured blockers" at three line numbers. Re-derived at implementation time:
**four defects, and both cited line-number sets are stale.**

| id | defect | site | measured consequence |
| -- | ------ | ---- | -------------------- |
| B1 | `--output-dir` never forwarded in native rtmlib session mode | `run.py:640-667` `_dispatch_sessions` → `process_session(s, camera_processor=…)` | `_resolve_session_output(session, None)` = `session.directory.parent/output/<event_id>` = **`sessions/output/<event_id>/`, inside the published tree**. `sessions.tree_digest` walks every entry but `generation.json`, so the first camera CSV moves the digest and every later `validate_generation` raises `SessionsError`. **The run poisons its own input tree.** Proven, not read: synthetic tree, digest before ≠ after. |
| B1b | diagnostics CSV is announced and never written | `process_source` (`run.py:370-380`) has **no `output_diag` parameter**; `_camera_processor` (`run.py:649`) accepts it and discards it; `process_session` (`multicam.py:517`) prints `Wrote diagnostics: {diag_path}` | A shipped surface states a file exists that nothing writes. Also removes the natural publication surface for B3. |
| B2 | group-level silent drops | `analysis/clinical_features.R:1310,1317,1323,1326,1328` | **5 sites, not 3**; plan's `1294,1300,1305` are stale by 16 and name three of five. Each drops a whole `(video, person_idx)` group with zero rows and zero record. |
| B2b | QC evidence is 3D-only | `clinical_features.R:1475` (`if (is_3d)` at emission) + `:1962` (`if (is_3d)` at write) | The 2D path M2.8.2 runs emits **zero** QC rows. A dropped subject leaves no trace anywhere in 2D. |
| B3 | CFR fallback rate uninstrumented | `video_io.py:53-79` `SourceTimestampClock.timestamp` | Two silent substitutions (`idx/fps`; `last + 1/fps`) with no counter. `pts_monotonic = 0` on 123/379 is an upper bound on exposure, never a measurement. |

Name note: the class is `SourceTimestampClock`. Roadmap and memory both call it `SourceClock`;
that spelling matches nothing in the tree.

## 3. Design decisions

**D01 — B1 closes by forwarding AND by a containment guard, not by forwarding alone.**
Forwarding fixes the explicit-flag path and leaves the default hazardous whenever `--sessions-dir`
names the published tree. `sessions.run` already carries the identical rule (`--out` must overlap
neither `--corpus` nor `--inventory` in either direction), so the same hazard class gets the same
guard: a resolved session output path lying inside the sessions tree **refuses loudly** rather than
corrupting a validated generation. Loud refusal over silent corruption is the publishers' standing
posture.

**D02 — `process_source` gains `output_diag` and `_camera_processor` forwards it.** A shipped
surface that prints a claim about a file nothing writes is a false claim, the same class M2.7.3
closed on drifted claim prose.

**D03 — the clock counts three dispositions, and the rate is derived, never stored.**
`pts_accepted`, `index_fallback` (PTS absent/non-finite/negative/non-increasing → `idx/fps`),
`monotonic_forced` (fallback still non-increasing → `last + 1/fps`). CFR fallback rate =
`(index_fallback + monotonic_forced) / total`. A stored rate and stored counts can disagree; counts
alone cannot.

**D04 — group dispositions publish to their own artifact in BOTH modes.**
Reusing `window_qc` is refused twice over: it is window-keyed, so a group dropped before any window
exists has no key, and it is 3D-only by an explicit guard whose stated reason is that a numeric
column on a 2D output enters `aggregate_per_video()` as a feature unnoticed. A separate file is
invisible to both consumer discovery paths, which select positively (`_clinical.csv` /
`_clinical_3d.csv` suffix; `_clinical_windows\.csv$` glob). New artifact:
`<stem>_clinical[_3d]_group_qc.csv`.

**D05 — the invariant is a partition, because that is what makes a denominator trustworthy.**
For every input `(video, person_idx)` group, **exactly one** of: it contributes ≥1 window row, or it
contributes exactly 1 group-qc row. Never both, never neither. A count is trustworthy only when
every group reaches exactly one explicit outcome — the same acceptance M2.1 set over 382 files.

**D06 — one reason code per drop site, derived from the sites rather than transcribed.**
`too_few_frames` (n < 4) · `invalid_cadence` (`fs` non-finite or ≤ 0) · `no_finite_timestamps`
(`t_start`/`t_end` non-finite) · `shorter_than_window` (span < `window_sec`) · `no_window_starts`
(empty `win_starts`). A sixth site (`sum(win_mask) < 4`, line 1333) drops a **window**, not a group,
and stays out of scope: the window artifact already reports emitted windows only.

**D07 — the pilot measures, it does not certify.** It is stratified over both codecs, all four
`(model, OS)` device configurations and all four rotation values, and it publishes redaction-safe
aggregates alone. Its product is the CFR fallback rate plus a green partition check on real assets;
it is not a claim about the full corpus.

## 4. Invariant surfaces

1. `src/pose_estimation/video_io.py` — `SourceTimestampClock` timestamp values stay
   **byte-identical**; only counters are added. The strictly-increasing postcondition is unchanged.
2. `src/pose_estimation/run.py` + `multicam.py` — session orchestration; no change to per-camera
   CSV schema or to `world3d.csv`.
3. `analysis/clinical_features.R` — the six committed 2D goldens (`2d_csv4dp_*`, `2d_cumsum_*`,
   `2d_idx_*`, each with a `_windows` companion) and `world3d_clinical_3d_window_qc.csv` stay
   byte-identical. Group-qc rows are a **new file**, never columns on an existing one.
4. Published window/QC row order and `qc_policy_tolerance` / `qc_coverage_tolerance` semantics.

## 5. Predicates

**Output routing (B1).**
- **P01** `_dispatch_sessions` forwards `args.output_dir` to `process_session`; a run with
  `--output-dir D` writes every per-camera CSV under `D/<session_id>/`.
- **P02** A resolved session output path inside the sessions tree raises loudly before any file is
  written. Negative control: default resolution against a published tree refuses.
- **P03** A published tree's `validate_generation` still passes after a session run that used
  `--output-dir` outside it — digest unmoved.
- **P04** The guard is containment-symmetric: output inside the tree and the tree inside output both
  refuse.

**Diagnostics (B1b).**
- **P05** `process_source` accepts `output_diag` and writes that file when given one.
- **P06** `_camera_processor` forwards `output_diag`; the path `process_session` prints exists after
  the call.
- **P07** With no `output_diag`, nothing is written and nothing is printed claiming it was.

**CFR instrumentation (B3).**
- **P08** The clock exposes `pts_accepted`, `index_fallback`, `monotonic_forced`; the three sum to
  the number of `timestamp()` calls.
- **P09** Counters attribute correctly: a capture returning valid increasing PTS scores
  `pts_accepted` only; one returning `-1` scores `index_fallback`; one returning a constant scores
  `index_fallback` (first) then whichever branch the postcondition forces, and the counts still sum.
- **P10** Timestamp values are unchanged against the pre-change implementation on every
  `tests/test_helpers.py` clock case.
- **P11** The fallback rate reaches the diagnostics CSV and a stdout summary line.

**Group dispositions (B2/B2b).**
- **P12** Each of the five drop sites emits exactly one group-qc row carrying its own reason code.
- **P13** Partition (D05): over any input, `#groups = #groups-with-windows + #group-qc-rows`, with
  the two sets disjoint. Checked in **both** 2D and 3D modes.
- **P14** The group-qc artifact is written in 2D mode, which today writes no QC file at all.
- **P15** Reason codes are re-derived from the source at check time, so a new drop site fails by
  name rather than being omitted.
- **P16** The six 2D goldens and the 3D window-qc golden are byte-identical after the change.

**Pilot (D07).**
- **P17** The pilot spans both codecs, all four device configurations and all four rotation values,
  and reports per-asset CFR fallback rate plus the P13 partition result. Aggregates only — no
  filename, path, subject token or per-row statistic.
- **P18** The pilot is a committed script, rerunnable from committed state.

## 6. Gate identity

Primary tree, every invocation prefixed `env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run
--no-sync`: `ruff check` · `ruff format --check` · `ty check` · `pytest`. Decisive suite green with
**0 skipped**; collection moves by exactly the new cases. Baseline at `dac21d8` = 1472 passed / 0
skipped. Never run the gate beside a decode or inference sweep — `test_r_timebase_truth.py::test_c8_08`
carries a subprocess timeout that CPU contention alone blows.

## 7. Negative-control seed

Each must fire and name its own predicate; the tree is restored byte-identical after each.

| id | seeded defect | expected |
| -- | ------------- | -------- |
| N1 | drop the `output_dir` forward in `_dispatch_sessions` | P01 |
| N2 | remove the containment guard | P02 |
| N3 | make the guard test one direction only | P04 |
| N4 | drop the `output_diag` forward | P06 |
| N5 | count `monotonic_forced` as `pts_accepted` | P08/P09 |
| N6 | perturb one timestamp branch | P10 |
| N7 | delete one drop site's group-qc emission | P12/P13 |
| N8 | gate the group-qc write on `is_3d` | P14 |
| N9 | add a sixth drop site with no reason code | P15 |
| N10 | reword a reason code in the checker only | P15 |

## 8. Amendments

*(none at freeze)*
