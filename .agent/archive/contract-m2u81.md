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

**A01 — D01's containment guard is scoped to PUBLISHED trees, not to every session tree.**
As frozen, D01 read as blanket containment against `session.directory.parent`. That refuses the
documented default for every ad-hoc session directory, because the default *is*
`session.directory.parent/output/<session_id>`. The invariant the guard actually protects is
narrower and sharper: **no file is created inside a generation whose digest a consumer checks.** So
the guard triggers only when a generation marker sits at or above the session directory. Ad-hoc
trees keep the documented default; published trees refuse. Implemented as `_published_root`.

**A02 — P04's symmetry holds, and the ancestor-base case correctly ALLOWs.**
Passing `--output-dir <parent-of-tree>` resolves to `<parent>/<session_id>`, a **sibling** of the
published tree rather than a parent of it, so nothing is written inside the generation and the guard
permits it. Direction 2 (`published in target.parents`) remains implemented and reachable — it fires
when the tree sits under a directory named for the session id. A probe asserting REFUSE on an
ancestor base is asserting the wrong outcome.

**A03 — a sixth reason code, `no_windows_emitted`, and D06 is corrected.**
D06 put the per-window `sum(win_mask) < 4` floor out of scope as a window-level rule, which is
right, but it leaves **P13 ill-defined**: a group that passes all five entry guards and whose every
candidate window fails the floor emits no window row and no disposition row, so the partition is
neither total nor disjoint. Recording that group after the window loop keeps the floor a
window-level rule *and* the disposition total. `GROUP_QC_REASONS` is frozen at six and
`group_qc_row` refuses an unlisted reason, which is what P15 checks against.

**A04 — P11's publication surface is the per-source diagnostics CSV, defined here.**
`SOURCE_DIAGNOSTIC_FIELDS` = `video, n_frames_decoded, pts_accepted, index_fallback,
monotonic_forced, cfr_fallback_rate, fps_nominal, latency_ms_mean, latency_ms_p95`. One row per
source, written from `process_source`'s `finally` arm so an interrupted run still reports what it
decoded. The counts therefore describe **frames processed**, never frames the asset holds.

**A06 — batch rulings on `test-m2u81` phase-1 rows P01-P08 (8 of 18 filled at MAIN's reserve).**
Every ruling below is already implemented at `f43e720`; they are recorded so the phase-2 suite
encodes the shipped semantics rather than re-deriving them.

- **P02/P04 identity — reading (A).** The authoritative comparison is the canonical **final** write
  root `<base>/<session_id>`, resolved non-strictly with existing symlink components dereferenced,
  against the canonical published root. Reading (B), comparing the user's base `D`, is rejected: it
  refuses a base that merely contains the tree while the actual destination is disjoint from it.
  Equality counts as overlap, and both strict directions refuse. The teammate is right that the two
  readings diverge — that divergence is exactly the ancestor-base case A02 rules ALLOW.
- **P05 — no file on capture-open failure.** A source that never opens returns before the clock
  exists, so nothing is written. A non-`None` request is not a promise of a file; it is a
  destination for a summary of frames actually decoded. Zero decoded frames *after* a successful
  open does write a row, with all three counters at 0.
- **P05 schema — A04 fixes it.** Exactly one row over `SOURCE_DIAGNOSTIC_FIELDS`; the rate is a
  formatted derived column, never a stored source of truth alongside the counts.
- **P06 — reading (B).** `process_session` verifies existence before announcing. It does not raise
  and does not synthesise a zero-count file: a camera that failed to open is a real outcome, and
  inventing a diagnostics row for it would publish a frame count for a source never read.
- **P07 — no stdout summary when `output_diag` is `None`.** Narrower than the teammate's proposed
  ruling, and compatible with it: P07 bans a file-existence claim, and the implementation emits
  nothing at all.
- **P08 — counters cover live captures, and dispositions are exclusive.** An accepted live monotonic
  value scores `pts_accepted`; the name is about the disposition, not the container field. A call
  that falls back and then still needs forcing increments **`monotonic_forced` alone**, never both —
  the branch is an `elif` chain classified by which arm produced the returned value, which is what
  makes the three sum to the call count.

**A07 — P16's golden wiring needs a dataset that actually drops, so a fifth one exists.**
Wiring the disposition artifact into the four existing golden datasets produces four header-only
goldens: correct bytes, but they pin no reason code, no row order and not D05's partition, and the
golden suite's own non-empty row floor cannot hold over them. That is the vacuous-predicate shape
M2.7.3's P07 and the zero-row CSV both recorded. `2d_drop` therefore joins `_2D_DATASETS` with
groups `((0, 91), (1, 3), (2, 20))` — one healthy group plus one per short-input drop reason —
truncating the healthy trajectory rather than rescaling it, so the drop reason is the only thing
separating the groups. Its golden carries 2 rows over 2 distinct reason codes, and 3 groups split
1 windowed / 2 dropped, which is D05 measured on committed bytes. The four header-only goldens stay:
they pin the artifact a clean run must still write, which is P14. Existing goldens are unmoved —
`git status` over `tests/goldens/` reports additions only, which is P16 measured.

**A08 — the group-disposition artifact carries one schema in both modes and takes no 3D tags.**
`world3d_clinical_3d_group_qc.csv` publishes the same five columns as its 2D siblings, without the
nine identity tag columns the other 3D artifacts carry last. Kept deliberately: D04's whole reason
for a separate file is that one artifact explains dropped groups in either mode, and forking its
schema by mode would rebuild the per-mode split D04 refused. The tags are reachable if a later unit
needs provenance on it; uniformity is what a single reader validates against today.

**A09 — the group-disposition header is frozen at five columns, which D06 never did.**
D06 and A03 froze the reason CODES and left the column names unstated, so the diff-blind suite
encoded `(video, person_idx, n_frames, reason)` against a shipped
`(video, person_idx, n_frames, drop_reason, qc_status)` and three predicates read as failures on a
correct artifact. The shipped spelling is authoritative and is what A07's committed goldens carry:
`drop_reason` because `qc_reason` already names the per-metric window verdict and one file must not
carry two columns a reader can confuse, and `qc_status` because the window-qc artifact publishes a
verdict column and a disposition row is a verdict. **A predicate quantifying over a table's cells
needs the header frozen beside the value set** — freezing the codes alone left three of five reds
adjudicating a name nobody had chosen.

**A10 — two real defects, both found by the diff-blind suite and both in the `finally` arm's blast
radius.** (1) `write_source_diagnostics` did not create its destination's parent, so a missing
directory raised `FileNotFoundError` **over whatever exception was unwinding** — A04 puts that write
in `finally` precisely so an interrupted run still reports, and an unwritable destination turned
that into traceback destruction. Fixed by `mkdir(parents=True, exist_ok=True)` at the write site.
(2) P11 requires the fallback rate to reach the diagnostics CSV **and a stdout summary**; only the
CSV was written, and `Wrote diagnostics:` carried no rate. Fixed by printing rate, both fallback
counts and the timestamp total. Neither is reachable from the session path, which creates its output
directory and reads the CSV — which is exactly why only a contract-driven suite found them.

**A05 — P07 is satisfied at the reporting site, not only at the writing site.**
`process_session` printed `Wrote CSV:` and `Wrote diagnostics:` unconditionally. A source that never
opens returns before either file exists, so the fix reports both from the filesystem
(`produced.exists()`) rather than from intent.
