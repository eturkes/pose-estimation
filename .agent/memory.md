# Project memory

Context retained only when source, tests, technical docs, roadmap, and git do not expose it cheaply.

## Data boundary

- Patient recordings + adjacent derivatives = sensitive. `output/`, real-data calibration files/directories, and logs stay outside agent context. **`videos/3-cam/` carries standing decode clearance** — MAIN and teammates may decode it, run the pipeline over it, and write derived outputs. Chat and reports carry redacted aggregates only: never imagery, filenames, or subject identifiers.
- Repo scope = `videos/3-cam/`. `videos/initial/` = preliminary, retired — kept on disk, never reprocessed. Sibling directories under the same data root are out of scope: `harness/` holds schematics for a capture harness that was never built, `database/` holds the hospital's SCI clinical records and is the eventual integration target (→ roadmap Backlog).
- `videos` is a symlink out of the repo, so git never traverses it. Path-taking tools still default to the old flat root, and both are non-recursive: `scripts/run_report.py --videos-dir` and `pose-estimation-run --list-sessions` each need an explicit subdirectory.
- Commit source + synthetic or de-identified fixtures only. `.gitignore` keeps raw media, derived outputs, rig geometry, credentials, and path-bearing logs out.
- A collaborator fork carrying unrelated history reconciles by content onto `main` with source-only attribution → importing its Git object history would retain patient data even after a file deletion.

## Corpus registry (`pose-estimation-inventory`)

- Family identity for the whole project. `capture_id` = `s{subject_ordinal:02d}-{task}-{side}` names a task-side FAMILY and carries no take component; a retake stays an asset inside that family with `repeat >= 1` and raises `view_conflict`. **A `view_conflict` family holds more than one physical take, so nothing may bind calibration or a session to it** — the instance grain is an M2.2 question. The full identity is the pair `(grammar_version, capture_id)`: a grammar migration can move membership while the readable key survives. Measured: 2 of 188 families conflict, and the corpus holds exactly one repeat-marked asset, inside one of them.
- `capture_id` is a low-entropy stable pseudonym — it supports linkage and does not resist enumeration. `asset_id` = blake2b-64 over the corpus-relative POSIX path (`surrogateescape`, so it is total over non-UTF-8 names); it is not unique by construction, and the guarantee is the explicit collision check that refuses to publish. A keyed HMAC was considered and rejected: a lost key makes every downstream identifier unreproducible, and determinism from a clean base is worth more than opacity.
- `inventory/` is gitignored and patient-adjacent — `assets.csv` carries source paths, so `--out` is as sensitive as the corpus, and the tool sets no file mode. `census.json` holds aggregates alone and is the only artifact whose numbers may be quoted. **Every consumer calls `inventory.validate_generation(out_dir)` before reading a row**: `generation` carries three digests — both CSVs and the census over its own remaining fields — so a half-published set, an edited CSV and an edited census all fail. Publication is per-file atomic, not set-atomic; detection is the guarantee, not prevention.
- Schema owner = `docs/technical/inventory.md`.
- **Two committed gates back the registry's claims; rerun both before quoting any of them.** `scripts/run_inventory_mutations.py` mutates 72 predicates across `inventory.py` + `video_io.py` and demands a committed red per mutant — 71 killed, `M028` alone surviving as a ruled equivalent, because both validation orders raise `InventoryError` and no contract pins precedence between two simultaneously corrupt tables. `scripts/check_inventory_determinism.py` runs 20 sweeps proving the three artifacts are a function of corpus bytes alone (hash seed, four locale settings, shuffled `iterdir`, path spelling, `--out` name, timezone, `umask`, `-O`), then puts 13 tamper classes through the consumer boundary and checks the **exception class**, so a leaked `OSError` reads as a failure rather than a pass. Both stream to `tests/inventory_*_results.json` per unit and refuse to append to a file measured against different source digests. The mutation catalogue self-validates — each patch must match exactly one occurrence and change bytes — so a stale mutant fails loudly instead of scoring a silent no-op kill. A new predicate earns a mutant in the same commit, and a new test file must join the runner's `TEST_COMMAND` or the oracle cannot see it.
- The lesson that produced those gates: `30280c3` shipped 12 review fixes with 96 tests green, and the first campaign showed **18 of 72 mutants surviving** — a green suite measured nothing about whether it pinned the fixes it was written for. Fix-plus-test is not fix-plus-*pinning*-test.

## Session tree (`pose-estimation-sessions`)

- **Upstream digests prove bytes, never shape.** `inventory.validate_generation` proves the registry on disk is the one that was published; it says nothing about duplicate ids, an unknown disposition, an absent column, or a `view_conflict` cell that contradicts its own rows. Every consumer re-derives what it reads: `sessions.plan` derives the conflict from the canonical rows by the registry's own rule and requires the published cell to agree. A future consumer inherits that obligation.
- **Three standard-library calls normalize away the property under test.** `Path.readlink()` drops a leading `./`, so link text compared through it is not the link's text — `os.readlink` is. `Path.is_dir()` follows a symlink, so a directory test alone lets a link to an outside tree pass as a child; take the kind from `is_symlink()` first. `shutil.rmtree(p, ignore_errors=True)` refuses a symlink and swallows the refusal, so link debris survives a cleanup that reports success.
- **A path-prefix containment test breaks at `/`.** `parent + os.sep` yields `//`, which no real path carries, so a filesystem-root corpus classifies every file below it as an escape. Strip the separator from the parent before appending one.
- **Poll a worktree teammate's report inside its worktree.** A teammate whose cwd is
  `.scratch/worktrees/<name>` writes `.scratch/agents/<name>.md` **there**, which is a different file
  from MAIN's `.scratch/agents/<name>.md`. MAIN's copy keeps the unfilled seed forever, so a flush
  counter reading it reports `0/N` against an agent that is fully on pace — one such agent reached
  its supersession trigger while at 5/6. Always poll
  `.scratch/worktrees/<name>/.scratch/agents/<name>.md`, and copy that file into MAIN's
  `.scratch/agents/` before removing the worktree, because `.scratch/` is gitignored in main and the
  report does not survive teardown otherwise.
- **A worktree branch tip moves after the poll that read it.** Deleting a branch prints `was <sha>`
  for a commit the poll never saw; check ancestry (`git log --oneline -3 <sha>`) before concluding
  work was lost, since the later sha is normally a descendant carrying one more report edit.
- **Gate invocations must strip the host OpenVINO leak.** Every gate command in this project runs as
  `env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync <cmd>`. Both halves are load-bearing:
  `PYTHONPATH` selects the primary tree for the non-pytest half, and unsetting `LD_LIBRARY_PATH` keeps
  the host OpenVINO build out of the loader path. Without them `conftest.py` dies at
  `ImportError … GLIBC_2.43 not found` before collecting one test.
- **`scripts/check_qualify_determinism.py` refuses to overwrite a result measured against different
  source digests** — that refusal is the stale-green barrier, so an intentional source change needs
  an explicit `rm -f tests/qualify_determinism_results.json` first. Any schema rename must also reach
  the script's own fixture, which builds real sidecar rows and fails loudly with
  `ValueError: dict contains fields not in fieldnames` when it lags the schema. Its `SOURCE_FILES`
  tuple is the tripwire's whole reach: a new module that shapes the published bytes is invisible to
  it until listed, so every module added to `measure.AXES` joins that tuple in the same commit.
- **Regeneration is a fixpoint, so it must be the last action before the commit.** `tests/test_qualify.py`
  sits in `SOURCE_FILES` *and* holds
  `test_m2u5_p20_committed_determinism_evidence_matches_its_sources`, which compares every recorded
  `source_sha256` against current bytes. Adding one test after regenerating therefore re-breaks the
  evidence the regeneration just produced. Land every source and test edit first, regenerate once,
  commit. The 40-sweep run costs minutes, so a late red suite harvest pays for it twice.
- **`scripts/run_m2u5_mutations.py` mutates the primary tree's `src/` in place.** It restores under
  `try/finally` and verifies the restored digests, so an interrupted run is safe, but a *concurrent*
  reader sees mutated source: a mid-run `grep` reported `GENERATOR_VERSION = "v3"`. Never run it
  beside anything that reads or edits `src/`, and re-read a surprising source line after it exits.
  Same shape as `scripts/run_inventory_mutations.py`.
- **A cell alphabet must be the token set, never a shape that resembles it.** `STATUS_CELL =
  re.compile(r"[a-z_]+")` guarded both `offset_status` and `sync_status` and accepted every lowercase
  token the two partitions exclude, so an invented status would have published cleanly through a
  check whose whole job was refusing one. `_token_alphabet(frozenset)` builds the pattern from the
  constant set, which also makes a token added to a partition reach its alphabet automatically. The
  tell: an alphabet expressed as a character class where the contract names an enumeration.
- **Published row order is contract-bearing, and two tables answer to different rules.**
  `_canonical(rows, key)` sorts `pairs_qc`/`cameras_qc`/`events_qc` at the publish site so their order
  is a function of the rows rather than of a loader's return order. `assets_qc.csv` is deliberately
  exempt: it publishes in registry order, which groups a capture's assets for a reader, while
  `asset_id` is a content hash whose ordering means nothing — `check_m2u5_determinism.py`'s D09 pins
  that instead. Permuting a loader's already-canonical return is fault injection past a validated
  postcondition, so the campaign leaves `load_assets` unwrapped by design.
- **A published artifact must not contradict its own census.** Ingesting the sync axis filled
  `pairs_qc.csv` and left all 193 `events_qc.csv` rows at `sync_unmeasured` while
  `qualification.json` claimed the axis measured. After wiring any axis into one table, grep the
  other tables for that axis's unmeasured sentinel before publishing.
- **Event membership lives in `sessions/placements.csv`, never in the capture family.** A
  view-conflict family resolves to several single-camera events, so any family-wide derivation
  credits each of them with cameras it does not hold — it published `above|left` on 7 single-camera
  events. Where the session tree already publishes a per-event cell, copy it; re-deriving published
  text is how two spellings of one fact drift apart.
- **Two closure statistics exist and neither substitutes for the other.** `events_qc.csv` groups by
  event and accepts on R6's fused verdict (30 triangles, 5.403/30.286 ms). `scripts/probe_sync_policy.py`
  groups by capture family and accepts on audio alone (35 triangles, 4.451/30.286 ms — the P38
  figure). Same corpus, same estimator, different populations. Always name the population beside the
  number.
- **`--corpus` is `videos/3-cam`, never `videos`.** `videos/` holds two trees, `3-cam` (the 16 subject
  directories this milestone measures) and `initial` (12 loose `IMG_*.MP4` files outside the registry).
  `assets.csv` `source_path` is relative to `3-cam`, so pointing `--corpus` one level up fails every
  row with `SessionsError … reason="source_missing"` before a single measurement runs. `videos/` is
  patient data and Read-denied; probe it with Python `pathlib`, never `ls`.
- **Publication replaces a whole tree, so the output is a destructive path.** `--out` must overlap neither `--corpus` nor `--inventory` in either direction; a symlinked `--out` publishes to the path it resolves to, which keeps the link and replaces the tree it named.
- **Sweep crash debris only after the swap lands.** A kill between the two renames leaves the sole complete generation as a *retired sibling under a dead pid*, so a sweep before the swap deletes it and a failed swap then has nothing to restore. The empty-root rollback needs its own `retiring.exists()` guard too, or it raises `FileNotFoundError` over the real error.
- **Four validation predicates read as stricter than they are.** `Pattern.match` with `^…$` accepts one trailing newline, so every alphabet must use `fullmatch` (an exported pattern needs `\Z`, because consumers call `.match` themselves). `str.isdigit()` is true for superscripts, which then raise `ValueError` out of `int()`, and for other scripts' digits, which `int()` silently normalizes into a value the cell never spelled — require ASCII `[0-9]+`. A CSV with zero rows carries its schema in the header alone, so per-row column checks never run and a short header publishes an empty artifact instead of failing. `os.kill(int(pid), 0)` raises `OverflowError`, not `ValueError`, on a suffix wider than a C long.

## Upstream instruction refreshes

- `CLAUDE.md` + `.claude/commands/*.md` arrive as upstream drop-ins landing over local adaptations, and two successive refreshes dropped the same clause: `session-roadmap.md:3`'s DESCOPED terminal status, the only rule keeping the DESCOPED M3 out of MODE dispatch once M2 reaches REVIEWED (`1a26bdc` restored it the first time). After every refresh, diff each refreshed file against its prior commit and re-apply every local clause the drop-in lost; `git log --grep upstream` lists the reconciliations. `.agent/roadmap.md`'s M3 heading now declares its own terminality, so a third drop leaves the dispatch decision intact.

## Publisher trust roots — the surface no digest covers

- **Every self-describing publisher has exactly one unverified file: its own marker.** `tree_digest` must exclude the marker because the marker carries that digest, so nothing inside the set covers it. Three properties therefore have to be checked directly, and `measure` had all three while `qualify` had none until M2.3 window 8 — sibling publishers drift, so check each one rather than assuming the idiom spread. (1) The marker is a regular non-symlink file (`lstat` + `S_ISREG`); a symlink puts the trust root outside the set, and where ownership gates a recursive delete it lets a foreign directory license its own deletion. (2) The parse rejects duplicate keys (`object_pairs_hook`); `json.loads` keeps the last silently, so one document carries two claims. (3) The census digest covers the provenance block minus its own self-referential key — excluding the whole block leaves the upstream claims consumers trust most as the only uncovered content in the set. The four publishers are `inventory`, `sessions`, `qualify`, `measure`; **unverified: whether `inventory` and `sessions` do (1) and (2) on their own markers.**
- **A keyless digest detects, it does not authenticate — never let a pass restate it as tamper-proofing.** No artifact in this project carries a signing key, so anyone who edits a claim and recomputes the digest produces a document every check accepts. What is ruled out is corruption and every edit that stops at the claim. Both `qualify.py` and `docs/technical/qualification.md` state the bound; keep it stated.
- **Ownership includes the generator version, so a version bump orphans the previous tree.** `_is_own_generation` requires *this* generator's version, which is what keeps a foreign tool's directory from licensing its own recursive deletion — and the same rule means a bumped generator does not own its predecessor's output. `qualify.run` then refuses to replace it, so the operator must `rm -rf` the `--out` tree by hand before the first run of a new `GENERATOR_VERSION`. Not a defect: the alternative is a version-blind ownership test, which is the hole the check exists to close. Applies to `check_qualify_determinism.py` too, whose own `rm -f` barrier is separate and additional.
- **Committed evidence binds to source digests, never to a head SHA.** A regeneration run always precedes the commit that carries its output, so a recorded `git rev-parse HEAD` names the parent state and can never be checked — regenerating to fix it just re-lags. `check_qualify_determinism.py` dropped the field; `source_sha256` over `SOURCE_FILES` is the real dependency and refuses regeneration on mismatch. **Open instances: `scripts/check_inventory_determinism.py:257` and `scripts/run_inventory_mutations.py:817`.**

## Corpus geometry — what the three cameras do and do not share

- **Cross-view scene-feature correspondence does not exist in this corpus.** All 246 within-family pairs: 0 recoverable poses, mutual SIFT matches median 13.5, F-inliers median 8.0 = the algebraic minimum for a fundamental matrix, so those inliers are a minimal-sample fit and carry no evidence. Never plan calibration, view verification or scene SfM on cross-view matching here. `scripts/probe_crossview_pose.py`, ruling → `.agent/archive/rulings-m2u3.md` R4.
- **That null has two controls, so it is geometric rather than procedural.** Baseline ladder (`scripts/control_crossview.py`): same frame 2812 mutual matches → same asset adjacent 1252 → same asset far 962 → cross-view 12–19, with 1740–3355 SIFT keypoints present in every view. And in-corpus: the only 2 rich pairs of 244 are the `above|above` (298.5) and `left|left` (732.5) `view_conflict` pairs — correspondence returns exactly where two cameras share a viewpoint. A cross-view null therefore needs no re-litigation; a *positive* cross-view result would be the surprising claim.
- **Extrinsics route = the subject's own keypoints**, where correspondence is assigned (keypoint `k` in A is keypoint `k` in B) rather than matched. Supported by `detect_rate` median 1.0 over 379 assets, 133 keypoints per detection, and sync agreeing across two independent modalities to median 12.89 ms. Degrades on near-coplanar keypoints and a near-static subject — both live risks for seated upper-limb tasks.
- **`(device_config, view)` is the coarsest key naming a stable geometry.** The view token alone is not one: `above` is 85% rigidity-unmeasurable on iPad(5)/16.6 and 3% on Air-M2/26.5. **iPad(5)/16.6 `above` = 89 assets, 23% of the corpus, an unstable camera** — `valid_fraction` median 0.212 against 1.000 in every other cell, `decode_status` ok on 89/89, highest quiet-border motion energy, and the landing site of 27 of the visual spike's 28 independent flags. `left`-vs-`right` handedness is unresolved by anything in this corpus.
- **A gate constant must never double as an instrument parameter.** P21 accepted `drift_p95 ≤ 4 px` while the retired producer used that same constant as the MAGSAC inlier threshold, so no residual above the gate could ever be reported and 210/286 assets "failed" a gate judged against a quantity it pinned. The tell is a near-constant statistic sitting inside its own gate; the test is a threshold sweep. `residual_p95` tracks the threshold monotonically over 8× while inliers grow 6% — it measures where the cut falls, not the scene. `measure/rigidity.py` now separates `RANSAC_THRESHOLD_PX = 8.0` from `DRIFT_P95_GATE_PX = 20.0`, pinned by `test_r2_gate_is_independent_of_ransac_threshold`. Sweep any statistic before gating on it, ratios included.
- **A population count carries the instrument setting that produced it.** Rigidity's eligible 298 is not a corpus property: holding the gate at 20 px and sweeping RANSAC 4→48 px moves eligibility 298→311, plateauing at 32 px (`scripts/probe_rigidity_saturation.py`). Every recovered asset lands in `camera_motion`, so rigid 280 and families 71/137 are threshold-invariant — a loosened instrument buys assets that fail the gate, never extrinsics. Quote a denominator with its threshold or not at all; the same sweep is what distinguishes the two cases.

- **"Families connected" names at least two different statistics — always say which.** P38's 122/137 quantifies over *one camera per view*: a family counts when some one-asset-per-view selection is spanned by **cross-view** accepted pairs, which is what M2.6 consumes, since a family holding two files of one view needs only one and a same-view edge carries no cross-view geometry. Whole-family connectivity (every asset joined, same-view edges counted) gives **121** on the identical 210 accepted pairs. Both are in `scripts/probe_sync_policy.py` as `families_view_recoverable` and `families_all_assets_connected`. A one-family gap between a port and its spike is this, not an estimator defect — check the rule before opening an investigation. The visual spike's 26/137 still matches neither and is an open register row.

## Geometry recovery from subject keypoints

- **Structural checks and geometric checks disagree, and only the geometric ones are evidence.**
  M2.6's pilot ran pooled two-view `recoverPose` over synchronized human keypoints on 3 three-camera
  events / 9 pairs / 72 pair-frames: a pose came out for **9/9** pairs, **7/9** cleared 30 cheirality
  inliers, and the accepted edge graphs connected **3/3** events. Every one of those is a structural
  property. Then the three-camera rotation cycles closed at **34.87-59.03 deg (median 47.34)** against
  a predeclared 10 deg bound — **0/3** — with 0/9 pairs yielding a per-frame quality pose, split-pose
  rotation spread median 51.16 deg (max 168.31), and held-out epipolar support below 0.5 on all 4
  pairs where it was measurable. A gate keyed on pose existence, inlier count or graph connectivity
  publishes this as recovered extrinsics. **Cycle closure and temporal stability are the cheapest
  checks that refuse it; run them before believing any pose.** Probe ships at
  `scripts/probe_calibration_observability.py`.
- **Alignment was not the blocker and the probe proved it separately.** Realized reference-time
  residual over the same sampled frames is median **6.31 ms**, p95 21.88, max 32.71 — inside one
  33.3 ms frame. Always separate the sync residual from the geometry verdict; otherwise a geometric
  null gets misread as an alignment defect and re-opens a closed unit.
- **A null bounds the route it ran, never the route it skipped.** That pilot bounds *pooled pairwise
  `recoverPose` initialization*. It says nothing about linear rotation/translation recovery followed
  by bone-length-regularized bundle adjustment under a per-model intrinsics prior, which is the
  published route (Lee et al., IEEE RA-L 7(4) 2022) and which the pilot never ran. State the unrun
  arm explicitly beside every negative.
- **Extrinsic recovery from subject keypoints is CLOSED NEGATIVE on this corpus, and the cause is
  measured.** Cross-view keypoint correspondence bias: per-keypoint mean signed epipolar residuals
  reproduce at split-half **r = 0.703** across disjoint held-out frame blocks (39 pairs), against
  calibrated references of **0.010-0.120** for zero-mean noise and **0.993-0.998** for fixed bias;
  residual magnitude median **20.8 px**, systematic component **15-20 px** at 1080p, an order above
  the 1-4 px regime published keypoint calibration works in. Refuted as causes: planar degeneracy,
  low parallax, alignment, undersampling, estimator/initialization. Two repairs priced and refused —
  independent pairwise BA makes closure **worse** (median 39.0 → 78.9 deg, damage growing with data,
  the biased-estimator signature), and no low-bias keypoint subset beats all 65 on held-out events
  (cleanest ten still 49.6-53.9 px). The negative bounds RTMW-L keypoints on this corpus under
  per-model intrinsic priors; it does not bound a different keypoint source or a calibrated capture.
  **The detector and the viewpoint separation are what a future attempt must change — not the solver
  and not the sample size.** `scripts/probe_calibration_bias.py`, contract §11b A09-A13.
- **Calibrate the instrument against known ground truth BEFORE interpreting any result from it.** Two
  windows were spent narrowing hypotheses against an uncalibrated instrument. The synthetic positive
  control — known extrinsics driven through the real cache's own validity masks, sizes, models and
  event population, so only the geometry becomes known — cost one script and returned cycle closure
  **median 0.000 deg** at zero correspondence error. That single number exonerated the estimator,
  priced the error budget in px (10/10 events close out to sigma = 8 px; the corpus's 2/10 needs
  ~30 px), and supplied the reference distributions that made the bias statistic readable at all.
  A null is not interpretable until the instrument's response to a known input is known.
- **A statistic that discriminates needs calibrated references, not just a suggestive value.** "Bias
  is the only surviving hypothesis" and "bias measured at r = 0.703 against 0.010 noise / 0.997 bias
  references" are different epistemic objects; only the second refuses repair routes or gets
  published. Where a diagnostic is invented, run it on synthetic data spanning both mechanisms across
  the magnitude range that brackets the real value, and check the confound explicitly — here the
  sigma-32 noise arm carries a pose as wrong as the corpus's and still returns 0.120, which is what
  proves a wrong pose alone does not manufacture the structure.
- **Held-out reprojection on the solve's own keypoints is self-consistency, never accuracy.** Pätzold
  et al., GCPR 2022 measured a keypoint-recovered calibration beating the reference calibration on
  human reprojection (4.01 vs 4.57 px) while losing to it on independent AprilTags by 3.05 px (5.00 vs
  1.95 px). Same discipline as M2.3's acoustic closure: an in-family statistic certifies consistency
  and cannot certify correctness.
- **Writing an arbitrary-scale extrinsic into `CameraCalibration` falsifies six shipped surfaces.**
  `_types.py:91` calls `tvec` metres, and the claim propagates: calibration docs ->
  `triangulation.py:12,454` -> `export.py:443-487` (world columns named `_x_m/_y_m/_z_m`) ->
  `validation.py:1265-1293` (multiplies by 1000, renders millimetres) -> `analysis/clinical_features.R`
  (`coord_space="world-metric-3d"`, `distance_unit="m"`). The projection math is unit-agnostic, so
  every stage accepts arbitrary units numerically while its contract lies. Any arbitrary-scale
  geometry needs its own type with explicit scale provenance.
- **`session.json` is unpatchable after publication** — `sessions.validate_generation` hashes the
  tree, so an in-place manifest edit turns a valid generation invalid. Any later unit that wants to
  add a field republishes the tree or publishes beside it.

## Frozen contracts carry stale numbers

- **A contract's stated census can be derived under a rule a later predicate replaced — re-derive
  every census at implementation time rather than trusting the frozen text.** M2.5's P19 published
  "329 offsets / 50 not" while its own P07 mandated partial publication, whose true census is
  **355 / 24** over 379 rows. 329 is a real number naming a different population — cameras inside a
  graph-connected event — so the defect was invisible to every consistency check that treated it as a
  count. The recount cost one broadcast to four live teammates; waiting for review would have cost a
  wrong published artifact. Whenever one predicate changes which rows exist, grep the contract for
  every number quantifying over rows and recompute each one.
- **Name the population beside the count, always.** This project has now hit the same trap three
  times: two closure statistics over different event sets, two "families connected" figures, and now
  329-vs-355. A bare count is the shape of defect that survives review.

## Multi-agent traps

- **A superseded ruling in a shared file poisons every downstream artifact.** M2.2's rulings file kept A07 and A12 after two later rulings replaced them; the test teammate conformed to the file, and its suite read as a regression against correct code. Amend a ruling in place, in the same turn as the code it rules — a "later rulings win" note at the bottom is not enough, because the table is what gets read.
- **Keep a spike's worktree after picking a design.** M2.2's rejected spike carried an independent 184-line generator; rerunning it against the shipped one made it a differential oracle — identical partition over 193 blocks and 379 assets, 0 `capture_id` disagreements, from code sharing no line with the winner. Keeping the bytes costs nothing and the evidence is unreproducible once the worktree is gone.
- **A reviewer's worktree diff against `main` carries its own staleness.** `wt/rev-m2u2` showed a *revert* of a MAIN change that landed after its branch point. Read a reviewer's diff as findings, never as a patch to apply.
- **A saturated reviewer's persistence is a context artifact, not evidence.** `rev-m2u3` re-filed a strengthened red against one closed ruling twice while above 97%, each time reading the ruling as unresponsive rather than as decided. Weigh a finding by its argument; when the same row returns from an agent near its ceiling, restate the ruling once and stop, and prefer a fresh successor to another round with the saturated instance. The inverse trap is real too — see the next bullet — so the discriminator is the reviewer's gauge, not its insistence.
- **A reviewer's DONE marker is not the end of its yield.** M2.2's two reviewers sent nine more accepted defects after both phase-2 markers, four of them in the minutes before the close commit. While a reviewer has context left, keep it pointed at the surfaces MAIN just changed and take findings by message; ask for the report file last, since writing it costs the context that finds the next defect.

## Media fixtures (PyAV)

Every audio-bearing fixture in `tests/` is muxed by PyAV; three ordering rules decide whether the file a test writes is the file it meant.

- **Set `layout` (and every other codec-context attribute) immediately after `add_stream`, before the first `mux` call.** Muxing opens every codec in the container, and an opened codec context ignores a later attribute write — silently. A video-only `mux` earlier in the same function is enough to freeze the audio stream's layout at its default.
- **An audio frame carries no `pts`; the encoder assigns it.** Setting one is not how a silent track gets its timing — supply `sample_rate`, `format` and `layout` on the frame and let the encoder do it. The video path is the opposite (`frame.pts` is set explicitly, `tests/test_measure_cache.py`), so the two must not be written from one template.
- **A declared stream with no packet never reaches the container.** `add_stream` alone produces a file whose header the probe reads and whose `streams.audio` is empty, which reads exactly like a real no-audio asset. A fixture meaning "has audio" must mux at least one encoded packet and flush the encoder.

## Environments

- The container + host share the checkout through different absolute paths, so uv environments are layer-specific. Container work uses `.venv`; host work uses `.venv-host`. Recreate the matching environment after a move; repair text shebang/activation/editable-path metadata only when offline, and regenerate binary/cache artifacts.
- **Source the accelerator env before any in-container OpenVINO run** (`CLAUDE.local.md` → `~/agents/docs/openvino.md`). The inherited `PYTHONPATH` selects the *host* OpenVINO build, which needs glibc 2.43 and raises `ImportError: … GLIBC_2.43 not found` in the container — a loud failure, not a CPU fallback. Sourcing prepends the container build. Stripping the env entirely (`env -u PYTHONPATH -u LD_LIBRARY_PATH`) falls back to the `.venv` pip wheel at `['CPU']`, which is what a generic checkout gets. Confirm with `openvino.Core().available_devices` in the correct uv environment.
- **The same leak kills a PRIMARY-tree gate run, not just a worktree one.** `tests/conftest.py` imports `pose_estimation` → `openvino`, so a bare `uv run --no-sync pytest` in the primary tree dies at `ImportError … GLIBC_2.43 not found` before collecting a single test. `PYTHONPATH="$PWD/src"` belongs in front of **every** gate invocation in both trees — the worktree recipe's `PYTHONPATH` export was doing double duty and hid this.

## Device placement — detector vs pose

- Detector + pose model take separate devices (`--det-device` / `--pose-device` on `run`/`main`/`benchmark`/`validate`). No `--device` flag exists anywhere; a bare `--device` is an argparse error, not a silent default.
- **rtmlib YOLOX must not run on NPU.** In-graph NMS ⇒ dynamic `dets` shape; NPU demands static ⇒ fixed 100-row buffer whose unused rows are never written. Symptom: every frame reports exactly 100 detections, all rows sharing one score, values outside `[0,1]` (observed 1.128, 1.263). CPU on the same frames returns 1–3 detections, max 0.918. Compiles cleanly ⇒ `rtmlib_openvino.py`'s NPU→CPU fallback never fires; failure is numerical, silent, and reaches the CSV.
- Pose models are NPU-safe: RTMW-L NPU vs CPU = 0.505 px mean / 2.265 px p95 / 5.3 px p99 keypoint deviation, score MAE 0.00056. Per-call 7.17 ms NPU vs 134.26 ms CPU (~19×). Detector 109.82 ms NPU (garbage) vs 445.21 ms CPU. Projected 15 455-frame batch: all-CPU 51 min, det-CPU/pose-NPU 18 min.
- MediaPipe is unaffected — SSD anchors + NMS decode in Python (`detection.py`), graphs stay static-shaped ⇒ both roles default NPU. `models.DETECTOR_MODELS` selects which compile on `--det-device`.
- **The padding is a device property, and only NPU pads with garbage.** One synthetic probe separates the three devices with no patient data: read `yolox_m_8xb8-300e_humanart-c2c7a14a.onnx` (rtmlib cache), compile per device, infer `np.zeros((1,3,640,640))`, print `dets`/`labels` shape + range. CPU keeps the dynamic output (`(1,1,5)`); GPU and NPU both materialise a fixed 100-row buffer with `labels` all `-1`; only NPU's `dets` hold uninitialised memory (`-0.1471…1.0215` on an all-zero image, so scores clear any threshold), while GPU's read exactly `0.0000`. That makes GPU a live candidate for the detector and keeps NPU excluded. Median latency on that probe: GPU 9.7 ms, NPU 108.4 ms, CPU 213.1 ms.
- Reproduce the real-frame findings: `.scratch/det_npu_vs_cpu.py`, `.scratch/pose_npu_vs_cpu.py`, `.scratch/device_timing.py` (scalars only, no imagery/identifiers) — these decode patient clips, so they need clearance; the synthetic probe above does not.

## rtmlib `PoseTracker` — stateful, and `tracking=False` does not disable it

- `PoseTracker.__call__` guards its stateless branch with `not self.tracking and self.det_frequency != 1`. **`det_frequency=1` sends `tracking=False` down the IoU-tracking branch anyway**, where each frame is matched against the previous one at `tracking_thr=0.3` and unmatched boxes are dropped. `bboxes_last_frame` then holds tracker state, never the current frame's detections.
- Fatal for **sampled** frames: seconds-apart samples share no IoU, so the list empties permanently and any count taken from it reads 0. M2.3's detectability run published `detect_rate` median 0.0 over 379 assets from exactly this, while the detector saw a subject in 24/24 probe frames. One tracker instance reused across assets compounds it.
- Sampled-frame analysis must drive `det_model` + `pose_model` per frame and take counts from the detector return. Reserve `PoseTracker` for consecutive video (`run.py:758`, rtmlib's intended mode; note its default `tracking=True` still drops a person whose frame-to-frame IoU falls under 0.3).
- Defaults worth knowing: `det_frequency=1`, `tracking=True`, `tracking_thr=0.3`, `backend='onnxruntime'`, `device='cpu'`.

## Devices available in-container

`Core().available_devices` = `['CPU','GPU','NPU']` — Core Ultra 7 268V, Arc 140V iGPU, AI Boost NPU; `intel-accel/selftest.py` reports `OK correct=True` for all three under OpenVINO 2026.3. The two historical GPU blockers are both cleared by the `intel-accel` farm: it links the host `libstdc++` for the driver's `GLIBCXX_3.4.35` requirement, and it pins an IGC matched to the host GPU driver, which removes the compute-runtime abort at `command_stream_receiver.cpp:1205`. An IGC that falls behind the host driver reinstates that abort and kills CPU and NPU work in the same process → after a host Intel driver update, rebuild the farm and re-run the self-test.

## R analysis layer — non-obvious hazards

- **Retention rule, standing (user).** M2's numerical output is eventually destined for the `../rehab` dashboard over the hospital SCI database. Any artifact serving that path is kept whatever its milestone's status, DESCOPED included, and is never traded away for tree tidiness. Anything off that path is deleted outright rather than tagged or archived — a superseded artifact costs a future agent a wrong start. The decisive question is which of the two an artifact is, never how old it is. **Operational test for a retained branch or oracle = one named open dependency.** `wt/spike-m2u3-audio` holds because R6's connectivity-reconciliation polish row consumes its `_family_coverage` as the P38 oracle; `wt/test-m3u3` holds because its remaining diff-blind cases are exactly the five cut 3D clinical groups on the `../rehab` path. Everything else went, and each deletion was decided on content rather than label: `spike-m3u3-long`/`-wide` implemented the losing side of frozen carrier decisions plus two specifically forbidden changes (R-10, R-11); `spike-m2u2-a` was a design not picked, confined to gitignored `.scratch/`; `spike-m2u2-b`'s oracle role ended when M2.2 closed, its one unique verdict already recorded in `rulings-m2u2.md` A08. Prove no-dependency by sweeping `.agent/ scripts/ tests/ src/ docs/` with a positive control before deleting — an empty search and a broken search print the same bytes.

- `analysis/utils.R:59-87` `aggregate_per_video()` treats **every numeric non-metadata column as a feature**. Adding a count, coverage or QC column to an output that legacy consumers read makes it enter per-video means, z-scores, correlations and PCA silently. The R gate invokes no downstream consumer (`tests/test_r_pipeline.py` covers the producer, `features.R`, `arthrose_diag.R` only), so the full suite stays green while downstream tables change meaning.
- The 2D/3D partition is enforced by regex alone: consumers glob `_clinical\.csv$` / `_clinical_windows\.csv$`, which cannot match `_clinical_3d.csv`. Six consumers replicate that discovery, so widening one is a local edit with global consequences.
- Producer keys are `video`/`person_idx`/`window` only. No task, condition, trial or session identity exists anywhere in the schema — any "session" or "trial" grain has to come from metadata that does not yet exist, not from the CSVs.
- Gate constants are duplicated across languages: reprojection 20 px and triangulation angle 1° live in `src/pose_estimation/triangulation.py:423-424`, `src/pose_estimation/validation.py:77,86` and `analysis/clinical_features.R:49,54`. Changing one silently desynchronises the R adapter from fusion.

- `stats::filter(x, rep(1/5, 5), sides = 2)` replaced `zoo::rollmean(x, 5, fill = NA, align = "center")` (M3.1 dropped `zoo`): NA propagation identical at centre, both edges and every hole pattern; values differ ≤2 ULP (2.8e-14 absolute on ~100-magnitude input) and no golden pins that column. `dplyr` masks `stats::filter`, so the call must stay namespace-qualified.
- **Two grids, deliberately.** Evidence counts the window's nominal slots; estimates keep `trajectory_grid()`'s narrower grid anchored on the first observed sample. `compute_window_features()` pads the kernel's `valid` mask by `lead_absent`/`trail_absent` and recounts through `grid_evidence()`. Widening the estimate grid instead would move `nj` through its `T_dur` term and break P08's byte-identical goldens. A new evidence field must pad; a new estimate must not.
- Window enumeration reads cadence as `nominal_fs(ts, magnitude = TRUE)`; movement segmentation reads `nominal_fs(ts, magnitude = FALSE)` (M2.4). The magnitude is load-bearing: a signed estimate drops a descending clip before any window is keyed, so the QC pass never gets to report `invalid_timebase` (V21-V24). `segment_movements()` keeps the signed form — no QC artifact depends on it. Only C2.07's regex pins the segmentation call site, so no numeric oracle proves that site consumes the value.
- **Golden-regeneration tests cannot prove an artifact's absence.** `regenerate()` copies a filename whitelist out of a staging directory it deletes, so an unexpected output never reaches the golden directory. Assert absence by running the producer into a preserved directory and listing it, with a positive control proving the run happened.
- `grid_evidence()` is the single masking path for QC counts (M3.3). Every frame/interval count, coverage, duration and gap figure must flow through it, so a group's evidence and the metric it explains can never disagree about which samples were usable. Adding a count elsewhere reintroduces exactly the producer/reader disagreement the unit exists to prevent.
- Artifact-name filters in consumers are blacklists and break silently on a new suffix. `_aggregate_clinical()` (`src/pose_estimation/validation.py`) skipped `windows`/`movement_phases` by substring, so `_clinical_3d_window_qc.csv` (singular `window`) would have entered the per-frame clinical means as metrics. Now selects per-frame artifacts positively by `_clinical.csv`/`_clinical_3d.csv` suffix. Check every consumer filter when adding an artifact; `analysis/*.R` globs anchor on `_clinical_windows\.csv$` and are unaffected, and directory-mode rescan exclusion is pinned by `test_world3d_outputs_not_rescanned`.
- **The 3D path deliberately skips `adapt_2d_confidence()`** (`analysis/clinical_features.R:1452-1466`). This looks like a missing gate and is not: `world3d.csv` confidence is a fused mean over already-accepted points, and fusion applied `min_confidence` upstream (`src/pose_estimation/triangulation.py:538,559-572`). Adding a 3D confidence predicate creates a new gate and moves every shipped 3D estimate. An M3.3 spike "repaired" this seam and was reverted.
- `sum(win_mask) < 4` skips a window entirely (`analysis/clinical_features.R`), so no row exists for it. Any per-window artifact covers emitted windows only, and changing the skip moves the shipped window row set.
- The three-frame gap at 30 Hz now lands exactly on the provisional 0.10 s threshold, because `nominal_fs()` reads 30.0000 where the legacy estimator read 30.03 Hz and computed 0.0999 s. Split slacks are what make that comparison decidable: `qc_policy_tolerance = 1e-4` relative on gap, `qc_coverage_tolerance = 1e-9` on coverage. One shared 1e-9 slack made the verdict cycle pass/pass/fail with clip length through the residues of 3, which is why two independent M3.3 spikes disagreed on this case under the biased estimator.

## Scratch validators pending port

- `.scratch/steq.py` — ASD-STE100 register scan over the human-facing surface (inventory: `docs/technical/conventions.md` → *Text register*). Drops fences/tables/headings/frontmatter, joins wrapped lines into blocks so a sentence is measured whole, splits on `.!?`, flags `LONG` (> `--max`; 20 for instructions, 25 for descriptions), `FILLER`, `CONTRACTION` (also fires on possessive `'s`), `PASSIVE` (be-verb + participle heuristic). Code-file mode samples quoted `help=`/`description=`/`title=` strings only. Measured at `--max 20`: `README.md` 14 → 2, `docs/capture_protocol.md` 20 → 7. Residual flags are 21-25-word descriptions, which the rule allows. Port scheduled in `.agent/polish.md`.
- `.scratch/fidelity.sh <base-ref> <file>…` — pairs with it: diffs the multiset of format specifiers, `--flags`, backticked spans, file names and numbers between a base ref and the working tree. A register-only edit must show no delta; every delta needs an explanation. Caught the p-value reformat (`p<.05` → `p < 0.05`) and confirmed 14 R files invariant.

## Worktree gate recipe (`.scratch/worktrees/<name>`)

Every teammate worktree runs the full Python gate concurrently off the one primary environment, read-only:

```sh
export UV_PROJECT_ENVIRONMENT=<primary-tree>/.venv PYTHONPATH="$PWD/src"
uv run --no-sync ruff check && uv run --no-sync ruff format --check \
  && uv run --no-sync ty check && uv run --no-sync pytest
```

- **Two resolution paths with opposite defaults — the most expensive trap in this project.** `pyproject.toml` sets `pythonpath = ["src", "tests"]`, which pytest resolves against **rootdir** and inserts at `sys.path[0]`. So `pytest` launched from a worktree imports **that worktree's** `pose_estimation`, with or without `PYTHONPATH`. Everything else — `python -c`, `python -m pose_estimation.<mod>`, `ty check` — resolves through the hatchling editable install to the **primary** tree's `src/`. `PYTHONPATH="$PWD/src"` is therefore mandatory for the non-pytest half of the gate and redundant for the pytest half. The review consequence is sharp: a reviewer running its red suite from its own worktree tests its own copy, so every primary-tree fix reads as still-broken. **Print `<module>.__file__` before believing a red**, and copy a test file into the primary `tests/` to exercise primary code.
- `UV_PROJECT_ENVIRONMENT` must be exported on **every** call: `uv run` inside a worktree otherwise creates and selects a worktree-local `.venv`, which `--no-sync` leaves empty, and the gate dies at `No module named pytest`.
- `--no-sync` keeps the shared environment unmutated, which is what makes concurrent worktree gating safe.
- Tool caches (`.ruff_cache`, `.pytest_cache`, `.ty_cache`, `.coverage`) are cwd-relative → already private per worktree; no extra state paths needed.
- `videos` is gitignored too, so a fresh worktree cannot see the corpus and every real-data command fails at a missing path rather than at a wrong answer. Link it read-only alongside the R library: `ln -sfn <primary>/videos <worktree>/videos`. Never write through either link.
- `renv/library/` is gitignored, so a fresh worktree has no R library and every R case SKIPs. Symlink it read-only and the worktree gate becomes fully equivalent: `ln -sfn <primary>/renv/library <worktree>/renv/library` → 469 passed/0 skipped, `tests/test_r_pipeline.py` 25 passed/0 skipped, same as primary. Concurrent worktrees share it safely because R only reads packages; never `renv::install`/`renv::snapshot` through the link.
- `inventory/` is gitignored and is now the third required link: `ln -sfn <primary>/inventory <worktree>/inventory`. Without it `tests/test_sessions.py::test_p05_real_corpus_headline_counts` SKIPs on the absent registry, and that single skip fails C8.08, whose A32 reconciliation demands zero skipped. `videos` + `renv/library` alone stopped being sufficient once a collection-reconciling case entered the suite. Read-only, like the other two — `inventory/` carries source paths and is as sensitive as the corpus.

## Session launch cost

`headroom wrap claude` blocks on `uvx … serena project index` (`cli/wrap.py:_index_serena_project`, 300 s cap) before Claude Code starts → anything that stalls Serena's indexer is felt as launch latency, and only in the repo that holds it.

- Budget: full cold index = ~12 s (109 files, 7 language servers), warm ~6 s. A launch stalling far past that means one file is eating an LS request timeout (`serena_config.yml` `tool_timeout` 240 − 5 = 235 s each) → `.serena/logs/indexing.txt` names the file.
- `**/*.Rmd` is excluded in `.serena/project.yml` for exactly that reason (R LS never answers `documentSymbol` for R Markdown; `.R` files are unaffected at ~4 files/s). Reach `analysis/analysis_summary.Rmd` by `Read`/`rg`; Serena's symbol + search tools do not see it.
- Serena's own session start is repo-independent (~3.5 s, dominated by the bash LS) and asynchronous — it never blocks the MCP handshake.
