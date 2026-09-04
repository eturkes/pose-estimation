# Project memory

Context retained only when source, tests, technical docs, roadmap, and git do not expose it cheaply.

## Data boundary

- Patient recordings + adjacent derivatives = sensitive. `output/`, real-data calibration files/directories, and logs stay outside agent context. **`videos/3-cam/` carries standing decode clearance** — MAIN and teammates may decode it, run the pipeline over it, and write derived outputs. Chat and reports carry redacted aggregates only: never imagery, filenames, or subject identifiers.
- **Every published tree is patient-adjacent and stays unread**: `inventory/`, `sessions/`, `qualification/`, `measurements/`, `output/`, `calibration/` — rows carry `capture_id`/event ids/source paths. Quotable exceptions are the redaction-safe markers `inventory/census.json` and `qualification/qualification.json`, plus the whole `calibration_qc/` tree, which is redaction-safe by contract. **No tool gate enforces this**: `permissions.deny` is Bash-only, so nothing refuses a path-keyed read, and `.gitignore` only keeps the bytes uncommitted. Reach a number through a bounded query (`rg -c`, `jq`, a counting script), never through a full read.
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
  patient data: probe it with Python `pathlib` and report counts alone — a directory listing puts
  subject directory names and clip filenames straight into context.
- **Publication replaces a whole tree, so the output is a destructive path.** `--out` must overlap neither `--corpus` nor `--inventory` in either direction; a symlinked `--out` publishes to the path it resolves to, which keeps the link and replaces the tree it named.
- **Sweep crash debris only after the swap lands.** A kill between the two renames leaves the sole complete generation as a *retired sibling under a dead pid*, so a sweep before the swap deletes it and a failed swap then has nothing to restore. The empty-root rollback needs its own `retiring.exists()` guard too, or it raises `FileNotFoundError` over the real error.
- **Four validation predicates read as stricter than they are.** `Pattern.match` with `^…$` accepts one trailing newline, so every alphabet must use `fullmatch` (an exported pattern needs `\Z`, because consumers call `.match` themselves). `str.isdigit()` is true for superscripts, which then raise `ValueError` out of `int()`, and for other scripts' digits, which `int()` silently normalizes into a value the cell never spelled — require ASCII `[0-9]+`. A CSV with zero rows carries its schema in the header alone, so per-row column checks never run and a short header publishes an empty artifact instead of failing. `os.kill(int(pid), 0)` raises `OverflowError`, not `ValueError`, on a suffix wider than a C long.

## Upstream instruction refreshes

- `CLAUDE.md` + `.claude/commands/*.md` arrive as upstream drop-ins landing over local adaptations. After every refresh, diff each refreshed file against its prior commit and re-apply every local clause the drop-in lost; `git log --grep upstream` lists the reconciliations. **Standing local-clause set — re-apply on sight, both verified by `rg -n 'DESCOPED|agent/contracts' .claude/commands/ CLAUDE.md`:**
  1. **`session-roadmap.md:3` DESCOPED terminal status + its dispatch sentence.** Three successive refreshes have dropped it; it is the only rule keeping the DESCOPED M3 out of MODE dispatch once M2 reaches REVIEWED. `.agent/roadmap.md`'s M3 heading also declares its own terminality, so a further drop leaves the dispatch decision intact.
  2. **Acceptance-contract path = `.agent/archive/contract-m<m>u<u>.md`, never upstream's `.agent/contracts/m<m>u<u>*.md`.**
- **The `Read()` path-exclusion control is RETIRED — never restore it.** A refresh replaced the two-control read-exclusion set with `.gitignore` currency + Serena `ignored_paths`, and `.claude/settings.json` was deleted with it, because all 31 of its entries were `Read()` rules. `permissions.deny` is Bash-only now: a path-keyed `Read()` rule also gates `Bash` by static command text, so a batched command dies for naming the path it excludes, and one command whose cwd the matcher cannot resolve halts the session under `bypassPermissions`. `.agent/archive/m2u71.md` and `contract-m2u2.md` §6 still name that deny list; both are frozen unit records of a control that no longer exists, never a live surface. The path is a generated data field: `scripts/make_calibration_qc_fixtures.py` writes it into `tests/fixtures/calibration_qc_set/manifest.json`, and `tests/`, `scripts/` and `.agent/` carry 21 more references over 10 files (`rg -c --hidden 'archive/contract-'` — `.agent/` is hidden, so an unflagged sweep reports only the 5 non-dot files). `check_calibration_qc_fixtures.py` validates digests and never resolves that field, so a move that misses the generator leaves a dangling pointer no gate reports. Upstream's requirements — committed, outside the attached set, read on demand, dispatchable by MILESTONE-REVIEW — are all already met at the archive path, so the rename buys nothing and is never worth its blast radius.

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
- **The cross-view keypoint bias is a per-RECORDING property, not a global (view, keypoint) field, so
  there is nothing for a bias model to estimate.** Measured over the **full eligible population** —
  all 115 events, 178 pairs, 103 events yielding a pair. Same statistic as A10, split across events
  instead of frame blocks: per-keypoint mean signed epipolar residual vectors correlated between
  DISTINCT events sharing an ordered view pair give **r median 0.011 (n=4341)** against that corpus's
  own permutation null of **0.005** and a within-event ceiling of **0.814** (129/178 above 0.5).
  Stricter groupings do not rescue it — same device-model pair 0.010 (n=2738), same task 0.010
  (n=1071), same subject **-0.030** (n=275). Calibrated references through the real masks with
  per-event jittered rigs, 3 field draws each: shared image bias 8/32 px **0.966 / 0.763**, shared 3D
  anatomical bias (Malleson's own parameterization) 20/40/80 mm **0.941 / 0.219 / 0.626**, holding
  **>= 0.180** even at 1.2 m rig jitter; per-event bias **-0.011 / 0.005**; noise **0.016 / 0.030**.
  Shared arms 0.180-0.966, non-shared arms -0.011-0.030, corpus inside the non-shared band; the
  shared arms are not monotone in magnitude because field realization dominates at 3 draws, so quote
  the 0.180 floor rather than any single arm. **Correlating |residual| instead gives 0.150 pooled and
  0.146 within subject: the same keypoints are hard everywhere, at every grouping, while the offset
  DIRECTION is redrawn every event — and a difficulty ranking cannot be subtracted from a
  coordinate.** `scripts/probe_bias_transfer.py`.
- **Two observability caches, and they are not interchangeable.** `.scratch/calib-obs-f32/` = the
  22-event `--stratum-events 2` sample that every A09-A13 number is measured on;
  `.scratch/calib-obs-wide/` = all 115 eligible events. `--stratum-events` selects a hash-ranked
  prefix, so widening only ADDS events to the selection (22 ⊂ 115 by cache key, verified), but it
  joins the cache fingerprint (2 → `2b84d350…`, 25 → `d427f95f…`) and `load_event` rejects a
  mismatch, so widening IN PLACE re-collects everything. **Always collect a wider population into
  its own `--cache` directory, then re-verify the narrow cache still reproduces its record.**
- **`probe_calibration_bias.py` reads the cache with NO fingerprint check** — its own
  `load_event(path)` takes no fingerprint and validates nothing, unlike the observability module's
  two-argument version. Every M2.6/M2.6b number is a replay of whatever `.npz` files sat in the named
  directory, so **the directory name is the entire provenance guard**, and the stored value could not
  serve as one anyway: all 137 entries across both caches carry the single `meta.fingerprint`
  `084174f403dae02f`, which distinguishes neither population. Editing
  `probe_calibration_observability.py` at all changes `_source_digest()` and so the live fingerprint,
  leaving both caches stale for `collect` (re-collection = full sweep) while staying readable by the
  analysis probes. Point an analysis at a directory whose population you just counted.
- **Replay reproduces the M2.6b record exactly, so re-derive rather than trust a recorded number.**
  `probe_bias_transfer.py --cache .scratch/calib-obs-wide` returns 178 pairs / 103 events, signed r
  0.0108 (n=4341), within-event 0.8138 (129/178), residual median 17.038 px, magnitude 0.1499 pooled
  / 0.1462 within subject, and every synthetic arm. It streams line-delimited JSON per arm and its
  final `print` emits only the sorted key list — the numbers are the earlier lines, not the last one.
- **Never run the gate beside a decode or inference sweep.** `tests/test_r_timebase_truth.py::test_c8_08`
  drives a subprocess with a timeout; CPU contention from a concurrent collection alone is enough to
  raise `subprocess.TimeoutExpired` and fail a suite that is green when run alone.
- **A joint multi-camera solve destroys the only acceptance statistic this corpus has.** Rotation
  cycle closure is an algebraic identity of a solve that parameterizes all three poses together —
  which is why contract A02 kept A12's pairwise BA poses independent. Held-out reprojection on the
  solve's own keypoints is self-consistency and prohibited as accuracy, and cross-event transfer is
  measured absent. So a per-event bias-and-pose model could be built and never credited here. **Check
  what a parameterization does to the acceptance statistic before adopting it**, not after.
- **Test a repair's premise before building the repair.** The funded route was a joint bias-and-pose
  parameterization; its premise is a bias with fewer DoF than the data, which needs either transfer
  across events or Malleson's external anchor (calibrated cameras + optical bone transforms — never
  available retrospectively). One re-split of an existing statistic refuted the premise in one window
  and no estimator was written. **A repair route names a premise; find it, and measure that.**
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
- **A published tree is unpatchable by ANY writer, and the pipeline's own default was the violator.**
  `tree_digest` covers every entry but `generation.json` — "including a file nobody explained" — so a
  results file is as fatal as an edited manifest. `_resolve_session_output(session, None)` resolved to
  `session.directory.parent/output/<session_id>`, i.e. `sessions/output/<event_id>/`, and
  `_dispatch_sessions` never forwarded `--output-dir`, so a corpus run **destroyed the generation it
  was still reading** after its first camera. Fixed by refusing any destination overlapping a
  published tree in either direction, where published = a `generation.json` at or above the session
  directory — the marker scoping is what keeps ad-hoc session dirs on the documented default.
  **Forwarding a flag is not the fix when the DEFAULT is the hazard**; the guard is.

## Frozen contracts carry stale numbers

- **A contract's stated census can be derived under a rule a later predicate replaced — re-derive
  every census at implementation time rather than trusting the frozen text.** M2.5's P19 published
  "329 offsets / 50 not" while its own P07 mandated partial publication, whose true census is
  **355 / 24** over 379 rows. 329 is a real number naming a different population — cameras inside a
  graph-connected event — so the defect was invisible to every consistency check that treated it as a
  count. The recount cost one broadcast to four live teammates; waiting for review would have cost a
  wrong published artifact. Whenever one predicate changes which rows exist, grep the contract for
  every number quantifying over rows and recompute each one.
- **Name the population beside the count, always.** This project has now hit the same trap four
  times: two closure statistics over different event sets, two "families connected" figures,
  329-vs-355, and a window gauge recorded `at high-water` from a mid-window sample — 231K/96% against
  a true pre-compaction peak of 240K/100%. A bare count is the shape of defect that survives review.
- **A context gauge is a time series, so name which reading you mean.** `context-gauge` bare reports
  MAIN's LAST TURN and a teammate's high-water, so a MAIN number copied from it is a sample unless
  the whole transcript is scanned. High-water re-derives offline from
  `~/.claude/projects/<project>/<session>.jsonl`: max over assistant turns of
  `input + cache_creation + cache_read + output`, with an auto-compaction boundary showing as a >50K
  single-turn drop that splits the window into two regimes — report the pre-compaction peak.
- **A projection carried forward as a budget is a frozen number of the same class**, and it is worse
  than a stale census because nothing in the text marks it as derived. The 6.5 h corpus-run estimate
  was a per-call micro-benchmark extrapolation that reached the roadmap as a unit's sizing, was
  re-quoted through planning and two unit windows, and measured 4-5× low the first time anything ran
  end to end. **Record a number's provenance beside it — measured / projected / assumed — and refuse
  to size work on a projection whose measurement is affordable.** The M2.8.1 pilot cost 49 minutes
  and moved a 6.5 h plan to 26-31 h.
- **A frozen predicate is the artifact most likely to be wrong, and a diff-blind suite is what proves
  it.** `test-m2u82-2` graded M2.8.2's contract against the shipped surface with no sight of MAIN's
  diff: **11 findings, 11 contract defects, 0 driver defects.** The recurring shape is a predicate
  written before the surface existed — a scope term never defined (P10), a witness set blind to a
  file it quantifies over (P11), `never stored` asserted over a schema already published (P12), a
  shape test called an allowlist (P13), a budget figure carried unlabelled (P14). **Cost of the
  ruling is one amendment each; cost of trusting the text is a green gate over a false claim.** Budget
  a diff-blind grading pass per kernel unit and expect its findings to land in the contract.
- **A case that grades a stand-in grades nothing** (M2.8.2 A10). Two suite defects of one kind: a
  discovery regex over `[A-Z_]*DISPOSITION[A-Z_]*\s*=` could not match the repo's own annotated form
  `ASSET_DISPOSITIONS: tuple[str, ...] = (`, so it reported an absent constant that was present; and a
  local `_manifest_verdict` re-implementation passed while the shipped `validate_manifest` was broken.
  **Encode a predicate over shipped behaviour against the shipped symbol.** Finding a *name* is also
  only a spelling test — a comment passes it.
- **Two suite idioms that pay for themselves, adopt by default.** (1) *Re-derive, never transcribe*:
  R's reason codes / schema header / directory filter / 3D marker, the pilot's key pattern + stratum
  axes + allowlist call site, the contract's own superseded figures — all parsed from source at check
  time, so a rename moves the case instead of silently passing it. (2) *Every quantified predicate
  carries an explicit `nonvacuous` clause*: five M2.8.2 predicates go green over an empty set — no
  landmark CSV, an empty tree, zero recorded frames, a report with no strings, an unsized unit.

## Multi-agent traps

- **A superseded ruling in a shared file poisons every downstream artifact.** M2.2's rulings file kept A07 and A12 after two later rulings replaced them; the test teammate conformed to the file, and its suite read as a regression against correct code. Amend a ruling in place, in the same turn as the code it rules — a "later rulings win" note at the bottom is not enough, because the table is what gets read.
- **Keep a spike's worktree after picking a design.** M2.2's rejected spike carried an independent 184-line generator; rerunning it against the shipped one made it a differential oracle — identical partition over 193 blocks and 379 assets, 0 `capture_id` disagreements, from code sharing no line with the winner. Keeping the bytes costs nothing and the evidence is unreproducible once the worktree is gone.
- **A reviewer's worktree diff against `main` carries its own staleness.** `wt/rev-m2u2` showed a *revert* of a MAIN change that landed after its branch point. Read a reviewer's diff as findings, never as a patch to apply.
- **A saturated reviewer's persistence is a context artifact, not evidence.** `rev-m2u3` re-filed a strengthened red against one closed ruling twice while above 97%, each time reading the ruling as unresponsive rather than as decided. Weigh a finding by its argument; when the same row returns from an agent near its ceiling, restate the ruling once and stop, and prefer a fresh successor to another round with the saturated instance. The inverse trap is real too — see the next bullet — so the discriminator is the reviewer's gauge, not its insistence.
- **Copy a reviewer's red suite into the primary tree and rerun it; the pass/fail split is the adjudication ledger.** M2.7.1 window 4 scored `rev-m2u71` + `rev3-m2u71` at **41 failed / 4 passed** before fixes and **7 failed / 38 passed** after, where the 7 are exactly the rows ruled rejected. The rerun is what caught the half-fixes: window 3 recorded C01/C11 as closed after correcting the module docstring, and the red stayed red because the roadmap headline still carried the unscoped claim. Reds scoring against an older branch point separate into fixed / open-scope / live only under execution — a report's own tally cannot make that split.
- **A reviewer's DONE marker is not the end of its yield.** M2.2's two reviewers sent nine more accepted defects after both phase-2 markers, four of them in the minutes before the close commit. While a reviewer has context left, keep it pointed at the surfaces MAIN just changed and take findings by message; ask for the report file last, since writing it costs the context that finds the next defect.

## Committed fixtures — the gitignore trap that eats a whole deliverable

- **Four `.gitignore` rows are slash-free component names and therefore match at ANY depth**:
  `inventory:60`, `sessions:78`, `qualification:85`, `calibration_qc:94`. A committed test fixture
  placed under one of those names — `tests/fixtures/calibration_qc/`, or an inner
  `inputs/qualification/` — is silently uncommittable, and `git add` reports success while
  committing nothing. **Verify every intended fixture path with `git check-ignore -v` BEFORE
  building the tree**, root and inner directories alike; the failure is invisible at every later
  step. Marker FILES are safe by construction: `qualification.json` is not a component named
  `qualification`, and the `*.*/` siblings carry a trailing slash so they match directories alone.
  M2.7.2 layout that clears all four: root `tests/fixtures/calibration_qc_set/`, upstream tree
  `inputs/upstream/`, golden `expected/published/`.
- **A scan's own documentation must not spell the needles it forbids.** M2.7.2's privacy scan failed
  on the fixture README, which quoted the forbidden path text as documentation. Same shape as
  M2.7.1 A05's ruling that `PROHIBITED_PARAPHRASES` stays module-side and never publishes — **a
  document inside the scanned set carries every string it quotes.** Ruled identically both times:
  keep the needle list in the checker, describe the rule without spelling it, and the scan stays
  total over every byte instead of buying itself an exemption. An exclusion list is the wrong fix —
  it opens a hole exactly where data could hide.
- **A deliverable-first seed leaves its suite wiring FAIL on purpose.** M2.7.2's P10 (the checker is
  called from `tests/`) is red at seed, because a case asserting `rc == 0` would be red until the
  fixtures land and a skip guard breaks the zero-skip invariant C8.08 reconciles. The test lands in
  the same commit as the artifact it grades. Seeding a passing stub instead would encode the defect.
- **A validator graded only against an absent artifact is graded on its error paths.** M2.7.2's
  checker called `shutil.copytree(inputs, dest)` onto a live `TemporaryDirectory` — missing
  `dirs_exist_ok=True` — and its three replay predicates were already FAIL for missing fixtures, so
  the `FileExistsError` stayed invisible until a teammate's first live replay. **Grade a new
  validator against a hand-made minimal POSITIVE before funding production behind it**; an
  all-`unknown` seed proves only that the failure path prints.
- **The `calibration_qc` fixture set re-derives in seconds and is that publisher's byte oracle.**
  `python scripts/make_calibration_qc_fixtures.py --force` then
  `python scripts/check_calibration_qc_fixtures.py` = 12/12 in 1.9 s, both under the standard
  `env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync` prefix. `inputs/upstream/` is one
  `qualify.run()` generation over `_canonical(90, "above")` — subject 90 keeps the capture inside the
  scan's synthetic namespace, and `calibration_qc.run` validates that tree standalone, so no
  registry, session tree or media is committed. Refusal matrix = **26 file-only reasons** (17 run,
  9 validate) + **4 state-only**: `claim_missing`, `corpus_cardinality`, `tree_unreadable`, and
  `output_overlap` (overlap needs a symlink the fixture set forbids). `claim_prohibited` IS
  file-only, reached through an arm label carrying a prohibited paraphrase.
- **A generator whose `--force` deletes its destination needs the publishers' ownership rule.**
  `make_calibration_qc_fixtures.py` refuses any non-empty destination whose `manifest.json` does not
  name it, and still accepts an empty directory, which is how the idempotence predicate drives it.

## Claim set — one constant, two pinned prose copies

- **`calibration_qc.CLAIMS` is the single source of truth for all 15 supported statements**, and two
  documents quote it verbatim: `docs/technical/calibration_qc.md` (agent register, C01-C15 under
  *Claim boundary*) and `docs/calibration_finding.md` (human register, the shipped report).
  `scripts/check_claim_report.py` P01/P02 pin both. Never reword a claim in either document — the
  publisher checks published bytes against the constant, so a reworded copy hands an editor text
  `_assert_claim_conformance` refuses. Measured at M2.7.3 entry: the technical copy had already
  drifted on **6 of 15** rows (C03, C05, C06, C09, C12, C15) with nothing referencing the document.
- **The report never spells a prohibited paraphrase; it states each refused overreach by shape.**
  Measured: **no claim contains any of the 23 `PROHIBITED_PARAPHRASES` entries under `_fold`**, so
  quoting all 15 claims verbatim and carrying zero needles are simultaneously satisfiable. That is
  what keeps P03 total over both documents with no excluded span — the same ruling as M2.7.2's
  fixture README, reached by measurement instead of by analogy. `_fold` flattens `_` and `-` alike,
  so a hyphenated respelling of the unrun arm followed by an outcome word is caught; an intervening
  word is not, and that adjacency gap is the constant's, not the checker's.
- **`docs/calibration_finding.md` is the sole published home of the `calibration_bias` numbers.**
  The publisher cites and digests that probe and ingests nothing from it, so `evidence_qc.csv`
  carries `bias_transfer` rows alone. Quote C01-C04's closure, control, bundle-adjustment and subset
  figures from the report, never from a published cell — there is none.
- **A green predicate whose detail line reports zero items is a failing predicate.** M2.7.3's P07
  passed as `0 named repo paths all resolve` because the report used no backticked repo path. Every
  set-quantified predicate needs a non-empty floor; the row-wise check over a zero-row CSV is the
  same defect. Read a validator's detail line, never its rc alone.
- The report re-derives nothing: `python scripts/check_claim_report.py` is 9 predicates in under a
  second, driven identically from `tests/test_claim_report.py` through `runpy`.

## Prospective capture specification — two capture documents, and which governs which

- **`docs/prospective_capture.md` governs a FUTURE calibrated acquisition; `docs/capture_protocol.md`
  governs a capture made with the tooling this repo ships.** Neither amends the other, and both carry
  the banner `Which document governs which capture.` naming the other. The split is forced: the
  specification requires 8 cameras, 120 fps, global shutter, a wired trigger, traceable scale and
  sealed held-out targets, none of which `pose-estimation-calibrate` supports. Merging them would
  make a shipped operational document state requirements its own commands cannot meet.
- **`python scripts/check_prospective_capture.py` = 13 predicates in under a second**, driven
  identically from `tests/test_prospective_capture.py` through `runpy`. It owns the section spine
  (20 frozen ids + titles), the five per-section fields, the `MUST` binding of the five
  non-negotiables, the local-decision labelling of the five measured absences, citation
  resolvability, and an ASD-STE100 25-word sentence bound. Negative-control runner =
  `.scratch/nc_m2u74.py` (9/9 fire, mutates the doc in place and restores under `try/finally`);
  scratch-local, so re-run it rather than trusting the recorded result.
- **Imperative prose states an obligation to a human and none to a checker.** S11 shipped written
  entirely in imperatives — the register's preferred form — and carried no `MUST` while a
  non-negotiable bound it. A reading pass cannot see this; P03 can. Where a section is normatively
  load-bearing, definite modality has to be literal.
- **A needle list is scoped to the surface it was written for.** `PROHIBITED_PARAPHRASES` belongs to
  `calibration_qc`'s claim surface. Ranged over `capture_protocol.md` it fires on
  *"This clinical-validity gap stays open"* — `_fold` flattens `-`, so the concession of a gap reads
  as the overreach the needle exists to catch. Scope the scan; never reword a DONE unit's shipped
  document to satisfy a rule it was not written to.
- **NIST traceability = an unbroken chain to a STATED reference with documented uncertainty at every
  link — TN 2156 §5.3.3 allows SI or another specified reference.** Requiring the SI metre is this
  project's choice, not NIST's rule, and §5.2.4 says a certificate number alone does not prove
  traceability. Two further over-attributions corrected at the same time: Anipose's synchronization
  routes are in its FAQ rather than the paper, and BioCV publishes zone statistics rather than a
  spatial error map.
- **Verification is 16/20 rows.** L1's absence is `confirmed`; L2-L5 (reprojection threshold,
  board-to-volume ratio, Japanese retention period, published millisecond sync figures) are
  unverified. They fail conservatively — each is already labelled a local decision, so a standard
  that does exist means the document understates its authority. Re-run as a MILESTONE-REVIEW `audit`
  row. Report at `.scratch/agents/res-m2u74-2.md`.

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
- Pose models are NPU-safe: RTMW-L NPU vs CPU = 0.505 px mean / 2.265 px p95 / 5.3 px p99 keypoint deviation, score MAE 0.00056. Per-call 7.17 ms NPU vs 134.26 ms CPU (~19×). Detector 109.82 ms NPU (garbage) vs 445.21 ms CPU.
- **Those per-call numbers do not predict run throughput — measured end-to-end they are 4-5× optimistic.** The projection they support (pose 7.17 + det 445/7 ≈ 70 ms/frame ⇒ 6.5 h for the corpus) survived planning and two unit windows unchallenged. The M2.8.1 pilot ran the shipped configuration (`rtmw-l`, `hands-arms`, `--single-subject`, det CPU / pose NPU, `--det-frequency 7`) over 8971 frames: **2959.56 s = 3.03 fps** including per-event process start, 3.60 fps from mean latency ⇒ **26.0-30.9 h** for 337 090 frames. **Never size a run from per-call latency; run a stratified pilot and multiply.**
- **The ~40× per-asset bimodality is CLOSED: it was the `PoseTracker` freeze, not a cost profile.** Both M2.8.1 candidates — per-detected-box cost and detector device — are refuted; see *rtmlib `PoseTracker`* above for the mechanism and the fix. It was answered by re-reading M2.8.1's own committed per-frame latency logs at the detector cadence, which separates detector frames from pose-only frames: pre-fix `run-00` read det 342.1 / non-det 356.6 ms (frozen at residue 0), post-fix it reads det 313.4 / non-det 6.9 ms, a clean 1-in-7 cadence. **Before funding a measurement, re-read the logs the last measurement already wrote.**
- `--det-device GPU` stays unqualified and is deferred, not adopted: the synthetic probe reads GPU 9.7 ms vs CPU 213 ms median and GPU zero-fills its padded rows (unlike NPU), but a dynamic-output model needs its per-device shape+range qualification before use, and a corpus run is the wrong place to run one.
- MediaPipe is unaffected — SSD anchors + NMS decode in Python (`detection.py`), graphs stay static-shaped ⇒ both roles default NPU. `models.DETECTOR_MODELS` selects which compile on `--det-device`.
- **The padding is a device property, and only NPU pads with garbage.** One synthetic probe separates the three devices with no patient data: read `yolox_m_8xb8-300e_humanart-c2c7a14a.onnx` (rtmlib cache), compile per device, infer `np.zeros((1,3,640,640))`, print `dets`/`labels` shape + range. CPU keeps the dynamic output (`(1,1,5)`); GPU and NPU both materialise a fixed 100-row buffer with `labels` all `-1`; only NPU's `dets` hold uninitialised memory (`-0.1471…1.0215` on an all-zero image, so scores clear any threshold), while GPU's read exactly `0.0000`. That makes GPU a live candidate for the detector and keeps NPU excluded. Median latency on that probe: GPU 9.7 ms, NPU 108.4 ms, CPU 213.1 ms.
- Reproduce the real-frame findings: `.scratch/det_npu_vs_cpu.py`, `.scratch/pose_npu_vs_cpu.py`, `.scratch/device_timing.py` (scalars only, no imagery/identifiers) — these decode patient clips, so they need clearance; the synthetic probe above does not.

## Corpus run — the pilot instrument, and how far diagnostics reach

- **`scripts/pilot_corpus_run.py` is the standing instrument for any claim about the real corpus.** It is the only path that measures the run against the published tree, which no test case can reach. Rerun: `source /var/home/eturkes/.local/app/intel-accel/env.sh` then `PYTHONPATH="$PWD/src:$PYTHONPATH" .venv/bin/python scripts/pilot_corpus_run.py`; `--reuse-run` re-analyses an existing `--out` tree without decoding. Outputs are patient-adjacent ⇒ `.scratch/pilot-m2u81/` (gitignored), report at `pilot_report.json`.
- **It samples EVENTS, not assets** — `process_session` runs every camera of a session, so an asset cannot be drawn alone. Axis coverage first (hash-ranked event per uncovered value of `codec`, `device_config`, `rotation_deg`), then hash-ranked replication to `--min-assets`; the rule is length-free, so the sample carries no duration bias. `_select_events` raises rather than running an uncovered sample. `pts_monotonic` is reported, never required — it is confounded with codec and device on this corpus (hevc/one tablet = 0 on all 123, h264/another = 1 on all 256), so demanding it as an axis over-constrains the draw.
- **Redaction in any corpus-touching report = membership in BOTH placements, never a shape test** (M2.8.2 A09). Value admissible ⟺ published label (stratum label | R reason code | disposition code) ∪ code-authored constant the emitting program spells at its own call site. Key admissible ⟺ frozen report field name ∪ published label. A key pattern like `[a-z][a-z0-9_]*` is a **denylist wearing an allowlist's name**: it admits every identifier of that shape, so one capture id is refused as a value and admitted as a key — same string, two verdicts. It also refuses shapes the reports really publish (`_coverage` keys by integer stratum value) and has no class for the generator, its version or the device echoes. `_assert_redacted` keeps the weaker allowlisted-OR-matching guard as a runtime backstop; the suite is the membership oracle. Subprocess stdout carries identifiers (`Session 's02-…'`, camera names, paths) ⇒ redirect to log files under `--out/logs/`, never echo.
- **`scripts/corpus_run_2d.py` = the full-corpus driver; the pilot stays its reader.** It loads the pilot by `importlib` for the three-table join and the redaction allowlist, so the corpus claim and the pilot claim share one instrument. Resume is keyed on `event_complete.json`, written after an event's outputs are final — **never on output presence**, because a killed run leaves a partial CSV no row count can distinguish from a complete one (the true count is unknown until the source is fully decoded). A re-attempt `rmtree`s the event tree first: partial work is destroyed, never credited. `--analyse-only` republishes manifest + report with no decode; `--limit N` runs the first N due events; exit 1 ⟺ any verdict false, which includes an unfinished run.
- **`run_manifest.csv` is total by construction: one row per canonical registry asset, six frozen codes** (`pose_estimation.corpus_run.ASSET_DISPOSITIONS`) — `ok`, `not_placed`, `not_run`, `run_failed`, `clinical_failed`, `no_landmarks`. `not_run` was **forced by the partition, not chosen**: a partial pass must still publish a total manifest, and folding "never attempted" into `run_failed` hides real failures behind resumability (same shape as M2.8.1's `no_windows_emitted`). `validate_manifest` enforces row count, key-set equality and key **uniqueness** as three separate conjuncts, refusing emptiness first — a manifest that duplicates one asset and drops another keeps both the count and the set, so uniqueness is the only clause that sees it.
- **The event is the isolation grain, and it always publishes.** R is invoked once per event directory and its `stop()` ends that process, so one rejected asset voids that event's whole clinical pass. The driver does not recover it: `_attempt_event` reads the exit code and `corpus_run.asset_disposition` lands **every** asset of that event on `clinical_failed`. The loss is recorded rather than silent, and the neighbouring event is untouched.
- **Judgment-bearing decisions live in `src/`, never in a driver script** (M2.8.2 A12): the suite cannot import `scripts/`, and a gate backing a durable claim must exercise the shipped decision. `sessions.generation_digest` (marker-bytes witness) and `corpus_run.asset_disposition` (+ `STAGE_RUN`/`STAGE_CLINICAL`) both moved out of the driver for exactly that reason. The driver keeps orchestration alone.
- **Corpus run, measured whole: 8.702 h** (run wall 30 666.53 s + clinical 660.90 s), 193 events / 379 assets / **337 090 decoded frames**, **10.99 fps incl. startup**, manifest **379/379 `ok`**, CFR pooled fallback **0.0**, partition 379 = 379 + 0 disjoint, 21 483 window rows, all 11 verdicts true. The post-fix pilot projected 7.07-7.61 h from 16 assets → **14% low**: stratification covers axes, not per-frame cost. Size any successor against 8.70 h.
- **A per-invocation accumulator paired with a whole-corpus numerator is a false rate the moment the job is resumable.** The driver divided all-corpus `frames_decoded` by the seconds THIS process spent, so the final resumed pass published 13.88 fps / 6.903 h — a rate the pipeline never reached. Fix: sum the wall from the per-event markers (`run_s`/`clinical_s`), which is correct across any pass count, and label `sample: corpus` only when `events_measured == events_total`. **Check every published rate for a numerator and denominator drawn from different populations.**
- **A read-only claim over the published session tree needs THREE witnesses.** `tree_digest` excludes `generation.json` (a document cannot digest itself) and `validate_generation` compares that document's *fields* — so reindenting the marker rewrites bytes inside the tree while both stay green. `sessions.generation_digest` is the third, and the run publishes `generation_marker_unmoved` beside `generation_digest_unmoved`.
- **A healthy sample cannot exercise a failure path.** The pilot's 16 input groups partitioned 16 windowed / 0 dropped, so every drop reason went untested there; `2d_drop`'s golden is what pins them. Pair every real-corpus probe with a synthetic negative — the probe proves reach, the golden proves the branch.
- **Diagnostics ship only through the session path.** `process_source` writes its source-summary CSV from `output_diag`, and the single-source and batch entry points pass none, so a run outside `process_session` produces no diagnostics at all — not an empty file, no file. Any claim of the form "every asset has a disposition" holds for the session path alone.

## rtmlib `PoseTracker` — stateful, unsound, and DISABLED in the run path

- **`run.py` constructs it with `tracking=False` (M2.8.2 D01). Never restore `tracking=True`.** The
  IoU branch reorders the CURRENT frame's keypoints by PERSISTENT track id —
  `keypoints = np.array([keypoints[i] for i in self.track_ids_last_frame])` — while `track_by_iou`
  mints `track_id = next_id++` for any unmatched box above `MIN_AREA = 1000`. One missed match
  therefore indexes a one-person array at `[1]`, raises `IndexError`, and hits a bare `except` that
  returns **before `frame_cnt += 1` and before `bboxes_last_frame` is replaced**. Both freeze for the
  rest of the source, permanently, and the pre-reorder keypoints still return, so yield stays ~0.99
  and nothing downstream looks broken.
- **The residue of the frozen counter picks which failure you get, and one of them is silent data
  corruption.** At `det_frequency = 7`: residue 0 → the detector re-runs on every frame (correct
  output, ~6× cost); residue ≠ 0 → the detector never runs again, `track_by_iou`'s pops drain
  `bboxes_last_frame` to empty, and `RTMPose.__call__` opens with
  `if len(bboxes) == 0: bboxes = [[0, 0, w, h]]` — so a top-down pose model estimates from the
  **whole 1080p frame** instead of a person crop, at confident-looking scores.
- **This is what M2.8.1's 40× bimodality was**, and both of that unit's candidate causes are refuted:
  not per-detected-box cost, not device placement. Measured bands 7.2-12.2 and 338.6-543.5 ms/frame;
  synthetic repro over 140 frames returns frozen-at-3 → 1 detector call / 135 whole-frame pose calls
  / 10.5 ms, frozen-at-7 → 134 detector calls / 343.0 ms, and `tracking=False` → 20 calls / 0 /
  58.0 ms on both stimuli. `scripts/probe_tracker_freeze.py`, 7 verdicts, rc=0, no corpus needed.
- **The fix is a removal, not a patch, because the tracker's output was already redundant.**
  `KeypointSmoother` owns temporal association through Hungarian `gated_assignment`
  (`src/pose_estimation/smoothing.py`); rtmlib's tracker contributed only the IoU drop of unmatched
  people, which `--single-subject` overrides by taking the confidence argmax.
- `det_frequency=1` still routes `tracking=False` down the IoU branch (the stateless guard is
  `not self.tracking and self.det_frequency != 1`), but a freeze is harmless there: `frame_cnt % 1`
  is 0 at every residue, so the detector runs every frame and the box list is never starved.
- **Any pre-M2.8.2 output from `run.py` was produced under the freeze** and is suspect per asset, not
  per run — including `output/rtmw-l_body_single/`. M2.3's `detect_rate` is unaffected: it drives
  `det_model` + `pose_model` per frame and never instantiates the tracker.
- Fatal for **sampled** frames: seconds-apart samples share no IoU, so the list empties permanently and any count taken from it reads 0. M2.3's detectability run published `detect_rate` median 0.0 over 379 assets from exactly this, while the detector saw a subject in 24/24 probe frames. One tracker instance reused across assets compounds it.
- Sampled-frame analysis must drive `det_model` + `pose_model` per frame and take counts from the detector return. Reserve `PoseTracker` for consecutive video (`run.py:758`, rtmlib's intended mode; note its default `tracking=True` still drops a person whose frame-to-frame IoU falls under 0.3).
- Defaults worth knowing: `det_frequency=1`, `tracking=True`, `tracking_thr=0.3`, `backend='onnxruntime'`, `device='cpu'`.

## Devices available in-container

`Core().available_devices` = `['CPU','GPU','NPU']` — Core Ultra 7 268V, Arc 140V iGPU, AI Boost NPU; `intel-accel/selftest.py` reports `OK correct=True` for all three under OpenVINO 2026.3. The two historical GPU blockers are both cleared by the `intel-accel` farm: it links the host `libstdc++` for the driver's `GLIBCXX_3.4.35` requirement, and it pins an IGC matched to the host GPU driver, which removes the compute-runtime abort at `command_stream_receiver.cpp:1205`. An IGC that falls behind the host driver reinstates that abort and kills CPU and NPU work in the same process → after a host Intel driver update, rebuild the farm and re-run the self-test.

## R analysis layer — non-obvious hazards

- **Retention rule, standing (user).** M2's numerical output is eventually destined for the `../rehab` dashboard over the hospital SCI database. Any artifact serving that path is kept whatever its milestone's status, DESCOPED included, and is never traded away for tree tidiness. Anything off that path is deleted outright rather than tagged or archived — a superseded artifact costs a future agent a wrong start. The decisive question is which of the two an artifact is, never how old it is. **Operational test for a retained branch or oracle = one named open dependency.** `wt/spike-m2u3-audio` holds because R6's connectivity-reconciliation polish row consumes its `_family_coverage` as the P38 oracle; `wt/test-m3u3` holds because its remaining diff-blind cases are exactly the five cut 3D clinical groups on the `../rehab` path. Everything else went, and each deletion was decided on content rather than label: `spike-m3u3-long`/`-wide` implemented the losing side of frozen carrier decisions plus two specifically forbidden changes (R-10, R-11); `spike-m2u2-a` was a design not picked, confined to gitignored `.scratch/`; `spike-m2u2-b`'s oracle role ended when M2.2 closed, its one unique verdict already recorded in `rulings-m2u2.md` A08. Prove no-dependency by sweeping `.agent/ scripts/ tests/ src/ docs/` with a positive control before deleting — an empty search and a broken search print the same bytes. **A retained branch survives only while a ref points at it.** The Close order sweeps `wt/` branches wholesale, which takes the retained exceptions with the rest and leaves their commits dangling — reachable by `git fsck` but pruned by `git gc` once unreachable objects age past `gc.pruneExpire` (2 weeks default). Both named branches were found deleted this way and re-anchored: `wt/spike-m2u3-audio` = `049684a` (carries `scripts/spike_audio_offsets.py` `_family_coverage`, the P38 oracle), `wt/test-m3u3` = `7026e2c`. Verify with `git branch --list 'wt/*'` before trusting any retention claim, and recover a missing one by matching `git fsck --no-progress | grep 'dangling commit'` against subjects — the tip is the dangling commit in that line with no dangling descendant.

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
- **`compute_window_features` dropped a whole `(video, person_idx)` group at FIVE sites, not three, and published QC for none of them in 2D.** The five: `n < 4`, non-finite/≤0 `fs`, non-finite `t_start`/`t_end`, span `< window_sec`, empty `win_starts`. Window QC was 3D-only at both emission and write, so a 2D run lost a person with zero rows and zero record — which moves a cohort `n_subjects` with nothing saying so. Now every group reaches exactly one outcome: window rows, or one row in `<stem>_clinical[_3d]_group_qc.csv`, published in **both** modes and always written, empty or not, so a reader can tell "nothing was dropped" from "the step never ran". `GROUP_QC_REASONS` freezes six codes; `group_qc_row` refuses an unlisted one.
- **The partition needed a sixth code the drop sites do not supply.** A group passing all five entry guards whose every candidate window fails the `sum(win_mask) < 4` floor emits nothing. The floor stays a window-level rule; `no_windows_emitted` is recorded after the loop. **When an invariant reads "every input reaches exactly one outcome", enumerate the paths that produce NOTHING, not just the ones spelling `next`** — the silent case is the loop that ran and yielded no row.
- **A window-keyed QC table cannot carry a group-level fact, and the 3D-only guard refuses it twice.** A group dropped before any window exists has no window key, and the `is_3d` guard exists because a numeric column on a 2D output enters `aggregate_per_video()` as a feature unnoticed. A separate file escapes both: consumer discovery selects positively (`_clinical.csv` / `_clinical_3d.csv` suffix; `_clinical_windows\.csv$` glob), so a new suffix is invisible to it.
- **`_expected_outputs` in `scripts/regenerate_r_clinical_goldens.py` is the golden whitelist.** An artifact absent from it never reaches the golden directory, and the staging dir is deleted, so the omission is silent. **Six places enumerate the artifact set and the producer names none of them**: that tuple, `_DATASETS` + `_BASE_WIDTHS` in `tests/test_r_clinical_goldens.py`, `_EXPECTED_GOLDENS` in `tests/test_r_timebase_truth.py`, the sorted `produced ==` list in `tests/test_r_pipeline.py::test_world3d_outputs_not_rescanned`, and — whenever a `src/` module moves with it — the `source_digests` in `tests/qualify_determinism_results.json` + `tests/calibration_qc_determinism_results.json`. Wiring one new artifact moved all six. Budget a published artifact by its enumerators, not by its writer.
- **A golden built only from well-formed input pins nothing about the failure path it exists for.** The group-disposition goldens are header-only on the three healthy 2D datasets and on `world3d`, so `2d_drop` was added with groups `((0,91),(1,3),(2,20))` — one healthy plus one per short-input drop reason, truncating the healthy trajectory so the drop reason is the only difference. Its golden carries 2 rows over `too_few_frames` + `shorter_than_window`, and the three `2d_drop` goldens together pin D05 on committed bytes: 3 groups = 1 windowed + 2 dropped, disjoint. Existing goldens stayed byte-identical because the healthy datasets keep the default single group. Same lesson as M2.7.3's vacuous P07 and the zero-row CSV: **a set-quantified golden needs a populated member.**
- **The group-disposition artifact carries one schema in both modes and takes no 3D identity tags**, unlike `world3d_clinical_3d_window_qc.csv`. Deliberate: D04's reason for a separate file is that one artifact explains dropped groups in either mode, and tagging only the 3D copy rebuilds the per-mode split D04 refused. `test_clinical_golden_schema_exact` exempts `kind == "group_qc"` from both the tag assertion and the non-empty row floor.
- **Freezing a value set is not freezing the header.** `GROUP_QC_REASONS` froze six reason codes and left the column names unstated, so a diff-blind suite encoded `reason` against a shipped `drop_reason` + `qc_status` and three predicates failed on a correct artifact. Header is now `video, person_idx, n_frames, drop_reason, qc_status` — `drop_reason` because `qc_reason` already names the per-metric window verdict, `qc_status` because a disposition row is a verdict. A predicate quantifying over cells needs the header frozen beside the values.
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

- **`../rehab` = M2's single planned consumer, and its conventions bind any artifact this repo
  publishes for it.** Separate git repo, dashboard over the hospital SCI database. Sole raw source
  `data/raw/ALL_SCIDATA.csv` (**cp932**, Japanese headers). `schema/columns.yaml` = **67 direct
  descriptors + 152 family-expanded = 219 `ColumnSpec`**; observed key union
  `{raw, ja, en, group, role, dtype, unit, range, levels}`, six required. **`short_ja`/`short_en`/
  `description` are a comment-only aspiration — ZERO literal descriptors carry them**, and short
  labels are absent from `ColumnSpec` and from dashboard code, so dense charts reuse the full labels.
  Groups `demographics id injury isncsci isncsci_motor isncsci_sensory meta scim`; roles
  `feature id meta outcome`; dtypes `categorical datetime numeric ordinal`.
  `missing_sentinels: ["_","","NA","NT","ND"]`. **Bilingual ja/en is a requirement** —
  `ui_strings.yaml` holds 395 keys, every one exactly `ja`+`en`, 0 missing arms; `ui_str()` resolves
  lang → `ja` → `en` → key; the dashboard defaults to `ja`.
- **Appending a group is cheap; making it a FEATURE is not.** `load_schema()` enforces required keys
  by `KeyError` alone — no unique-raw check, no enum check — and duplicate raw names silently keep
  the last, so a new `group: pose` enters automatically. But schema rows alone do nothing:
  `ADMISSION_FEATURES`/`NUMERIC_FEATURES`/`CATEGORICAL_FEATURES` are hard-coded at
  `../rehab/src/rehab_sci/data/dataset.py:40-107`. **Deliver an append-ready `columns.yaml` fragment**;
  a separate `schema/pose_features.yaml` is worse, forcing a `schema.py` load/merge change. Fonts ship
  as **10 SUBSET WOFF2 faces** (`scripts/02_build_fonts.py`), so new Japanese glyphs plausibly force a
  consumer-side font rebuild. `data/processed/` is gitignored and pyarrow is installed → Parquet is
  natural, CSV acceptable. Grain = **episode × assessment occasion**: 1200 × 26 = 31 200 raw rows,
  893 × 26 = 23 218 clean; `TIMES` = ordinal 1-26, `TIME_Name` = 26 levels `0day 72h … 10y discharge`.
- **The pose↔rehab join is CUT by user ruling — M2.8 publishes cohort aggregates only.** No
  per-subject rows, no patient identifier, no join column, no join template. The ruling SHRINKS the
  consumer delta rather than only making it safer: full ingest was priced at one untracked table plus
  six tracked files, whose three heaviest items are the authorized identifier map, a join-validating
  loader and the dataset merge — a cohort artifact carries no `IDNumber`/`TIMES`, so all three vanish.
  Do not re-plan a join surface; `analysis/make_templates.R`'s operator-filled `sessions.csv` pattern
  is not extended.
- **Cohort aggregation is well-conditioned at `(task, side)`**: 12 cells over `cap coin glass key nut
  peg` × `l r`, each **15-16 distinct subjects**, one family per subject per cell, 188 = 12×16 − 4
  absent. **Zero cells below 5**, so no small-cell suppression at that grain; a finer grain (adding
  `view`) shrinks cells and reopens it.
- **Container PTS reordering cannot reach the R layer.** `pts_monotonic = 0` on 123/379 assets, but
  `SourceClock.timestamp()` (`src/pose_estimation/video_io.py:28-77`) guarantees strictly-increasing
  timestamps — a regressing or repeated `cv2.CAP_PROP_POS_MSEC` falls back to `idx/fps`, then a second
  guard forces `last + 1/fps`. 123 is an upper bound: qualification measured PyAV demux order, the run
  path reads cv2 presentation order. **The CFR fallback rate is now measured and it is zero** —
  `index_fallback` 0 + `monotonic_forced` 0 over 8971 decoded frames on 16/16 pilot assets, 4 of them
  drawn from the 123-asset `pts_monotonic = 0` population. Counters ship in `SourceTimestampClock`, so
  any future run re-measures it per source; treat a nonzero rate as a real finding, not as noise.
- **The 2D clinical feature path already exists and is golden-pinned**, so a 2D delivery unit is
  delivery + schema work, never feature development: `analysis/clinical_features.R` with six goldens
  (`2d_csv4dp_*`, `2d_cumsum_*`, `2d_idx_*`, each plus `_windows`). M2.4's `nominal_fs()` adoption
  already corrected the cadence underneath every rate-based feature.
