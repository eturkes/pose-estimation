# Contract — M2.7.1 · `calibration_qc/` publisher (F1a)

Frozen acceptance contract. Predicates bind; every downstream artifact decides against them.
Baseline `2eed63d`. Tier `kernel`. Analog `qualify` (`src/pose_estimation/qualify.py`).

## §1 Scope

F1a = the **fifth publisher**. It publishes the corpus-level ruling that extrinsic-calibration
recovery is not achievable on this corpus, **beside** `qualification/`, **never patching it**.

In scope: contract + validator, canonical tree, integrity + consumer boundary, ownership + atomic
publication, CLI, docs, red suite, determinism + mutation campaigns.

Out of scope, explicitly:
- **Any estimator.** F1a computes no statistic. Evidence arrives as input, is validated, is
  published. A tool that recomputes a probe number owns that number's correctness and F1a must not.
- **Extraction of `src/pose_estimation/publication.py`** (polish pri 3/M). Refused for this unit:
  it touches three shipped publishers whose mutation gates pin them, its acceptance bar demands all
  three published trees stay byte-identical, and it sits behind the pri-1 row recording that
  `scripts/run_inventory_mutations.py` cannot replay from committed state. F1a is the fifth copy of
  the idiom by ruling, not by oversight.
- **Any per-event, per-asset, per-capture or per-subject verdict.** D04 makes that unrepresentable.
- `qualification/` bytes. D03.

## §2 Design decisions, ruled before dispatch

| id | decision | why |
| -- | -------- | --- |
| D01 | Tree = `calibration_qc/`; module `src/pose_estimation/calibration_qc.py`; CLI `pose-estimation-calibration-qc`; marker `calibration_qc.json`. **User-confirmed.** | The token `calibration` is taken four times — `calibration.py` (`CameraCalibration`), `calibration_cli.py`, CLI `pose-estimation-calibrate`, `docs/technical/calibration.md` — and `.gitignore:89` `calibration/` already swallows a repo-root tree of that name under a rig-geometry comment, so the roadmap's `calibration/` would be ignored for a stated reason that is false. `calibration_qc` parallels the shipped `*_qc.csv` vocabulary and collides with nothing. |
| D02 | The generated tree is **gitignored**, two-pattern shape `calibration_qc` + `calibration_qc.*/` matching `sessions`/`qualification`. **User-confirmed.** | Matches all four siblings. The negative reaches humans through M2.7.3's committed report and the two committed probes; M2.7.2's committed de-identified fixtures are the tree's regression oracle. Committing the tree would make "regenerates byte-identical to the committed copy" a gate, which is the regeneration-fixpoint trap `.agent/memory.md` records. |
| D03 | **`qualification/` is read-only to F1a.** No patch, no republish, no new column, no sentinel rewrite. | `sessions.validate_generation` hashes the session tree and `qualify`'s own `census_digest` covers its generation block, so an in-place edit turns a valid generation invalid. M2.5 already refused the publisher cycle this would reopen. |
| D04 | The published rows carry **no event, asset, capture, subject, path or filename key**, and the alphabets make that unrepresentable rather than merely unwritten. | The tree is redaction-safe by contract, which is what earns it a claim boundary M2.7.3 can quote and M2.7.2 can scan. A shape that could carry an identifier needs a scan to prove it does not; a shape that cannot needs none. |
| D05 | Upstream binding = **`qualification/` alone**, through `qualify.validate_generation(out, sessions_dir, inventory_dir)`. | It is a real upstream of the evidence, not a convenience: `scripts/probe_calibration_observability.py:156-210` validates inventory + sessions + qualification, reads `assets_qc.csv` and `cameras_qc.csv`, and stores all three generations in the cache fingerprint payload at `:255`. Binding `qualification/` therefore binds the whole chain transitively. |
| D06 | Evidence enters as **captured probe stdout** (line-delimited JSON), one file per cited probe, each accompanied by the SHA-256 of the probe script that produced it. | Makes the ruling re-derivable and keeps the estimator out. The probes already emit exactly this: `probe_bias_transfer.py` streams one compact JSON object per labelled arm and its final pretty-printed object carries only the sorted key list. |

## §3 Evidence, re-derived by MAIN from committed state

`scripts/probe_bias_transfer.py --cache .scratch/calib-obs-wide`, rc=0, replayed this window.
Reproduces the M2.6b record exactly, so the roadmap numbers are credited by rerun rather than
carried forward.

- Population: **178 pairs over 103 events**; residual magnitude median **17.038 px**.
- Within-event ceiling **+0.8138** (129/178 above 0.5).
- Between-event signed r by grouping: same view pair **+0.0108** (n=4341, 787 above 0.5);
  `above|left` **+0.0311** (n=1207); `above|right` **−0.0029** (n=1858); `left|right` **+0.0117**
  (n=1276); + same model pair **+0.0102** (n=2738); + same task **+0.0103** (n=1071); + same
  subject **−0.0296** (n=275); keypoints permuted (null) **+0.0051** (n=4692).
- Magnitude |r|: pooled **+0.1499**, within subject **+0.1462**, `left|right` **+0.2191**, null
  **−0.0324**.
- Shared references (repairable mechanism): image bias 8 px **+0.9662**, 32 px **+0.7632**; under
  1.2 m rig jitter 8 px **+0.7961**, 32 px **+0.1799**; anatomical 20/40/80 mm **+0.9411 / +0.2192
  / +0.6264**. Floor across every shared arm = **+0.1799**.
- Non-shared references: per-event bias 8/32 px **−0.0106 / +0.0052**; noise 8/32 px **+0.0159 /
  +0.0303**.
- Separation: shared arms 0.180–0.966, non-shared arms −0.011–0.030, corpus −0.030–0.031. The
  corpus sits inside the non-shared band. Quote the **0.180 floor**, never a single shared arm —
  the shared arms are not monotone in magnitude because field realization dominates at 3 draws.

One schema fact from the replay: `shared_fraction` is **nullable** — `SYNTH noise sigma=8.0px`
emits `null`. The evidence table's alphabet must admit an empty cell there.

## §4 Carrier constraints, measured by MAIN this window

Read from the published `qualification/` tree (`GENERATOR_VERSION` v4). All five confirm the
roadmap's statement, so the contract rests on measurement rather than on the plan's text.

| surface | measured | consequence for F1a |
| ------- | -------- | ------------------- |
| `events_qc.geom_qualified` | blank on **193/193** | Stays blank. F1a publishes no per-event geometry verdict. |
| `events_qc.qualified` | blank on **193/193** | Untouched. |
| `events_qc.reason` | `geom_unmeasured` on **193/193** (173 alone, 20 with `sync_unqualified`) | The `geom` token stays UNMEASURED at event grain. The corpus-level ruling is a different grain and must say so in words. |
| `assets_qc.scale_ref_class` | blank on **379/379** | Untouched. |
| `assets_qc.qc_flags` | `scale_unmeasured` on **379/379** | Untouched. |

**One latent inconsistency, found by measurement and ruled here.** `qualification.json` publishes
`measured_axes = [detect, orientation, rigidity, sync, timebase]` and `unmeasured_axes = [scale]`.
`geom` appears in **neither**, because `qualify.py:1480` builds `unmeasured_axes` as
`sorted(set(SIDECAR_AXES) - measured_axes)` and `geom` is not a sidecar axis — yet every one of the
193 event rows carries `geom_unmeasured`, and `docs/technical/qualification.md:108` instructs a
consumer to read those two lists first. So a consumer reading the census alone learns nothing about
`geom`.

Ruled: **the reconciliation is a documentation pointer, never an artifact patch.** D03 forbids
touching the published tree, and widening `unmeasured_axes` would additionally require a
`GENERATOR_VERSION` bump plus a corpus republish plus a determinism regeneration for a field no
consumer reads. `docs/technical/qualification.md` gains a pointer to `calibration_qc/`; the
published bytes do not move. The axis-census gap itself is a `qualify` defect and goes to
`.agent/polish.md` rather than into this unit.

## §5 Published shape

Wave 1 delivered the inputs; MAIN writes the column list next window against
`.scratch/agents/scout-m2u71.md` E08. Two shape rulings hold already:

- **Three entries**: `calibration_qc.json` (marker + generation block), `corpus_qc.csv` (exactly
  one row), `evidence_qc.csv` (one row per cited probe arm). The marker is excluded from
  `tree_digest` and carries it, exactly as `qualify.py:1533-1550` does.
- **`evidence_qc.csv` flattens the probes' nested statistic dicts.** Both probes emit
  `{n, median, min, max, above_0p5}` per statistic, and `shared_fraction` is nullable — measured:
  `SYNTH noise sigma=8.0px` emits `null`. The alphabet must admit an empty cell there without
  admitting an empty cell in a required column.

## §6 Predicates

MAIN writes the predicate table next window from `.scratch/agents/map-m2u71.md` S12 (anchored
normative checklist) and S06 (registration surface). Wave 1 already fixes the claim half:

**Claim matrix C01-C15** — `.scratch/agents/scout-m2u71.md` E04, each row a required positive
statement with its evidence anchor and its exact prohibited paraphrase. Accepted as the contract's
claim predicates. The five that bind hardest, because each one is a plausible overreach the
artifact must actively refuse:

| id | required statement | prohibited paraphrase |
| -- | ------------------ | --------------------- |
| C03 | The shipped estimator is exact on exact synthetic correspondence, and independent BA worsens corpus closure. | "No estimator could recover extrinsics from these observations." |
| C07 | Held-out reprojection on the solve's own keypoint family is self-consistency. | "…measures calibration accuracy." |
| C11 | A lower-bias keypoint source and a detector trained for multi-view consistency stay outside the measured bound. | "No detector could do this." |
| C13 | The per-event double-centered bias-and-pose synthetic-control arm is **unrun**. | "A per-event joint bias-and-pose solve cannot recover known extrinsics." Never `failed`, `refused` or `impossible`. |
| C14 | One corpus-level ruling holds while every per-event geometry cell stays unmeasured. | "All 115 eligible events independently failed calibration and are geometry-unqualified." |

**Every synthetic arm is instrument calibration** (C15): its meaning arises only in contrast with
the corpus row, so the artifact must never let a synthetic value stand alone as a corpus claim.

## §7 Invariant surfaces

1. `qualification/` — all five entries byte-identical before and after any F1a run. D03.
2. The 193/193 and 379/379 sentinel censuses of §4 — unmoved.
3. The decisive suite — 1284 passed / 0 skipped at `41efc55`, plus F1a's own cases, 0 skipped.
4. `docs/technical/qualification.md` — gains a pointer to `calibration_qc/` and nothing else; §4's
   axis-census gap is a `qualify` defect and stays out of this unit.

## §8 Gate identity

Primary tree, MAIN-run, never beside a decode or inference sweep:

```sh
env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync ruff check \
  && env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync ruff format --check \
  && env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync ty check \
  && env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync pytest
```

Baseline to beat: **1284 passed / 0 skipped / rc=0** at `41efc55`, unmoved through M2.6 and M2.6b.

## §9 Probe-corpus seed

Eight classes the red suite must cover. Classes 1-5 are the idiom's known crash and trust-root
states, inherited from `qualify` and from the four "reads stricter than it is" traps; 6-8 are
F1a's own.

1. **Ownership** — non-empty unowned `--out`; marker that is a symlink; marker carrying a duplicate
   key; marker from a different `GENERATOR_VERSION`; `--out` that exists and is not a directory.
2. **Atomicity** — kill between the two renames leaves the only complete generation under a dead
   pid, so the sweep runs strictly after the swap; failed `staging.rename(out)` restores the
   retirement only when the root stayed absent; `_sweep_orphans` survives an over-wide pid
   (`OverflowError`, not `ValueError`).
3. **Disjointness** — `--out` overlapping `qualification/` or an evidence file, in either
   direction; a symlinked `--out` publishes to the resolved path.
4. **Integrity** — each of: an edited CSV, an edited marker, a file added to the tree, a file
   removed, a stale upstream `qualification/` generation — every one refused by `validate_generation`
   with its own message, checked by **exception class** rather than by message text.
5. **Alphabets** — every cell pattern is `fullmatch` and every enumerated cell is built from its
   token frozenset, never from a character class; a zero-row CSV must fail rather than publish,
   because a table with no rows carries its schema in the header alone.
6. **D04 unrepresentability** — an evidence or corpus row carrying an event, asset, capture,
   subject, path or filename key is refused by the schema, not merely absent from the corpus run.
7. **Evidence validation** — probe stdout that is truncated mid-line, carries an unknown arm label,
   omits a required statistic key, or arrives with a probe-script digest that does not match the
   committed script, is refused before the output tree is touched.
8. **Claim conformance** — the C01-C15 matrix: for each prohibited paraphrase, a case proving the
   published text does not contain it, and for each required statement, a case proving it does.

## §10 Amendments

*(none yet)*
