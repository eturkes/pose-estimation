# Calibration ruling

`pose-estimation-calibration-qc` publishes one corpus-level calibration ruling and the evidence set that supports it.
The ruling covers extrinsic recovery from RTMW-L keypoints on this corpus at 1080p under per-model intrinsic priors.

The tool publishes beside `qualification/`. It never modifies that tree or converts the ruling into per-event geometry verdicts.
It is the fifth artifact publisher, after `inventory`, `sessions`, `qualify`, and the `measure` sidecar.
It runs no probe and computes no statistic.

## Run the tool

```sh
pose-estimation-calibration-qc \
  --qualification qualification \
  --evidence evidence \
  --probes scripts \
  --out calibration_qc \
  --sessions sessions \
  --inventory inventory

python -m pose_estimation.calibration_qc \
  --qualification qualification \
  --evidence evidence \
  --probes scripts \
  --out calibration_qc
```

| Flag | Requirement | Help text |
| ---- | ----------- | --------- |
| `-h`, `--help` | Optional | show this help message and exit |
| `--qualification` | Required | Directory that holds `qualification.json`. |
| `--evidence` | Required | Directory that holds one `<probe>.jsonl` capture and one `<probe>.sha256` beside it. |
| `--probes` | Required | Directory that holds the cited probe scripts. |
| `--out` | Required | Directory to publish the ruling into. |
| `--sessions` | Optional | Session tree to check the upstream against. |
| `--inventory` | Optional | Registry directory to check the upstream against. |

The parser description is `Publish the corpus-level calibration ruling and the evidence behind it.`
Pass every available upstream root. An omitted optional root receives no freshness check.

Help exits 0 before dispatch. A successful publication prints a redaction-safe summary and exits 0.
Argparse usage errors exit 2 before dispatch.
A refusal from this tool or from any upstream validator prints one `Error:` message to stderr and exits 2.
The handled classes are `CalibrationQcError`, `QualifyError`, `SessionsError`, and `InventoryError`.
Other exceptions propagate.

The success summary contains the ruling, its bound, evidence counts, ingested probe names, and the claim count.
It contains no corpus identifier or path.

The current inputs publish in 1.6 seconds from a cold start and 0.5 seconds when warm.

## Inputs

The tool validates `qualification/` before it reads captured evidence.
It passes `--sessions` and `--inventory` to the qualification validator when you supply them.
It validates all captured evidence before it judges the output tree.
A refused input therefore leaves the output tree untouched.

The evidence directory must contain these files:

| File | Purpose |
| ---- | ------- |
| `bias_transfer.jsonl` | Captured stdout from `probe_bias_transfer.py`, with one compact JSON arm per line. |
| `bias_transfer.sha256` | The SHA-256 of the script that produced the capture. The first whitespace-delimited field is the digest. |

The probe directory must contain both cited scripts:

- `probe_bias_transfer.py`
- `probe_calibration_bias.py`

The tool digests both scripts. It refuses a missing script or a capture recorded under different script bytes.

Only `bias_transfer` records enter `evidence_qc.csv`.
The `calibration_bias` script remains cited and digested, but its differently shaped stdout is not ingested.
Its numerical findings belong in the claim-bounded report.

### What the tool refuses

| Defect | `reason` code |
| ------ | ------------- |
| The capture or its digest sidecar resolves inside the output tree. | `output_overlap` |
| A record carries a key outside `label`, `pairs`, `events`, `realizations`, `shared_fraction`, and the four statistic blocks. | `forbidden_key` |
| A statistic block is absent, or it omits `n`, `median`, `min`, or `max`. | `evidence_schema` |
| One arm label appears twice. | `arm_duplicate` |
| A cited arm or a reference band is absent. | `arm_missing` |
| A cited arm reports fewer than 178 pairs or 103 events. | `population_mismatch` |
| A line is cut in the middle, so it is not one JSON document. | `evidence_malformed` |
| The recorded digest does not match the live script bytes. | `probe_digest` |
| A cited probe script is absent. | `probe_missing` |

The capture must retain these cited arms:

- `REAL same view pair`
- `REAL same view pair + same model pair`
- `REAL same view pair + same task`
- `REAL same view pair + same subject`
- `REAL same view pair, keypoints permuted (null)`

It must also retain one arm for each reference band: `SYNTH shared image bias `, `SYNTH per-event bias `, and `SYNTH noise sigma=`.
Each cited arm must report the full eligible population of 178 pairs over 103 events.
The arm set stays open above that floor. The evidence table is a transcript of the arms that the probe emitted.

The output must not equal, contain, or sit inside any input directory.
The same rule applies when an input contains the output.

## Outputs

The tool publishes exactly three entries. The schema is `GENERATOR_VERSION` `v1`.

| Entry | Grain | Purpose |
| ----- | ----- | ------- |
| `corpus_qc.csv` | One row for the corpus | Publishes the fixed ruling and its measured scope. |
| `evidence_qc.csv` | One row per probe, arm, and statistic | Publishes the long-form captured evidence. |
| `calibration_qc.json` | One generation | Publishes aggregate counts, the claim bound, and the generation marker. |

The current generation contains one corpus row, 84 evidence rows over 21 arms, and 15 claims.

The complete set is redaction-safe by schema.
No column can key a row to an event, asset, capture, subject, path, or filename.
The output root and its staging or retiring siblings are gitignored.

## `corpus_qc.csv`

The table contains exactly one row. The ruling is a module constant, so no argument can publish another verdict.

| Column | Published value | Meaning |
| ------ | --------------- | ------- |
| `ruling_grain` | `corpus` | Applies the ruling once to this corpus. |
| `recovery_status` | `unachievable` | Records the measured recovery verdict inside the stated bound. |
| `reason` | `cross_view_keypoint_bias` | Names the measured mechanism behind the verdict. |
| `transfer_status` | `absent` | Records the tested signed-bias transfer result. |
| `keypoint_source` | `rtmw_l` | Binds the ruling to RTMW-L keypoints. |
| `image_height_px` | `1080` | Binds the ruling to the measured image height. |
| `intrinsics_basis` | `per_model_prior` | Binds the ruling to per-model intrinsic priors. |
| `unrun_arm` | `per_event_double_centered_bias_and_pose` | Names the remaining synthetic-control arm. |
| `unrun_arm_status` | `unrun` | States that the named arm has no measured outcome. |
| `cited_probes` | `bias_transfer\|calibration_bias` | Names both committed probes that support the ruling. |

Every closed status uses its exact token set. The `unrun_arm_status` alphabet admits only `unrun`.

## `evidence_qc.csv`

Each `bias_transfer` arm produces four rows, one for each required statistic.
Rows sort by `(probe, arm, statistic)`.

| Column | Meaning |
| ------ | ------- |
| `probe` | Contains the closed probe identifier. Published rows use `bias_transfer`. |
| `probe_sha256` | Contains the live SHA-256 of `probe_bias_transfer.py`. |
| `arm` | Preserves the arm label that the probe emitted. |
| `statistic` | Contains `between_event_r`, `between_event_r_abs`, `within_event_r`, or `median_abs_px`. |
| `n` | Contains the statistic sample count. |
| `median` | Contains the median, rendered to four decimal places. |
| `min` | Contains the minimum, rendered to four decimal places. |
| `max` | Contains the maximum, rendered to four decimal places. |
| `above_0p5` | Contains the count above 0.5. It stays empty when that statistic has no such field. |

`above_0p5` is the one nullable field. For example, `median_abs_px` carries no `above_0p5` value.
An empty cell is never a measured zero.
A capture that drops `n`, `median`, `min`, or `max` is refused instead of published as a short row.

## `calibration_qc.json`

The marker contains this closed top-level structure:

| Key | Contents |
| --- | -------- |
| `claims` | The 15 statements that bound every permitted reading of the set. |
| `corpus` | `rows` and the complete `ruling` object from `corpus_qc.csv`. |
| `evidence` | Row, arm, probe, and statistic censuses from `evidence_qc.csv`. |
| `schema_version` | The current `GENERATOR_VERSION`. |
| `generation` | File, tree, upstream, probe, version, and census digests. |

The `evidence` object contains `rows`, `arms`, `probes`, and `statistics`.
The `probes` census lists ingested probes, so it contains `bias_transfer`.
The generation block separately digests both cited scripts.

The generation block has exactly these keys:

| Key | Meaning |
| --- | ------- |
| `corpus_qc.csv` | SHA-256 of the published corpus table bytes. |
| `evidence_qc.csv` | SHA-256 of the published evidence table bytes. |
| `tree` | Recursive digest of every entry except `calibration_qc.json`. |
| `qualification` | The validated qualification generation that publication consumed. |
| `probes` | Mapping from each cited probe identifier to its script SHA-256. |
| `generator_version` | Identifies this schema and generator as `v1`. |
| `census` | Digest of the marker after only `generation.census` is removed. |

The census digest covers the upstream and probe provenance.
The tree digest catches a file that is added, removed, replaced, or changed.

## Validate before you read

Call `validate_generation` before you read either CSV.
Pass every upstream root that remains available.

```python
from pose_estimation import calibration_qc

generation = calibration_qc.validate_generation(
    "calibration_qc",
    qualification_dir="qualification",
    sessions_dir="sessions",
    inventory_dir="inventory",
    probes_dir="scripts",
)
```

The check proves these properties:

1. `calibration_qc.json` is a regular, non-symlink file with one unambiguous JSON document.
2. The generation key set and `generator_version` match this generator.
3. Both CSV files match their published SHA-256 values.
4. The marker matches its census digest, including its provenance claims.
5. No output entry was added, removed, replaced, or changed.
6. The supplied qualification chain still matches the generation that publication consumed.
7. The supplied probe scripts still match the bytes that publication cited.

Pass `qualification_dir` to activate qualification freshness.
Pass `sessions_dir` and `inventory_dir` with it to validate the complete available chain.
Pass `probes_dir` to catch a cited script that changed or disappeared after publication.

The function returns the validated generation block.
It raises `CalibrationQcError` for this set's defects. Upstream validators retain their own error classes.
Every refusal carries a machine-readable `reason` attribute, so a consumer can branch on the cause.

An omitted optional root receives no check.
The function does not inspect the original evidence capture after publication.
It also does not rerun a probe or rederive a statistic.
Consumers must still parse the exact columns and enforce their own semantic predicates.

These digests detect corruption and edits. They do not authenticate the set because the set carries no signing key.

## Publication and ownership

The tool builds a complete staging sibling before it changes the live root.
It validates the staged claim text before the first rename.
It then retires an owned root, promotes staging, and sweeps dead siblings after promotion.
A current-input republish is byte-identical across all three entries.
It leaves `qualification/` unchanged and leaves no staging or retiring sibling.

The tool refuses a non-empty output unless its marker has this generator's exact shape and version.
Ownership does not require fresh digests, so a stale generation remains replaceable by its own generator.
A different generator version does not own the existing tree.
Delete that tree before the first publication under a new version.

Process identifiers are reused. A retiring sibling under this process identifier can therefore be the only complete
generation that a killed run left between its two renames. The tool restores that sibling before it judges ownership.

## Claim boundary

The marker publishes these supported statements. Each one is quoted verbatim from
`calibration_qc.CLAIMS`, so never reword one here: the publisher checks the published bytes against
that constant, and a reworded copy sends an editor text the tool refuses.
`scripts/check_claim_report.py` P02 pins this list.

1. **C01.** Extrinsic recovery from RTMW-L keypoints on this corpus at 1080p under per-model intrinsic priors is measured unachievable.
2. **C02.** Within-event cross-view RTMW-L correspondence carries a measured 15-20 px systematic component at 1080p.
3. **C03.** The shipped estimator is exact on exact synthetic correspondence, and independent bundle adjustment worsens corpus closure.
4. **C04.** No disjointly selected RTMW-L subset beats all 65 keypoints on the measured corpus folds.
5. **C05.** Signed bias transfer is absent at the tested view-pair, device-model, task and subject groupings over the full eligible population.
6. **C06.** The same keypoints share difficulty across events while the signed offset direction is redrawn every event, so that magnitude is not a correctable coordinate offset.
7. **C07.** Held-out reprojection on the solve's own keypoint family is self-consistency.
8. **C08.** This evidence is internal geometric and QC evidence only.
9. **C09.** Every pixel and degree statistic here stays separate from absolute metric accuracy.
10. **C10.** No marker-based comparison was run.
11. **C11.** A lower-bias keypoint source and a detector trained for multi-view consistency stay outside the measured bound.
12. **C12.** Prospective calibrated capture stays outside the measured bound and is the route that can reopen 3D.
13. **C13.** The per-event double-centered bias-and-pose synthetic-control arm is unrun.
14. **C14.** One corpus-level ruling holds while every per-event geometry cell stays unmeasured.
15. **C15.** Each synthetic arm is instrument calibration whose meaning arises only in contrast with the corpus row.

The tool scans the staged bytes before the first rename. It refuses a set that drops a required
claim (`claim_missing`). It also refuses a set that carries a prohibited paraphrase of one
(`claim_prohibited`).
The scan folds case, and it folds `_` and `-` to spaces, so a snake_case cell and a hyphenated arm label read alike.
The named unrun arm has no outcome, so the scan refuses any arm label that gives it one.

## What the tool does not do

- It does not execute either probe or any estimator.
- It does not recompute or interpret `calibration_bias` records. It checks no `calibration_bias` output at all.
- It does not authenticate a capture. A digest match binds the capture to one script version and proves nothing about who wrote the bytes.
- It does not modify or republish `qualification/`.
- It does not fill `events_qc.geom_qualified` or `events_qc.qualified`.
- It does not remove `geom_unmeasured` from an event reason.
- It does not publish an event, asset, capture, subject, path, or filename key.
- It does not authenticate the output.
- It does not supply a clinical claim, metric ground truth, or marker reference.
- It does not evaluate another keypoint source, multi-view training method, or prospective capture.
- It does not turn the named unrun arm into a measured outcome.

See [Capture qualification](qualification.md) for the upstream tree.
See [Command-line entry points](entrypoints.md) for the console index.
