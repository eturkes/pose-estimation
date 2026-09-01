# Prospective calibrated capture specification

This document specifies a future motion-capture acquisition. That acquisition is the only route
past the claim boundary of the present corpus. **This specification defines a capture that nobody
has performed.** It is a design, not a record.

**Which document governs which capture.** This specification governs a *new, prospective*
acquisition built for calibrated 3D. `docs/capture_protocol.md` governs a capture recorded with the
tooling this repository already ships. That protocol stays correct for its own scope. Do not read
one document as an amendment to the other. The two differ on purpose: this specification requires
hardware and metrology that the shipped commands do not support.

## How to read a section

Each section carries five fields.

- **Owner** — the named role that answers for the section.
- **Record** — the artifact the capture must produce. A rule with no artifact is not auditable.
- **Threshold** — the value that decides pass or fail.
- **Failure action** — what the operator must do when the threshold fails.
- **Evidence** — the source behind the rule. See *Evidence sources*.

## The five non-negotiables

A capture that misses any one of these cannot support a 3D claim.

| id | requirement | sections |
| -- | ----------- | -------- |
| N1 | Intrinsic and extrinsic calibration | S11, S12 |
| N2 | Synchronization residuals, measured and gated | S14 |
| N3 | Orientation and drift control | S10 |
| N4 | Traceable metric scale | S13 |
| N5 | Identifiable-video governance | S04, S20 |

**Why these five.** Each one prevents a failure the existing corpus already shows.

- **N1.** The existing recordings carry no calibration and no camera-intrinsics metadata. Cameras
  that move between takes cannot share one rig calibration.
- **N2.** Cameras started by hand run at unequal rates. A single integer frame offset cannot express
  the drift between them.
- **N3.** Some existing clips change device orientation partway through. One rotation constant per
  clip cannot express that change.
- **N4.** Image geometry alone recovers shape up to one unknown factor. Every metre-valued quantity
  is unavailable without a physical length in the scene.
- **N5.** Raw video identifies a person through body shape, gait and room context. Consent to take
  part is not consent to release.

## Local decisions

Five thresholds in this document rest on a measured absence of any published standard. Each one is
a **local decision** of this study. Do not cite one as an external standard. A future committee may
set a different value without contradicting any source.

| id | absence | affected threshold |
| -- | ------- | ------------------ |
| L1 | No broadly accepted reporting checklist for clinical markerless motion capture exists. | The 20-section spine below is a **local decision**, synthesized from the sources in S01. |
| L2 | No universal reprojection-error threshold for clinical work exists. | The pixel gates in S11 and S12 are a **local decision**. |
| L3 | No sourced ratio of calibration-board size to capture volume exists. | The board sizing in S11 is a **local decision**. |
| L4 | No generic retention period for non-invasive Japanese research video exists. | The retention term in S04 and S20 is a **local decision**, set by the hospital and the committee. |
| L5 | No published millisecond synchronization figure for OpenCap, Pose2Sim or Anipose exists. | The residual gates in S14 are a **local decision**, derived from the task motion budget. |

---

## S01 — Scope, estimands and claim boundary

Fix the tasks, the joints and the measured quantities before the first capture. Every permitted
claim MUST name one acceptance gate in S19. Delete any claim that no gate decides. The study MUST
NOT report fitness for clinical decisions from this capture alone. The study MUST NOT report
agreement with a marker-based reference unless S17 supplies that reference.

- **Owner:** clinical lead.
- **Record:** `estimands.yaml` — tasks, joints, quantities, permitted claims, prohibited claims.
- **Threshold:** every permitted claim maps to exactly one S19 gate.
- **Failure action:** remove the claim, or add its gate before capture starts.
- **Evidence:** [E1], [E4], [E5]. Spine is a **local decision** under L1.

## S02 — Document control and responsibilities

Pin the protocol identifier and version before capture. Name one person for each of five roles.
The roles are capture operator, clinical safety lead, calibration approver, data steward and
incident owner. Every session manifest MUST carry the protocol version that produced it.

- **Owner:** data steward.
- **Record:** `protocol_control.yaml` — identifier, version, change history, five named roles.
- **Threshold:** every role names one person; manifest version equals protocol version.
- **Failure action:** hold capture until the roles and the versions agree.
- **Evidence:** [E1], [E13].

## S03 — Study design, population and sampling

State the design, the inclusion rules and the exclusion rules. Record impairment strata, the tested
side and hand dominance. Derive the sample size from the S19 statistic, not from convenience.
Report the recruitment flow and the subgroups the result applies to.

- **Owner:** clinical lead.
- **Record:** `cohort.yaml` — design, inclusion, exclusion, strata, side, dominance, target count.
- **Threshold:** sample size derived from a named S19 statistic and its effect size.
- **Failure action:** revise the sample size before the first subject.
- **Evidence:** [E1], [E5], [E11].

## S04 — Ethics, consent and identifiable-video governance

The study MUST hold ethics-committee approval before the first capture. Consent MUST be written and
layered. One layer covers participation. A separate layer covers release of identifiable video.
Never treat participation consent as release consent. Facial blurring MUST NOT be recorded as
anonymization. Treat raw video as controlled personal data that carries a separately held key.
Honour withdrawal by destroying the recording and its derivatives.

- **Owner:** principal investigator.
- **Record:** `consent_ledger.csv` — subject pseudonym, approval identifier, one column per consent
  layer, version, date, withdrawal state.
- **Threshold:** every capture traces to an approved consent row that carries the layer it needs.
- **Failure action:** hold the capture in the controlled tier, or destroy it.
- **Evidence:** [E12], [E13], [E6]. Retention term is a **local decision** under L4.

## S05 — Task script, safety and trial schedule

Fix the posture, the apparatus, the cues and the speed envelope. Fix the repetition count and the
rest interval. Define the stop rules for fatigue and for pain. Define the retry rule and the abort
rule. The operator MUST log every deviation on the trial row.

- **Owner:** capture operator.
- **Record:** `trial_log.csv` — trial identifier, script version, repetitions, deviations, stops.
- **Threshold:** every attempted trial carries a script version and a disposition.
- **Failure action:** mark the trial `repeat` or `reject`, and give the reason.
- **Evidence:** [E6], [E11], [E1].

## S06 — Room, lighting and scene controls

Dimension the capture volume in metres. Specify the illuminance and the flicker limit. Remove
reflective surfaces and moving occluders from the field of view. Fix the background. Photograph the
room state at the start of each epoch. The room state MUST NOT change inside one calibration epoch.

- **Owner:** capture operator.
- **Record:** `room_state.yaml` plus one reference photograph per epoch.
- **Threshold:** illuminance at or above 1000 lux, flicker-free, measured at the volume centre.
- **Failure action:** correct the room, then start a new calibration epoch.
- **Evidence:** [E6], [E3].

## S07 — Hardware and software inventory

Record every camera, lens, mount, trigger, calibration target and reference instrument. Record
serial numbers, firmware versions and the acquisition software version. Attach the calibration
certificate for each measured reference. A component with no serial number MUST NOT enter the rig.

- **Owner:** calibration approver.
- **Record:** `inventory.yaml` — one row per component, with serial, firmware and certificate.
- **Threshold:** every component in the rig appears in the inventory for that epoch.
- **Failure action:** stop capture until the inventory matches the rig.
- **Evidence:** [E7], [E4].

## S08 — Camera layout and visibility proof

Deploy eight cameras. Survey each position and each optical axis. Record the baselines and the
overlap. Every target joint MUST appear in three or more views throughout the task. Prove that
coverage in a preflight pass before the subject arrives.

- **Owner:** capture operator.
- **Record:** `layout.yaml` — surveyed positions, axes, baselines, plus the preflight coverage map.
- **Threshold:** three or more views per target joint, for every frame of the preflight pass.
- **Failure action:** move a camera, then repeat the preflight pass and the calibration.
- **Evidence:** [E6], [E7], [E11].

## S09 — Sensor mode and image-quality qualification

Lock the frame rate at 120 Hz. Use a global shutter. If no global shutter is available, measure the
readout time and keep it at or below 2 ms. Keep exposure at or below 1/1000 s. Never exceed
1/500 s. Use 1080p or better. The hand MUST span 150 px or more. The person MUST span 500 px or
more. Lock the focus, the gain and the codec, and disable stabilization.

At 1 m/s hand speed these settings give a sample spacing near 8.33 mm. An exposure of 1/1000 s
gives about 1 mm of blur. A 2 ms readout contributes about 2 mm of row skew.

- **Owner:** calibration approver.
- **Record:** `sensor_mode.yaml` plus a delivered-timing report with the dropped-frame count.
- **Threshold:** delivered frame rate within 1% of nominal, and zero dropped frames per trial.
- **Failure action:** reject the trial, then correct the mode before the next attempt.
- **Evidence:** [E14], [E3], [E7].

## S10 — Mechanical mounting, orientation and drift epochs

Lock every mount. Add a witness mark to each camera and to each tripod leg. Define the pixel
orientation convention once, and apply it everywhere. Check the witness marks and a fixed scene
sentinel at the start and at the end of every session. A camera that moves MUST close its
calibration epoch. Every trial inside a broken epoch MUST be invalidated or recalibrated.

Orientation MUST NOT change inside a clip. A per-clip rotation constant cannot express a mid-clip
change, so the capture must prevent one.

- **Owner:** capture operator.
- **Record:** `epoch_log.csv` — epoch identifier, start, end, sentinel residual, movement verdict.
- **Threshold:** sentinel residual at or below 2 px between the start check and the end check.
- **Failure action:** close the epoch, recalibrate, and bracket every trial since the last good check.
- **Evidence:** [E6], [E2], [E7]. The 2 px sentinel value is set by this study.

## S11 — Intrinsic calibration

The calibration target MUST be rigid, and its geometry MUST be measured. Measure its square pitch
with a traceable instrument, and record the uncertainty. Each camera MUST contribute twenty or more
board poses. Spread the poses across the
frame, the depth range and the tilt range. Keep each square at 20 px or more. Keep the board across
20% or more of the image area. Fix the optical state before the first pose, and never change it
inside an epoch. Record the distortion model, the solver flags and the software version.

- **Owner:** calibration approver.
- **Record:** `intrinsics/<camera>.json` — model, flags, parameters, residual distribution, version.
- **Threshold:** fit residual at or below 0.5 px target; above 1.0 px fails.
- **Failure action:** recapture the board sweep, then solve again.
- **Evidence:** [E2], [E3]. Pixel gates are a **local decision** under L2. Board sizing is a
  **local decision** under L3.

## S12 — Extrinsic calibration and coordinate frame

Sweep the target through the whole volume so that every camera pair co-observes it. Use twenty-seven
stationary placements across a three-by-three-by-three lattice. The camera graph MUST be connected
without a chain through an unobserved pair. Solve by bundle adjustment over all cameras. Declare the
origin, the axis directions and the handedness. Bind each solve to one epoch identifier from S10.

The fit residual grades the fit alone. It MUST NOT be reported as the accuracy of the recovered
geometry. S17 supplies the independent check.

- **Owner:** calibration approver.
- **Record:** `extrinsics/<epoch>.json` — poses, graph, residuals, origin, axes, handedness.
- **Threshold:** fit residual at or below 0.5 px target; above 1.0 px fails; graph connected.
- **Failure action:** sweep the uncovered pairs again, then solve again.
- **Evidence:** [E2], [E7], [E8]. Pixel gates are a **local decision** under L2.

## S13 — Metric-scale traceability

Image geometry alone recovers shape up to one unknown factor. The capture MUST therefore carry a
physical length into every calibration epoch. Traceability requires an unbroken calibration chain to
a stated reference. Every link in that chain MUST contribute a documented uncertainty. This
specification names the SI metre as that reference. Use a measured carrier that meets it, and state
its uncertainty. A certificate number alone does not prove traceability. Fit the scale on that
carrier. Seal a second length, and keep it out of the fit. Recover the sealed length, and compare it
against its measured value.

Never infer scale from anthropometric tables or from furniture. Never carry a scale across an epoch
boundary.

- **Owner:** calibration approver.
- **Record:** `scale/<epoch>.json` — carrier identifier, certificate, uncertainty, fitted factor,
  sealed-length recovery.
- **Threshold:** recovered sealed length within 1% of its measured value.
- **Failure action:** remeasure the carrier, then recalibrate the epoch.
- **Evidence:** [E10], [E9], [E7].

## S14 — Synchronization and rolling-shutter model

Prefer a wired trigger or genlock. If no cable route exists, use an encoded optical clock in view of
every camera. Fit the start offset and the drift across each session. Timecode alone, host clocks,
remote start and a free-field clap MUST NOT pass this gate on their own. Report the dropped-frame
count per camera. Where a rolling shutter is in use, model the row time and carry it into S19.

Let `T` be the frame period. The exposure-level residual MUST satisfy `p95 <= 0.10T` and
`max <= 0.25T`. Drift across the session MUST stay at or below `0.05T`.

- **Owner:** capture operator.
- **Record:** `sync/<session>.json` — route, per-camera offset, drift, residual distribution, drops.
- **Threshold:** `p95 <= 0.10T`, `max <= 0.25T`, drift `<= 0.05T`, at 120 Hz.
- **Failure action:** reject the session, then repair the timing route before recapture.
- **Evidence:** [E15], [E7]. Residual gates are a **local decision** under L5.

## S15 — Session, trial and provenance manifest

Link every trial to its consent scope, its subject pseudonym, its calibration epoch, its camera
streams, its sync record and its configuration hash. Record the operator and every deviation. Each
join MUST be unambiguous. A trial with an ambiguous join MUST NOT enter analysis.

- **Owner:** data steward.
- **Record:** `manifest/<session>.json` — one entry per trial, with every identifier above.
- **Threshold:** every trial resolves to exactly one epoch, one consent row and one configuration.
- **Failure action:** repair the manifest, or exclude the trial and record the exclusion.
- **Evidence:** [E7], [E4], [E13].

## S16 — Preflight, capture and postflight disposition

Run an executable checklist before every session. Check calibration freshness, storage headroom,
focus, exposure, target checks, joint visibility, sync residual and drift sentinels. Run the same
checklist after the session. Assign every attempted trial one disposition. The dispositions are
`accept`, `repeat` and `reject`. Every disposition MUST carry a reason code.

No trial may be silently absent from a denominator. An attempted trial with no row is a defect.

- **Owner:** capture operator.
- **Record:** `disposition.csv` — trial identifier, disposition, reason code, checklist results.
- **Threshold:** attempted trials equal the sum of the three dispositions.
- **Failure action:** reconcile the ledger before the session closes.
- **Evidence:** [E6], [E7].

## S17 — Independent reference and validation acquisition

Seal the check targets and the wand from the calibration fit. Collect thirty or more independent
check poses across the volume. Report object-space error, length error and angle error on that
held-out set. Where the study claims accuracy on human motion, capture a second modality at the same
time, and state its own uncertainty.

A residual computed on the data that produced the solve certifies self-consistency. It MUST NOT be
reported as accuracy. A keypoint-recovered calibration has beaten a reference calibration on human
reprojection while losing to it on independent targets.

- **Owner:** calibration approver.
- **Record:** `validation/<epoch>.json` — sealed targets, poses, per-axis error distributions.
- **Threshold:** thirty or more independent check poses per epoch, none used in the fit.
- **Failure action:** collect more sealed poses before any accuracy claim.
- **Evidence:** [E8], [E9], [E7], [E11].

## S18 — Processing and model contract

Pin the detector weights, the training domain and the keypoint schema. Pin the association rule, the
confidence handling, the missingness rule, the triangulation method and the filter settings. Pin the
biomechanical model and every coordinate transform. Record the exact commands and the container
digest. Scale MUST NOT be introduced at this stage. Scale comes from S13 alone.

- **Owner:** data steward.
- **Record:** `processing.yaml` plus the container digest and the command log.
- **Threshold:** a clean rerun from the record reproduces the outputs byte for byte.
- **Failure action:** pin the missing input, then rerun until the outputs match.
- **Evidence:** [E2], [E3], [E4].

## S19 — Acceptance statistics, uncertainty and exclusions

Predeclare every gate before capture. Declare gates per camera, per volume region, per task and per
subgroup. Report point error, length error, angle error and sync error from the S17 held-out set.
Report agreement statistics and measurement error. Report every failure, every missing value and
every denominator. A statistic with no stated denominator MUST NOT be published.

- **Owner:** clinical lead.
- **Record:** `acceptance.yaml` predeclared, plus `results.json` after capture.
- **Threshold:** every predeclared gate reports a verdict and a denominator.
- **Failure action:** report the gate as failed, and withdraw the claim it supported.
- **Evidence:** [E1], [E4], [E5], [E9], [E11].

## S20 — Security, release, reproducibility and change control

Hold raw video in the hospital-controlled zone. Encrypt at rest and in transit. Apply least
privilege and multi-factor access. Log every access and every export. Keep direct identifiers out of
paths and metadata. Administer the linkage key separately. Release aggregates, configuration and
synthetic fixtures by default. Any wider release MUST clear a re-identification assessment and the
S04 release layer. Record every recipient, purpose, date and deletion confirmation. Define the
requalification triggers and the claim-rollback rule.

- **Owner:** data steward.
- **Record:** `release_ledger.csv` — recipient, basis, purpose, fields, date, deletion confirmation.
- **Threshold:** every export appears in the ledger with an approved basis.
- **Failure action:** revoke the export, then run the incident procedure.
- **Evidence:** [E13], [E12], [E6]. Retention term is a **local decision** under L4.

---

## Evidence sources

- [E1] von Elm et al., STROBE statement, *PLOS Medicine*, 2007. https://doi.org/10.1371/journal.pmed.0040296
- [E2] Karashchuk et al., Anipose, *Cell Reports*, 2021. https://doi.org/10.1016/j.celrep.2021.109730
- [E3] Pagnon et al., Pose2Sim, *Sensors*, 2021. https://doi.org/10.3390/s21196530
- [E4] STARD-AI Steering Group, *Nature Medicine*, 2025. https://doi.org/10.1038/s41591-025-03953-8
- [E5] Mokkink et al., COSMIN, *Quality of Life Research*, 2024. https://doi.org/10.1007/s11136-024-03761-6
- [E6] Uhlrich et al., OpenCap, *PLOS Computational Biology*, 2023. https://doi.org/10.1371/journal.pcbi.1011462
- [E7] Evans et al., BioCV dataset, *Scientific Data*, 2024. https://doi.org/10.1038/s41597-024-04077-3
- [E8] Pätzold et al., GCPR, 2022. https://doi.org/10.1007/978-3-031-16788-1_19
- [E9] ASPRS, Positional Accuracy Standards, 2024. https://asprs.org/Main/Main/Standards/Positional-Accuracy-Standards.aspx
- [E10] NIST, Metrological traceability. https://www.nist.gov/calibrations/traceability
- [E11] Hansen et al., *Biomedical Engineering Advances*, 2024. https://doi.org/10.1016/j.bea.2024.100128
- [E12] MEXT, MHLW and METI, Ethical guidelines for human-subject life-science research. https://www.mhlw.go.jp/content/001457376.pdf
- [E13] Personal Information Protection Commission, APPI general guidelines. https://www.ppc.go.jp/personalinfo/legal/guidelines_tsusoku/
- [E14] Kim and Neville, *Scientific Reports*, 2023. https://doi.org/10.1038/s41598-023-29091-0
- [E15] Meyer et al., *Sensors*, 2026. https://doi.org/10.3390/s26031036
