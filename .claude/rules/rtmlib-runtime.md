---
paths:
  - "src/pose_estimation/run.py"
  - "src/pose_estimation/rtmlib_openvino.py"
  - "src/pose_estimation/rtmlib_smoothing.py"
  - "src/pose_estimation/models.py"
  - "src/pose_estimation/smoothing.py"
  - "src/pose_estimation/detection.py"
  - "scripts/probe_tracker_freeze.py"
  - "docs/technical/environment.md"
  - "docs/technical/tracking-modes.md"
---

# rtmlib runtime — device placement + the tracker

## Device placement — detector vs pose

- Detector + pose model take **separate** devices (`--det-device` / `--pose-device` on `run`/`main`/`benchmark`/`validate`). No `--device` flag exists anywhere; a bare `--device` is an argparse error, not a silent default.
- **rtmlib YOLOX must not run on NPU.** In-graph NMS ⇒ dynamic `dets` shape; NPU demands static ⇒ a fixed 100-row buffer whose unused rows are never written. Symptom: every frame reports exactly 100 detections, all rows sharing one score, values outside `[0,1]` (observed 1.128, 1.263). CPU on the same frames returns 1-3 detections, max 0.918. **It compiles cleanly**, so `rtmlib_openvino.py`'s NPU→CPU fallback never fires; the failure is numerical, silent, and reaches the CSV.
- **The padding is a device property, and only NPU pads with garbage.** One synthetic probe separates the three devices with no patient data: read `yolox_m_8xb8-300e_humanart-c2c7a14a.onnx` (rtmlib cache), compile per device, infer `np.zeros((1,3,640,640))`, print `dets`/`labels` shape + range. CPU keeps the dynamic output (`(1,1,5)`); GPU and NPU both materialise a fixed 100-row buffer with `labels` all `-1`; only NPU's `dets` hold uninitialised memory (`−0.1471…1.0215` on an all-zero image, so scores clear any threshold) while GPU's read exactly `0.0000`. Median latency: GPU 9.7 ms, NPU 108.4 ms, CPU 213.1 ms. NPU stays excluded.
- **GPU is qualified and adopted for the detector — at f32 only.** Over 240 corpus frames / 40 clips the f32 GPU detector matches CPU to IoU 1.0000, max score Δ 0.000003 at 6 dp (< 3.5e-6), raw box Δ < 0.0005 px in the 640×640 detector-input frame, 0 count mismatches; exact array equality holds on 8/240 frames alone. OpenVINO 2026.4.0 and 2026.4.1 give identical figures (`.scratch/ovupgrade/det_qual.py`). Per call 41.6 vs 318.7 ms (measured on an earlier build; the 2026.4.1 pilot wall sits inside the 2026.4.0 same-build spread). The plugin's f16 default drifts (IoU min 0.982, score delta 0.0018) and flips a box at the 0.3 score cut, so `rtmlib_openvino.py` pins `INFERENCE_PRECISION_HINT: f32` on every GPU compile. RTMW-X on GPU returns scores on another scale (92.9 % of body observations clip to 1.0) — never swap the pose model without checking its score scale, because every gate downstream reads it.
- Pose models are NPU-safe: RTMW-L NPU vs CPU = 0.505 px mean / 2.265 px p95 / 5.3 px p99 keypoint deviation, score MAE 0.00056. Per-call 7.17 ms NPU vs 134.26 ms CPU (~19×); detector 109.82 ms NPU (garbage) vs 445.21 ms CPU.
- **Those per-call numbers do not predict run throughput — measured end to end they are 4-5× optimistic.** The projection they supported (≈70 ms/frame ⇒ 6.5 h for the corpus) survived planning and two unit windows unchallenged; the pilot measured 3.03 fps over 8971 frames ⇒ 26.0-30.9 h. **Never size a run from per-call latency; run a stratified pilot and multiply.**
- MediaPipe is unaffected — SSD anchors + NMS decode in Python (`detection.py`), graphs stay static-shaped ⇒ both roles default NPU. `models.DETECTOR_MODELS` selects which compile on `--det-device`.
- Both call sites print the **requested** device rather than `compiled_model.get_property("EXECUTION_DEVICES")` — truthful only while exact device names are passed (`.agent/archive/polish.md`).

## `SubjectTracker` — the default (`--tracker subject`)

- **Every crop is a detector box, the detector runs every frame, and the pose never re-sizes the
  box.** Between detections and across detector misses the box moves by the median displacement
  of keypoints scoring >= 0.3 in consecutive poses; a track that sat out a frame moves by zero.
  This removes the mixed-convention loop below by construction.
- **Associate against the carried box AND the last detector box (max IoU).** One glitched pose
  carries the box off the person; against the carried box alone the next detection mints a new
  id and the orphan is posed on empty space for 15 frames — under `--single-subject` the orphan
  stays the subject.
- **Subject = largest detector box, sticky** (1.5× for 15 detector frames to switch). The rtmlib
  path's argmax of mean score over 133 keypoints (68 facial) picks a small fully visible bystander
  over a large truncated patient and ignores arms-only subjects.
- **The smoother keys on the tracker's ids**: a named track exports from its first frame and
  through its carried frames; `min_track_age` gates unnamed tracks alone.
- **Measured on the det_frequency sweep's own sample** (5 events / 11 assets / 400 frames, same
  instruments): `det_frequency=1` → 283.6 s, whole 0.000 %, isolated 0.000 %, alternation
  0.497, 2502 observed frames, 4.29 Hz peak/bg 0.850 (control range 0.840-1.161). The old f7
  arm: 327.3 s, 7.55 % / 8.06 %, 1.484, 1935, 1.336. Repaired arms 2-35 keep relocations near 0
  but sit at alternation 0.556-0.612, so 1 is the only arm meeting the ruling.
- **A generation names itself per event**: `pose_config.json` (`corpus_run.POSE_CONFIG_FIELDS`)
  beside the landmarks. Resume re-runs a complete event that records another configuration;
  `--analyse-only` and `--reuse-run` refuse to publish over one. **The OpenVINO build is not in
  that config**, and a build change moves output: 2026.4.0 → 2026.4.1 on the det_frequency sample
  changed 593 of 1.27 M landmark cells — 576 coordinates by 1e-6 to 4.1e-5 (6-dp quantum), 17
  confidences by 1e-4 (4-dp quantum) — and every feature column by
  ≤ 4.1e-4 of its own max, where the same build reruns byte-identical. Finish a corpus run on the
  build it started on (→ `.agent/deferred.md`).

## Score hygiene — before the smoother, always (`keypoint_hygiene.py`)

- **An out-of-frame keypoint is not an observation**: score 0 outside `[0, w) × [0, h)` of the
  decoded frame. The R stage counts every score > 0 as evidence, and the shipped generation
  carried out-of-frame shoulders on 42.3 % of above-view observations and faces on 67.2 % of
  left-view ones. Shoulder, elbow and hand features lose those frames; that loss is the fix.
- **A duplicate hand = the hidden hand drawn on the visible one**: overlap < 0.3 of the larger
  extent AND weaker mean < 0.5 AND weaker < 0.6 × stronger → the weaker hand's 21 scores are 0.
  Measured 7-12 % of both-hands frames; 15 sampled firings in 8 clips all showed one visible hand.
- **Hygiene runs on the tracker output, before the smoother** — after it, a zeroed point is
  smoothed toward instead of held (a negative control reds exactly that). Tracker state keeps
  the raw scores. On `--tracker rtmlib` the argmax subject pick reads the hygienic scores.
- **Face landmarks read iBUG-300W from the subject's own side** (MP 1 ← 42, 3 ← 45, 4 ← 39,
  6 ← 36, 9 ← 54, 10 ← 48). The mirrored table put eye points opposite the body eyes in 98-99 %
  of 11 315 frames.
- **Occlusion hallucination in general is not score-separable** — right-view knee median 0.48
  against wrist q25 0.49 — so no global confidence floor exists. For seated subjects the corpus
  drops what the camera cannot see (user ruling): knees, ankles and feet everywhere, which no
  feature reads, and the hips under the overhead camera, which moves trunk lean + rotation to the
  side views (posture symmetry reads shoulders alone) (`--drop-lower-body --drop-hips-camera above`); the rest stays a model limit. A row
  left with no positive score is carried by the smoother, not held — at visibility 0 either way.

## `PoseTracker` (`--tracker rtmlib`) — stateful, unsound; kept for comparison runs

- **`run.py --tracker rtmlib` constructs it with `tracking=False` (M2.8.2 D01). Never restore `tracking=True`.** The IoU branch reorders the CURRENT frame's keypoints by PERSISTENT track id — `keypoints = np.array([keypoints[i] for i in self.track_ids_last_frame])` — while `track_by_iou` mints `track_id = next_id++` for any unmatched box above `MIN_AREA = 1000`. One missed match indexes a one-person array at `[1]`, raises `IndexError`, and hits a bare `except` that returns **before `frame_cnt += 1` and before `bboxes_last_frame` is replaced**. Both freeze for the rest of the source, permanently, and the pre-reorder keypoints still return, so yield stays ~0.99 and nothing downstream looks broken.
- **The residue of the frozen counter picks which failure you get, and one is silent data corruption.** At `det_frequency = 7`: residue 0 → the detector re-runs every frame (correct output, ~6× cost); residue ≠ 0 → the detector never runs again, `track_by_iou`'s pops drain `bboxes_last_frame` to empty, and `RTMPose.__call__` opens with `if len(bboxes) == 0: bboxes = [[0, 0, w, h]]` — a top-down pose model estimating from the **whole 1080p frame** instead of a person crop, at confident-looking scores.
- **This is what M2.8.1's ~40× bimodality was**; both of that unit's candidate causes are refuted — not per-detected-box cost, not device placement. Measured bands 7.2-12.2 and 338.6-543.5 ms/frame; synthetic repro over 140 frames returns frozen-at-3 → 1 detector call / 135 whole-frame pose calls / 10.5 ms, frozen-at-7 → 134 detector calls / 343.0 ms, `tracking=False` → 20 calls / 0 / 58.0 ms on both stimuli. `scripts/probe_tracker_freeze.py`, 7 verdicts, rc=0, no corpus needed.
- **The fix is a removal, not a patch, because the tracker's output was already redundant.** `KeypointSmoother` owns temporal association through Hungarian `gated_assignment` (`src/pose_estimation/smoothing.py`); rtmlib's tracker contributed only the IoU drop of unmatched people, which `--single-subject` overrides by taking the confidence argmax.
- `det_frequency=1` still routes `tracking=False` down the IoU branch (the stateless guard is `not self.tracking and self.det_frequency != 1`), but a freeze is harmless there: `frame_cnt % 1` is 0 at every residue, so the detector runs every frame and the box list is never starved.
- **Fatal for *sampled* frames**: seconds-apart samples share no IoU, so the list empties permanently and any count taken from it reads 0. M2.3's detectability run published `detect_rate` median 0.0 over 379 assets from exactly this, while the detector saw a subject in 24/24 probe frames; one tracker instance reused across assets compounds it. **Sampled-frame analysis drives `det_model` + `pose_model` per frame and takes counts from the detector return.** Reserve `PoseTracker` for consecutive video (`run.py:758`, rtmlib's intended mode; its own default `tracking=True` still drops a person whose frame-to-frame IoU falls under 0.3).
- **Any pre-M2.8.2 output from `run.py` was produced under the freeze** and is suspect per asset, not per run — including `output/rtmw-l_body_single/`. M2.3's re-measured `detect_rate` is median 1.0 (mean 0.989886, min 0.333333, n=379).
- rtmlib defaults worth knowing: `det_frequency=1`, `tracking=True`, `tracking_thr=0.3`, `backend='onnxruntime'`, `device='cpu'`.

## The box feedback loop — the 2D instability's cause

- **Between detector calls the box is pose-derived, not stale.** `PoseTracker.__call__` ends with
  `self.bboxes_last_frame = bboxes_current_frame`, and in the stateless branch every entry there is
  `pose_to_bbox(kpts)`. So the crop follows the pose frame to frame — a closed pose→box→crop→pose
  loop with no external reference — and every `det_frequency` frames the detector injects one
  external correction (`bboxes = self.det_model(image)`). **Mixing the two box sources is the
  instability.** Reading the between-frames box as frozen predicts the opposite repair and is wrong.
- **A disagreement replaces the skeleton, it does not move it.** Whole-skeleton relocations measure
  509 px of centroid translation **and** 321 px of shape change with translation removed, against
  2.0/5.1 px for ordinary motion. A top-down model re-estimating inside a jumped crop returns a
  different pose, which is why no intra-skeleton predicate helps.
- **The artifact is periodic at `fps/det_frequency` and raising the value moves it rather than
  removing it.** Peak-over-background at the cadence, median over ~75-82 tracks: f3 1.814 @ 10 Hz ·
  f7 1.325 @ 4.29 Hz · f21 1.257 @ 1.43 Hz · f14 1.122 @ 2.14 Hz · f35 1.119 @ 0.86 Hz, against a
  6.7 Hz control reading 0.781-1.162 on every arm. **`det_frequency=7` is the worst available
  choice**: 4.29 Hz sits inside the clinical 2-5 Hz band and inside the 1.9-5.8 Hz intention-tremor
  band, so the artifact is indistinguishable from the signal the features measure.
- **The det_frequency curve is non-monotone because 1 runs different code.** The stateless guard is
  `not self.tracking and self.det_frequency != 1`, so at 1 the box is always the detector's and the
  loop never runs at all. Measured quality is therefore best at 1, **worst at 2** — where the crop
  alternates every other frame — and improves again upward. Never interpolate across that seam.
- **Every temporal stage downstream is keyed on track identity, so one break disarms them all.**
  `OneEuroFilter` clamps only the surprise term `|diff - dx_prev*dt|` to `outlier_cap` = 30 px and
  lets `dx_prev*dt` through unclamped; `BoneLengthSmoother` allows `tolerance` = 0.4 (40 % length
  slack) and learns per-`body_id` averages at `alpha` = 0.05, so a fresh track has no proportions to
  violate for ~20 frames. Tightening any one of them in isolation cannot reach a track-birth event.
- **The landmark export carries no track id** (304 columns; `person_idx` is 0 throughout under
  `--single-subject`), so attributing a relocation to the track lifecycle needs pipeline
  instrumentation. No query over published data can do it.
