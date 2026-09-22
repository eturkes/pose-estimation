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
- **The padding is a device property, and only NPU pads with garbage.** One synthetic probe separates the three devices with no patient data: read `yolox_m_8xb8-300e_humanart-c2c7a14a.onnx` (rtmlib cache), compile per device, infer `np.zeros((1,3,640,640))`, print `dets`/`labels` shape + range. CPU keeps the dynamic output (`(1,1,5)`); GPU and NPU both materialise a fixed 100-row buffer with `labels` all `-1`; only NPU's `dets` hold uninitialised memory (`−0.1471…1.0215` on an all-zero image, so scores clear any threshold) while GPU's read exactly `0.0000`. Median latency: GPU 9.7 ms, NPU 108.4 ms, CPU 213.1 ms. That keeps NPU excluded and makes **GPU a live candidate** — still unqualified against real detections, so `--det-device GPU` is deferred, not adopted (`.agent/deferred.md`).
- Pose models are NPU-safe: RTMW-L NPU vs CPU = 0.505 px mean / 2.265 px p95 / 5.3 px p99 keypoint deviation, score MAE 0.00056. Per-call 7.17 ms NPU vs 134.26 ms CPU (~19×); detector 109.82 ms NPU (garbage) vs 445.21 ms CPU.
- **Those per-call numbers do not predict run throughput — measured end to end they are 4-5× optimistic.** The projection they supported (≈70 ms/frame ⇒ 6.5 h for the corpus) survived planning and two unit windows unchallenged; the pilot measured 3.03 fps over 8971 frames ⇒ 26.0-30.9 h. **Never size a run from per-call latency; run a stratified pilot and multiply.**
- MediaPipe is unaffected — SSD anchors + NMS decode in Python (`detection.py`), graphs stay static-shaped ⇒ both roles default NPU. `models.DETECTOR_MODELS` selects which compile on `--det-device`.
- Both call sites print the **requested** device rather than `compiled_model.get_property("EXECUTION_DEVICES")` — truthful only while exact device names are passed (`.agent/archive/polish.md`).

## `PoseTracker` — stateful, unsound, DISABLED in the run path

- **`run.py` constructs it with `tracking=False` (M2.8.2 D01). Never restore `tracking=True`.** The IoU branch reorders the CURRENT frame's keypoints by PERSISTENT track id — `keypoints = np.array([keypoints[i] for i in self.track_ids_last_frame])` — while `track_by_iou` mints `track_id = next_id++` for any unmatched box above `MIN_AREA = 1000`. One missed match indexes a one-person array at `[1]`, raises `IndexError`, and hits a bare `except` that returns **before `frame_cnt += 1` and before `bboxes_last_frame` is replaced**. Both freeze for the rest of the source, permanently, and the pre-reorder keypoints still return, so yield stays ~0.99 and nothing downstream looks broken.
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
