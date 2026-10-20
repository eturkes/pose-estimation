---
paths:
  - "scripts/convert/sapiens.py"
  - "src/pose_estimation/bench/heatmap_arm.py"
  - "tests/test_sapiens_arm.py"
---

# Sapiens v1 screening arms

- Weights = gated HF repo `noahcao/sapiens-pose-coco` (CC BY-NC 4.0; access granted to the
  `HF_TOKEN` account), revision pinned in `scripts/convert/sapiens.py`; IR + checkpoints under
  gitignored `model/sapiens/`. The converter is a uv `--script` with its own pinned env (torch CPU,
  openvino wheel) and re-executes under `python -I`, so a downloaded directory never shadows an
  import.
- The mmengine checkpoints pickle numpy metadata → `torch.load(weights_only=True)` under an explicit
  allowlist (numpy `_reconstruct`/`ndarray`/dtypes, `_codecs.encode`). Never `weights_only=False`:
  it executes arbitrary pickle code from a third-party file.
- Parity (`--stage parity`) grades the export alone: reference and candidate share
  `heatmap_arm.crop` + `decode`, so a crop or decode defect passes it. `tests/test_sapiens_arm.py`
  pins crop geometry, bilinear sampling, DARK-UDP decode and the flip TTA's anatomical swap +
  mirror; public AP against the published table checks the whole.
- Screening precision = the f16 IR at `INFERENCE_PRECISION_HINT=f16` on GPU: GPU f32 has no XMX
  path (0.3B: 3688 vs 337 ms/call). NPU compiles 0.3B (263 s) yet runs 659 ms/call, slower than GPU.
- Screening-only arms live in `arms.SCREEN_ONLY` (`sapiens-<size>[-noflip]`), never `POSE_ARMS`:
  `test_a03_bench_arms_are_runtime_registry_entries` pins `POSE_ARMS` = runtime registry keys.
