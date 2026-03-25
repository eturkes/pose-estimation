"""Test which rtmlib ONNX models can compile on NPU.

Downloads the ONNX models used by rtmlib, reshapes them to static
shapes (batch=1), and attempts to compile on NPU via OpenVINO.
Reports success/failure per model.

Usage:
    python test_npu_compat.py
    python test_npu_compat.py --device NPU   # default
    python test_npu_compat.py --device GPU    # sanity check
"""

import argparse
import sys
import time
from pathlib import Path

# (name, url, input_shape) — input_shape is (B, C, H, W) with B=1
MODELS = {
    # ── Detectors ────────────────────────────────────────────────
    "yolox-nano (416)": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "yolox_nano_8xb8-300e_humanart-40f6f0d0.zip",
        (1, 3, 416, 416),
    ),
    "yolox-s (640)": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "yolox_s_8xb8-300e_humanart-3ef259a7.zip",
        (1, 3, 640, 640),
    ),
    "yolox-m (640)": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        (1, 3, 640, 640),
    ),
    # ── Body pose (17 kps) ──────────────────────────────────────
    "rtmpose-t body (256x192)": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip",
        (1, 3, 256, 192),
    ),
    "rtmpose-s body (256x192)": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip",
        (1, 3, 256, 192),
    ),
    "rtmpose-m body (256x192)": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
        (1, 3, 256, 192),
    ),
    # ── Whole-body (133 kps) ────────────────────────────────────
    "dwpose-t wholebody (256x192)": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmpose-t_simcc-ucoco_dw-ucoco_270e-256x192-dcf277bf_20230728.zip",
        (1, 3, 256, 192),
    ),
    "dwpose-m wholebody (256x192)": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.zip",
        (1, 3, 256, 192),
    ),
    "rtmw-m wholebody (256x192)": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
        "rtmw-dw-m-s_simcc-cocktail14_270e-256x192_20231122.zip",
        (1, 3, 256, 192),
    ),
    "rtmw-l wholebody (256x192)": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
        "rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip",
        (1, 3, 256, 192),
    ),
}


def download_onnx(url: str, cache_dir: Path) -> Path:
    """Download and extract ONNX model from a .zip URL. Return .onnx path."""
    from rtmlib.tools.file import download_checkpoint
    return Path(download_checkpoint(url))


def test_compile(onnx_path: Path, static_shape: tuple, device: str):
    """Try to compile an ONNX model on the given OpenVINO device."""
    import openvino as ov

    core = ov.Core()
    model = core.read_model(str(onnx_path))

    # Reshape to static (freeze batch dimension)
    model.reshape(static_shape)

    t0 = time.perf_counter()
    compiled = core.compile_model(
        model, device_name=device,
        config={"PERFORMANCE_HINT": "LATENCY"})
    dt = time.perf_counter() - t0

    # Quick inference test with zeros
    import numpy as np
    dummy = np.zeros(static_shape, dtype=np.float32)
    result = compiled([dummy])

    return dt, [r.shape for r in result.values()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="NPU", help="OpenVINO device (default: NPU)")
    args = p.parse_args()

    print(f"Testing model compilation on {args.device}")
    print(f"{'=' * 65}")
    print()

    results = []
    for name, (url, shape) in MODELS.items():
        print(f"  {name} ...")
        sys.stdout.flush()
        try:
            onnx_path = download_onnx(url, Path("model"))
            dt, out_shapes = test_compile(onnx_path, shape, args.device)
            status = f"OK ({dt:.1f}s compile, outputs: {out_shapes})"
            results.append((name, True, status))
        except Exception as exc:
            # Truncate long error messages
            msg = str(exc)
            if len(msg) > 120:
                msg = msg[:120] + "..."
            status = f"FAIL: {msg}"
            results.append((name, False, status))
        print(f"    {status}")
        print()

    # Summary
    print(f"{'=' * 65}")
    print(f"Summary for {args.device}:")
    ok = sum(1 for _, s, _ in results if s)
    print(f"  {ok}/{len(results)} models compiled successfully")
    print()
    for name, success, _ in results:
        mark = "OK  " if success else "FAIL"
        print(f"  [{mark}] {name}")


if __name__ == "__main__":
    main()
