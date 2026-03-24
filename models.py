"""Model downloading, conversion, and compilation."""

from pathlib import Path

import openvino as ov
import openvino.properties.hint as hints

MODEL_URLS = {
    "pose_detection": "https://storage.googleapis.com/mediapipe-assets/pose_detection.tflite",
    "pose_landmark": "https://storage.googleapis.com/mediapipe-assets/pose_landmark_heavy.tflite",
    "palm_detection": "https://storage.googleapis.com/mediapipe-assets/palm_detection_full.tflite",
    "hand_landmark": "https://storage.googleapis.com/mediapipe-assets/hand_landmark_full.tflite",
}


def download_file(url, filepath):
    """Download a file if it doesn't already exist."""
    import requests
    from tqdm import tqdm

    filepath = Path(filepath)
    if filepath.exists():
        return filepath

    filepath.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total = int(response.headers.get("Content-length", 0))

    with tqdm(total=total, unit="B", unit_scale=True, desc=filepath.name) as pbar:
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(16384):
                f.write(chunk)
                pbar.update(len(chunk))

    return filepath


def download_and_compile_models(model_dir="model", device="NPU"):
    """Download TFLite models, convert to OpenVINO IR, and compile.

    Returns a dict mapping model name to compiled model:
        {"pose_detection", "pose_landmark", "palm_detection", "hand_landmark"}
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    ir_files = {}
    for name, url in MODEL_URLS.items():
        tflite_name = Path(url).name
        tflite_path = model_dir / tflite_name
        download_file(url, tflite_path)

        ir_path = tflite_path.with_suffix(".xml")
        if not ir_path.exists():
            print(f"Converting {tflite_name} to OpenVINO IR...")
            ov_model = ov.convert_model(tflite_path)
            ov.save_model(ov_model, ir_path)
        ir_files[name] = ir_path

    core = ov.Core()
    config = {hints.performance_mode(): hints.PerformanceMode.LATENCY}

    compiled = {}
    for name, ir_path in ir_files.items():
        compiled[name] = core.compile_model(
            model=core.read_model(ir_path), device_name=device, config=config,
        )

    print(f"All models compiled for {device}.")
    return compiled
