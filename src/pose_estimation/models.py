"""Model downloading, conversion, and compilation."""

import hashlib
from pathlib import Path

import openvino as ov
import openvino.properties.hint as hints

MODEL_URLS = {
    "pose_detection": "https://storage.googleapis.com/mediapipe-assets/pose_detection.tflite",
    "pose_landmark": "https://storage.googleapis.com/mediapipe-assets/pose_landmark_heavy.tflite",
    "palm_detection": "https://storage.googleapis.com/mediapipe-assets/palm_detection_full.tflite",
    "hand_landmark": "https://storage.googleapis.com/mediapipe-assets/hand_landmark_full.tflite",
}

# SHA-256 of the canonical MediaPipe TFLite assets known to work with this
# pipeline.  When set, downloaded files are verified after fetch and
# re-downloaded once on mismatch.  Set the value to ``None`` to skip
# verification for an entry (e.g. if upstream replaces the asset).
MODEL_SHA256 = {
    "pose_detection": "9ba9dd3d42efaaba86b4ff0122b06f29c4122e756b329d89dca1e297fd8f866c",
    "pose_landmark": "59e42d71bcd44cbdbabc419f0ff76686595fd265419566bd4009ef703ea8e1fe",
    "palm_detection": "1b14e9422c6ad006cde6581a46c8b90dd573c07ab7f3934b5589e7cea3f89a54",
    "hand_landmark": "11c272b891e1a99ab034208e23937a8008388cf11ed2a9d776ed3d01d0ba00e3",
}

_HASH_BUFFER = 1 << 20  # 1 MiB


def _sha256_of(path):
    """Return the SHA-256 hex digest of *path*."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_BUFFER), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_checksum(path, expected, *, redownload):
    """Verify *path* matches *expected*; optionally delete on mismatch.

    When *expected* is None, returns True (verification disabled).
    Returns True if the checksum matches; False if it mismatched and
    *redownload* is True (caller should re-fetch).  Raises ValueError
    when *redownload* is False and the file is corrupt.
    """
    if expected is None:
        return True
    actual = _sha256_of(path)
    if actual == expected:
        return True
    if redownload:
        print(
            f"  WARNING: checksum mismatch for {Path(path).name} "
            f"(expected {expected[:12]}…, got {actual[:12]}…). Removing for re-download."
        )
        Path(path).unlink(missing_ok=True)
        return False
    raise ValueError(
        f"Checksum mismatch for {path}: expected {expected}, got {actual}"
    )


def _download_to(url, filepath):
    """Stream *url* into *filepath* with a progress bar."""
    import requests
    from tqdm import tqdm

    filepath.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total = int(response.headers.get("Content-length", 0))

    with (
        tqdm(total=total, unit="B", unit_scale=True, desc=filepath.name) as pbar,
        filepath.open("wb") as f,
    ):
        for chunk in response.iter_content(16384):
            f.write(chunk)
            pbar.update(len(chunk))


def download_file(url, filepath, expected_sha256=None):
    """Download a file if it doesn't already exist, verifying checksum.

    If a cached file fails the SHA-256 check, it is removed and
    downloaded once more.  A second mismatch raises ValueError.
    """
    filepath = Path(filepath)
    # Cached fast path: existing file passes verification and we're done.
    if filepath.exists() and _verify_checksum(filepath, expected_sha256, redownload=True):
        return filepath
    # Either no cache, or the cached file failed verification (now removed).
    _download_to(url, filepath)
    _verify_checksum(filepath, expected_sha256, redownload=False)
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
        download_file(url, tflite_path, expected_sha256=MODEL_SHA256.get(name))

        ir_path = tflite_path.with_suffix(".xml")
        if not ir_path.exists():
            print(f"Converting {tflite_name} to OpenVINO IR...")
            ov_model = ov.convert_model(tflite_path)
            ov.save_model(ov_model, ir_path)
        ir_files[name] = ir_path

    core = ov.Core()
    # openvino exposes `performance_mode()` via __getattr__ (ty can't see it)
    config = {hints.performance_mode(): hints.PerformanceMode.LATENCY}  # ty: ignore[unresolved-attribute]

    compiled = {}
    for name, ir_path in ir_files.items():
        compiled[name] = core.compile_model(
            model=core.read_model(ir_path),
            device_name=device,
            config=config,
        )

    print(f"All models compiled for {device}.")
    return compiled
