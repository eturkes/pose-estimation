# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "torch==2.14.1", "openvino==2026.4.1", "huggingface-hub>=0.27",
#   "numpy==2.4.6", "opencv-python-headless==4.13.0.92",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Convert Meta Sapiens COCO-WholeBody checkpoints without mmpose or CUDA.

  uv run --script scripts/convert/sapiens.py --size 1b --out model/sapiens
  uv run --script scripts/convert/sapiens.py --size 1b --out model/sapiens \
      --stage parity --images /path/to/val2017 --annotations /path/to/coco_wholebody_val_v1.0.json

HF repository requires access to noahcao/sapiens-pose-coco; set HF_TOKEN after accepting its terms.
Architecture: facebookresearch/sapiens (Meta, CC-BY-NC-4.0),
pretrain/mmpretrain/models/backbones/vision_transformer.py and
pose/configs/sapiens_pose/coco_wholebody/sapiens_{size}-210e_coco_wholebody-1024x768.py.
Parity isolates export error; public AP is the independent architecture/decode sanity check.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Keep downloaded checkpoint directories outside Python's import path, including under uv.
if __name__ == "__main__" and not sys.flags.isolated:
    os.execv(sys.executable, [sys.executable, "-I", str(Path(__file__).resolve()), *sys.argv[1:]])  # noqa: S606 — restart this script with the same interpreter

import cv2
import numpy as np
import openvino as ov
import torch
from torch import nn
from torch.nn import functional

ARCH = {
    "0.3b": (1024, 24, 16),
    "0.6b": (1280, 32, 16),
    "1b": (1536, 40, 24),
    "2b": (1920, 48, 32),
}
CHECKPOINT_AP = {"0.3b": "620", "0.6b": "695", "1b": "727", "2b": "745"}
HF_REPO = "noahcao/sapiens-pose-coco"
HF_REVISION = "b52e92075c8a4661dfcf2a5cd3285f2909225492"


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        x = functional.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.proj(x.transpose(1, 2).reshape(b, n, d))


class FFN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU()), nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        return self.layers(x)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.ln2 = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = FFN(dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class PatchEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.projection = nn.Conv2d(3, dim, 16, stride=16, padding=2)

    def forward(self, x):
        return self.projection(x).flatten(2).transpose(1, 2)


class Backbone(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int):
        super().__init__()
        self.patch_embed = PatchEmbed(dim)
        self.pos_embed = nn.Parameter(torch.empty(1, 64 * 48, dim))
        self.layers = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.ln1 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed
        for block in self.layers:
            x = block(x)
        return self.ln1(x).transpose(1, 2).reshape(x.shape[0], -1, 64, 48)


class Head(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        deconv = []
        for incoming in (dim, 768):
            deconv += [
                nn.ConvTranspose2d(incoming, 768, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(768),
                nn.SiLU(),
            ]
        self.deconv_layers = nn.Sequential(*deconv)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(768, 768, 1),
            nn.InstanceNorm2d(768),
            nn.SiLU(),
            nn.Conv2d(768, 768, 1),
            nn.InstanceNorm2d(768),
            nn.SiLU(),
        )
        self.final_layer = nn.Conv2d(768, 133, 1)

    def forward(self, x):
        return self.final_layer(self.conv_layers(self.deconv_layers(x)))


class Sapiens(nn.Module):
    def __init__(self, size: str):
        super().__init__()
        dim, depth, heads = ARCH[size]
        self.backbone = Backbone(dim, depth, heads)
        self.head = Head(dim)

    def forward(self, x):
        return self.head(self.backbone(x))


def checkpoint(size: str, out: Path) -> Path:
    from huggingface_hub import hf_hub_download

    name = f"sapiens_{size}_coco_wholebody_best_coco_wholebody_AP_{CHECKPOINT_AP[size]}.pth"
    return Path(
        hf_hub_download(
            HF_REPO,
            f"sapiens_host/pose/checkpoints/sapiens_{size}/{name}",
            revision=HF_REVISION,
            local_dir=out / f"checkpoint-{size}",
        )
    )


def load(size: str, path: Path) -> Sapiens:
    # mmap + meta/assign keep one weight copy even for the 2B checkpoint. The mmengine checkpoints
    # pickle numpy arrays in their metadata → allowlist exactly those globals (never a full unpickle).
    numpy_globals = [
        (np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        np.ndarray,
        np.dtype,
        *(type(np.dtype(c)) for c in "?bBhHiIlLqQefdgFDGSUVOMm"),
        (__import__("codecs").encode, "_codecs.encode"),
    ]
    with torch.serialization.safe_globals(numpy_globals):
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = state.get("state_dict", state)
    with torch.device("meta"):
        model = Sapiens(size)
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval().requires_grad_(False)


def runtime():
    import importlib.util

    # Import this leaf alone: package __init__ pulls in the production pipeline's dependencies.
    path = Path(__file__).resolve().parents[2] / "src/pose_estimation/bench/heatmap_arm.py"
    spec = importlib.util.spec_from_file_location("sapiens_heatmap_runtime", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensors(images: Path, annotations: Path, count: int, seed: int):
    data = json.loads(annotations.read_text())
    files = {r["id"]: r["file_name"] for r in data["images"]}
    rows = [
        r
        for r in data["annotations"]
        if not r.get("iscrowd") and r["bbox"][2] > 1 and r["bbox"][3] > 1
    ]
    indices = np.random.default_rng(seed).permutation(len(rows))[:count]
    if len(indices) != count:
        raise ValueError("not enough non-crowd positive-area person crops")
    api = runtime()
    for i in indices:
        row = rows[i]
        image = cv2.imread(str(images / files[row["image_id"]]))
        if image is None:
            raise FileNotFoundError(images / files[row["image_id"]])
        x, y, w, h = row["bbox"]
        tensor, center, scale = api.crop(image, (x, y, x + w, y + h))
        yield row["id"], tensor, center, scale


def export(size: str, path: Path, out: Path):
    model = load(size, path)
    n = sum(p.numel() for p in model.parameters())
    print(f"strict load: sapiens-{size} parameters={n}", flush=True)
    with torch.inference_mode():
        traced = torch.jit.trace(model, torch.zeros(1, 3, 1024, 768), check_trace=False)
        converted = ov.convert_model(traced, input=[1, 3, 1024, 768], share_weights=True)
    for precision in ("f32", "f16"):
        target = out / f"sapiens-{size}-{precision}.xml"
        ov.save_model(converted, target, compress_to_fp16=precision == "f16")
        print(f"exported {target}", flush=True)
    return {"parameters": n, "strict_load": True, "checkpoint": str(path)}


def _compile(a, precision: str, device: str):
    config = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": precision}
    if device == "CPU":
        config["INFERENCE_NUM_THREADS"] = a.threads
    return ov.Core().compile_model(str(a.out / f"sapiens-{a.size}-{precision}.xml"), device, config)


def parity(a):
    """20 COCO crops, no flip: PyTorch f32 reference vs the f32 IR on CPU + the f16 IR on GPU."""
    api = runtime()
    reference_path = a.out / f"sapiens-{a.size}-torch-reference.npz"
    crops = list(tensors(a.images, a.annotations, 20, a.seed))
    digests = [hashlib.sha256(t.tobytes()).hexdigest() for _, t, _, _ in crops]
    metadata = json.dumps(
        {
            "size": a.size,
            "seed": a.seed,
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__), Path(api.__file__))
            },
            "annotations_sha256": hashlib.sha256(a.annotations.read_bytes()).hexdigest(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
        sort_keys=True,
    )
    ref = None
    if reference_path.is_file():
        with np.load(reference_path) as z:
            if str(z["metadata"]) == metadata and list(z["input_sha256"]) == digests:
                ref = (z["keypoints"], z["scores"])
    if ref is None:
        model = load(a.size, a.checkpoint or checkpoint(a.size, a.out))
        kps, scores = [], []
        with torch.inference_mode():
            for n, (_, tensor, center, scale) in enumerate(crops, 1):
                xy, sc = api.decode(model(torch.from_numpy(tensor)).numpy()[0], center, scale)
                kps.append(xy)
                scores.append(sc)
                print(f"PyTorch reference {n}/20", flush=True)
        del model
        gc.collect()
        ref = (np.asarray(kps), np.asarray(scores))
        np.savez_compressed(
            reference_path,
            annotation_ids=[c[0] for c in crops],
            keypoints=ref[0],
            scores=ref[1],
            input_sha256=digests,
            metadata=metadata,
        )
    result = {"crops": len(crops), "flip_test": False}
    for precision, device in (("f32", "CPU"), ("f16", "GPU")):
        compiled = _compile(a, precision, device)
        xy_delta, score_delta, candidates = [], [], []
        for i, (_, tensor, center, scale) in enumerate(crops):
            xy, sc = api.decode(compiled([tensor])[compiled.output(0)][0], center, scale)
            candidates.append(xy)
            xy_delta.extend(np.linalg.norm(xy - ref[0][i], axis=1).tolist())
            score_delta.extend(np.abs(sc - ref[1][i]).tolist())
        np.savez_compressed(
            a.out / f"sapiens-{a.size}-parity-{precision}-{device.lower()}.npz",
            reference_xy=ref[0],
            candidate_xy=np.asarray(candidates),
            metadata=metadata,
        )
        result[f"{precision}-{device}"] = {
            "keypoints": len(xy_delta),
            "keypoint_delta_px": {
                "median": float(np.median(xy_delta)),
                "p95": float(np.quantile(xy_delta, 0.95)),
                "p99": float(np.quantile(xy_delta, 0.99)),
                "max": max(xy_delta),
                "over_1px": int(np.sum(np.asarray(xy_delta) > 1)),
            },
            "score_delta": {"median": float(np.median(score_delta)), "max": max(score_delta)},
            "execution_devices": compiled.get_property("EXECUTION_DEVICES"),
        }
        print(json.dumps(result[f"{precision}-{device}"]), flush=True)
        del compiled
        gc.collect()
    return result


def latency(a):
    tensor = next(tensors(a.images, a.annotations, 1, a.seed))[1]
    stats = {}
    for precision in ("f32", "f16"):
        compiled = _compile(a, precision, "GPU")
        for _ in range(3):
            compiled([tensor])
        times = []
        for _ in range(50):
            start = time.perf_counter()
            compiled([tensor])
            times.append(time.perf_counter() - start)
        stats[precision] = {
            "calls": len(times),
            "warmup": 3,
            "flip_test": False,
            "median_ms": float(np.median(times) * 1000),
            "execution_devices": compiled.get_property("EXECUTION_DEVICES"),
        }
        print(json.dumps(stats[precision]), flush=True)
        del compiled
        gc.collect()
    return stats


def npu(a):
    try:
        compiled = ov.Core().compile_model(str(a.out / f"sapiens-{a.size}-f16.xml"), "NPU")
        return {"compiled": True, "execution_devices": compiled.get_property("EXECUTION_DEVICES")}
    except RuntimeError as error:
        return {"compiled": False, "error": str(error)}


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--size", choices=ARCH, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument(
        "--stage",
        choices=("export-parity", "export", "parity", "latency", "npu"),
        default="export-parity",
    )
    p.add_argument("--images", type=Path, default=Path(".scratch/data/coco/val2017"))
    p.add_argument(
        "--annotations",
        type=Path,
        default=Path(".scratch/data/cocowb/coco_wholebody_val_v1.0.json"),
    )
    p.add_argument("--seed", type=int, default=20261019)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    if a.stage in ("export-parity", "parity", "latency") and (
        not a.images.is_dir() or not a.annotations.is_file()
    ):
        p.error("Set --images and --annotations to the public COCO-WholeBody validation data")
    start = time.perf_counter()
    if a.stage in ("export-parity", "export"):
        result = export(a.size, a.checkpoint or checkpoint(a.size, a.out), a.out)
        if a.stage == "export-parity":
            gc.collect()
            result = {"export": result, "parity": parity(a)}
    else:
        result = {"parity": parity, "latency": latency, "npu": npu}[a.stage](a)
    report = {
        "size": a.size,
        "stage": a.stage,
        "torch": torch.__version__,
        "openvino": ov.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "wall_s": time.perf_counter() - start,
        "result": result,
    }
    (a.out / f"{a.size}-{a.stage}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
