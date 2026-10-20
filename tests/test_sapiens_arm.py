"""Synthetic controls for the Sapiens heatmap arm: UDP crop, DARK-UDP decode, flip TTA, registry."""

import numpy as np
import pytest

from pose_estimation.bench import arms


def test_screen_only_arms_stay_out_of_the_runtime_registry():
    assert not arms.SCREEN_ONLY.keys() & arms.POSE_ARMS.keys()
    assert arms.SCREENED_ARMS == arms.POSE_ARMS | arms.SCREEN_ONLY
    for size in ("0.3b", "0.6b", "1b", "2b"):
        flip, noflip = (
            arms.SCREEN_ONLY[f"sapiens-{size}"],
            arms.SCREEN_ONLY[f"sapiens-{size}-noflip"],
        )
        assert (flip.kind, flip.flip_test, flip.input_size) == ("heatmap", True, (768, 1024))
        assert (noflip.flip_test, noflip.url) == (False, flip.url)
        assert flip.url == f"model/sapiens/sapiens-{size}-f16.xml"


def test_udp_crop_geometry_and_rgb():
    from pose_estimation.bench.heatmap_arm import crop

    image = np.zeros((240, 320, 3), np.uint8)
    image[:] = (10, 30, 70)
    tensor, center, scale = crop(image, (100, 40, 180, 200))
    np.testing.assert_array_equal(center, (140, 120))
    # 80x160 box * 1.25 padding, widened to the 3:4 input aspect.
    np.testing.assert_array_equal(scale, (150, 200))
    assert tensor.shape == (1, 3, 1024, 768)
    assert tensor.dtype == np.float32
    np.testing.assert_allclose(
        tensor[0, :, 512, 384],
        (np.array([70, 30, 10]) - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375],
        atol=1e-6,
    )
    with pytest.raises(ValueError, match="box"):
        crop(image, (10, 20, 10, 50))


def test_udp_decode_subpixel_gaussian_and_score():
    from pose_estimation.bench.heatmap_arm import decode

    yy, xx = np.mgrid[:256, :192]
    x, y = 72.35, 110.65
    heatmaps = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 6**2)).astype(np.float32)
    heatmaps = np.repeat(heatmaps[None], 133, axis=0)
    saved = heatmaps.copy()
    center, scale = np.array([150, 200]), np.array([300, 400])
    kps, scores = decode(heatmaps, center, scale)
    np.testing.assert_allclose(kps, np.tile([x / 191 * 300, y / 255 * 400], (133, 1)), atol=0.01)
    np.testing.assert_array_equal(scores, np.full(133, saved.max()))
    np.testing.assert_array_equal(heatmaps, saved)


def test_flip_is_anatomical_involution():
    from pose_estimation.bench.heatmap_arm import FLIP_INDICES, flip_heatmaps

    heatmaps = np.arange(133 * 3 * 4).reshape(133, 3, 4)
    np.testing.assert_array_equal(flip_heatmaps(flip_heatmaps(heatmaps)), heatmaps)
    # left hand root (91) ↔ right hand root (112), mirrored in x
    np.testing.assert_array_equal(flip_heatmaps(heatmaps)[91], heatmaps[112, :, ::-1])
    np.testing.assert_array_equal(flip_heatmaps(heatmaps)[5], heatmaps[6, :, ::-1])
    assert sorted(FLIP_INDICES) == list(range(133))


def test_flip_tta_restores_anatomy_before_averaging(monkeypatch):
    """The flipped pass sees a mirrored person: its left wrist is labelled right at the mirrored x.
    Averaging is right only after the anatomical swap + mirror → peaks coincide at full height."""
    from pose_estimation.bench import heatmap_arm

    yy, xx = np.mgrid[:256, :192]

    def peak(x):
        return np.exp(-((xx - x) ** 2 + (yy - 128) ** 2) / 72).astype(np.float32)

    def fake_heatmaps(self, tensor):
        flipped = tensor[0, 0, 512, 200] < 0  # bright left image half reaches column 200 upright
        maps = np.zeros((133, 256, 192), np.float32)
        # upright: left wrist (9) at x=40, right wrist (10) at x=120
        maps[9], maps[10] = (peak(191 - 120), peak(191 - 40)) if flipped else (peak(40), peak(120))
        return maps

    model = object.__new__(heatmap_arm.HeatmapArm)
    model.flip_test = True
    monkeypatch.setattr(heatmap_arm.HeatmapArm, "heatmaps", fake_heatmaps)
    image = np.zeros((400, 300, 3), np.uint8)
    image[:, :150] = 255
    kps, scores = model(image, [(0, 0, 300, 400)])
    _, center, scale = heatmap_arm.crop(image, (0, 0, 300, 400))
    for joint, x in ((9, 40), (10, 120)):
        assert abs(kps[0, joint, 0] - (x / 191 * scale[0] + center[0] - scale[0] / 2)) < 1.0
        assert scores[0, joint] > 0.99


def test_crop_samples_bilinearly():
    """Upstream TopdownAffine warps with INTER_LINEAR: on a 4-per-pixel ramp a bilinear sample
    lands within 1 of 4·x; a nearest sample steps by up to 2."""
    from pose_estimation.bench.heatmap_arm import INPUT_SIZE, MEAN, STD, crop

    image = np.repeat((4 * np.arange(64, dtype=np.uint8))[None, :, None], 48, axis=0)
    image = np.repeat(image, 3, axis=2)
    tensor, center, scale = crop(image, (10, 5, 50, 45))
    factor = (INPUT_SIZE[0] - 1) / scale[0]
    cols = np.arange(100, 700, 7)
    x_src = cols / factor - scale[0] / 2 + center[0]
    inside = (x_src > 0) & (x_src < 63)
    got = tensor[0, 0, 512, cols[inside]] * STD[0] + MEAN[0]
    assert inside.sum() > 50
    assert np.abs(got - 4 * x_src[inside]).max() <= 1.0
