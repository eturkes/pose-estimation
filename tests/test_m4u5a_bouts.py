"""M4.5a P01-P09: wrist-motion bouts + per-bout features in the R producer, and its qualification."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import pathlib
import subprocess
import tempfile
from typing import Any

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CLINICAL_R = _ROOT / "analysis" / "clinical_features.R"
_FEATURES = (
    "bout_duration",
    "bout_sparc",
    "bout_normalized_jerk",
    "bout_peak_time_share",
    "bout_speed_peaks",
    "bout_path_ratio",
    "bout_dwell",
)


def _r(body: str, payload: dict) -> Any:
    """Source the producer's functions (no landmark path → usage guard), run `body`, return JSON."""
    with tempfile.TemporaryDirectory(prefix="m4u5a-") as tmp:
        request = pathlib.Path(tmp) / "request.json"
        request.write_text(json.dumps(payload), encoding="utf-8")
        script = (
            'assign("commandArgs", function(...) character(0), envir = globalenv())\n'
            f"suppressWarnings(try(source({json.dumps(str(_CLINICAL_R))}), silent = TRUE))\n"
            'rm("commandArgs", envir = globalenv())\n'
            f"req <- jsonlite::read_json({json.dumps(str(request))}, simplifyVector = FALSE)\n"
            "nums <- function(v) vapply(v, function(e) if (is.null(e)) NA_real_ else "
            "as.numeric(e), numeric(1))\n"
            + body
            + "\ncat('\\n'); jsonlite::write_json(result, stdout(), auto_unbox = TRUE, "
            "digits = 17, na = 'null')\n"
        )
        proc = subprocess.run(
            ["Rscript", "-"], input=script, cwd=_ROOT, capture_output=True, text=True, timeout=300
        )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


_DETECT = """
out <- list()
for (s in seq_along(req$series)) {
  q <- req$series[[s]]
  x <- nums(q$x); y <- nums(q$y)
  det <- detect_bouts(nums(q$t), x, y, as.integer(nums(q$frame)), q$fs)
  f <- if (is.null(det$grid) || !nrow(det$bouts)) list() else bout_features(det, q$fs)
  b <- det$bouts
  out[[s]] <- list(
    start = as.list(if (is.null(b)) numeric(0) else det$grid$t[b$start]),
    end = as.list(if (is.null(b)) numeric(0) else det$grid$t[b$end]),
    censored = as.list(if (is.null(b)) logical(0) else b$censored),
    features = lapply(f, as.list)
  )
}
result <- out
"""


def _minjerk(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0, 1)
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def _series(fs=30.0, parts=((1.0, 1.0, 0.1, 0.0),), total=4.0, phase=0.0, scale=1.0, shift=0.0):
    """Wrist path from minimum-jerk parts (t0, duration, dx, dy); rest elsewhere."""
    t = np.arange(int(total * fs)) / fs + phase / fs
    x = np.full(len(t), 0.5)
    y = np.full(len(t), 0.5)
    for t0, dur, dx, dy in parts:
        s = _minjerk((t - t0) / dur)
        x = x + dx * s
        y = y + dy * s
    return {
        "t": t.tolist(),
        "x": (x * scale + shift).tolist(),
        "y": (y * scale + shift).tolist(),
        "frame": list(range(len(t))),
        "fs": fs,
    }


def _clean_sparc(parts, fs: float) -> float:
    """The shipped kernel on the noise-free speed of the whole movement (zero to zero)."""
    s = _series(fs, parts)
    v = (np.hypot(np.diff(s["x"]), np.diff(s["y"])) * fs).tolist()
    return _r(
        "result <- list(sparc = spectral_arc_length(nums(req$v), req$fs))", {"v": v, "fs": fs}
    )["sparc"]


def _detect(*series: dict) -> list[dict]:
    return _r(_DETECT, {"series": list(series)})


def _feature(out: dict, name: str, i: int = 0):
    return out["features"][name][i]


def test_p01_single_minimum_jerk_bout():
    parts = ((1.0, 1.0, 0.1, 0.0),)
    (out,) = _detect(_series(parts=parts))
    assert len(out["start"]) == 1
    assert out["censored"] == [False]
    assert _feature(out, "bout_speed_peaks") == 1
    assert _feature(out, "bout_path_ratio") == pytest.approx(1.0, abs=1e-6)
    assert _feature(out, "bout_peak_time_share") == pytest.approx(0.5, abs=0.05)
    assert _feature(out, "bout_sparc") == pytest.approx(_clean_sparc(parts, 30.0), abs=0.02)
    assert out["start"][0] == pytest.approx(1.0, abs=0.15)
    assert out["end"][0] == pytest.approx(2.0, abs=0.15)


def test_p01_overlapping_submovements_one_bout_two_peaks_less_smooth():
    # 40 % overlap, the second part perpendicular: the resultant speed dips to ~58 % of each peak.
    two = ((1.0, 0.8, 0.1, 0.0), (1.48, 0.8, 0.0, 0.1))
    one = ((1.0, 1.28, 0.18, 0.0),)
    out_two, out_one = _detect(_series(parts=two), _series(parts=one))
    assert len(out_two["start"]) == 1
    assert _feature(out_two, "bout_speed_peaks") == 2
    assert _feature(out_two, "bout_sparc") < _feature(out_one, "bout_sparc")


def test_p01_plateau_peak_times_its_midpoint():
    payload = {"v": [0, 1, 2, 3, 3, 3, 3, 2, 1, 0], "prom": 0.2, "spacing": 4.5}
    out = _r(
        "m <- speed_maxima(nums(req$v)); result <- list(pos = m$pos[1], "
        "n = count_speed_peaks(nums(req$v), req$prom, req$spacing))",
        payload,
    )
    assert out["pos"] == 5.5  # plateau intervals 4..7 (1-based) → midpoint 5.5
    assert out["n"] == 1


def test_p02_sparc_kernel_matches_the_author_example():
    t = np.arange(-1, 1, 0.01)
    out = _r(
        "result <- list(s = spectral_arc_length(nums(req$v), 100))",
        {"v": np.exp(-5 * t**2).tolist()},
    )
    assert round(out["s"], 5) == -1.41403


def _with_hole(series: dict, start: int, length: int, drop_rows: bool) -> dict:
    s = json.loads(json.dumps(series))
    idx = range(start, start + length)
    if drop_rows:
        for key in ("t", "x", "y", "frame"):
            s[key] = [v for i, v in enumerate(s[key]) if i not in idx]
    else:
        for i in idx:
            s["x"][i] = s["y"][i] = None
    return s


def test_p03_long_hole_censors_short_hole_keeps_speed_features_only():
    base = _series(parts=((1.0, 1.5, 0.12, 0.0),), total=4.0)
    long_na, long_drop, short = _detect(
        _with_hole(base, 45, 8, drop_rows=False),
        _with_hole(base, 45, 8, drop_rows=True),
        _with_hole(base, 45, 2, drop_rows=False),
    )
    assert long_na == long_drop  # explicit NA and omitted rows read alike
    # The hole splits the one movement in two; both fragments publish, censored (an empty list
    # would pass `all` while the bounds + censoring evidence vanished).
    assert long_na["censored"] == [True, True]
    assert long_na["features"] == [] or all(
        v is None for name in _FEATURES for v in long_na["features"][name]
    )
    assert short["censored"] == [False]
    assert _feature(short, "bout_normalized_jerk") is None  # path holds a missing slot
    assert _feature(short, "bout_path_ratio") is None
    assert _feature(short, "bout_sparc") is not None
    assert _feature(short, "bout_peak_time_share") is not None


def test_p03_dwell_is_finite_within_a_segment_and_na_across_a_split():
    parts = ((1.0, 0.8, 0.1, 0.0), (2.6, 0.8, -0.1, 0.0))
    together, split = _detect(
        _series(parts=parts), _with_hole(_series(parts=parts), 63, 6, drop_rows=False)
    )
    assert len(together["start"]) == 2
    assert _feature(together, "bout_dwell") == pytest.approx(
        together["start"][1] - together["end"][0], abs=1e-9
    )
    assert _feature(together, "bout_dwell", 1) is None  # last bout
    assert all(_feature(split, "bout_dwell", i) is None for i in range(len(split["start"])))


def _glitched(series: dict, at: int, length: int, size: float) -> dict:
    s = json.loads(json.dumps(series))
    for i in range(at, at + length):
        s["x"][i] += size if i % 2 else -size
    return s


def test_p04_outlier_episode_leaves_genuine_bouts_and_fires_on_max_normaliser():
    parts = tuple((1.0 + 2.5 * k, 1.2, 0.08 if k % 2 == 0 else -0.08, 0.0) for k in range(6))
    clean = _series(parts=parts, total=17.0)
    glitched = _glitched(clean, 470, 6, 0.08)  # ~0.6 % of the series: inside the top 1 %
    body = _DETECT.replace(
        "det <- detect_bouts(nums(q$t), x, y, as.integer(nums(q$frame)), q$fs)",
        "det <- detect_bouts(nums(q$t), x, y, as.integer(nums(q$frame)), q$fs, normaliser_q = q$q)",
    )
    base, robust, maxed = _r(
        body,
        {
            "series": [
                clean | {"q": 0.95},
                glitched | {"q": 0.95},
                glitched | {"q": 1.0},
            ]
        },
    )
    genuine = [(s, e) for s, e in zip(base["start"], base["end"], strict=True)]
    kept = [
        (s, e) for s, e in zip(robust["start"], robust["end"], strict=True) if not 15.5 < s < 16.0
    ]
    assert len(genuine) == 6
    assert kept == genuine
    assert len(maxed["start"]) < len(genuine)  # NC1: the max-normalised mutant loses bouts


def _run_producer(tmp: pathlib.Path, rows: list[dict]) -> tuple[list[dict], list[dict], list[str]]:
    gen = importlib.import_module("pose_estimation.export")
    header = gen.make_csv_header("body")
    path = tmp / "clip.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict.fromkeys(header, "") | row)
    proc = subprocess.run(
        ["Rscript", str(_CLINICAL_R), str(path)], cwd=_ROOT, capture_output=True, text=True,
        timeout=300,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr[-2000:]
    with (tmp / "clip_clinical_bouts.csv").open(newline="") as h:
        reader = csv.DictReader(h)
        bouts, fields = list(reader), list(reader.fieldnames or [])
    with (tmp / "clip_clinical_bout_qc.csv").open(newline="") as h:
        qc = list(csv.DictReader(h))
    return bouts, qc, fields


def _landmark_rows(series: dict, still_right: bool = True) -> list[dict]:
    rows = []
    for i, (t, x, y) in enumerate(zip(series["t"], series["x"], series["y"], strict=True)):
        row = {
            "video": "clip.avi",
            "frame_idx": str(i),
            "timestamp_sec": f"{t:.4f}",
            "person_idx": "0",
        }
        for name in ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow"):
            row |= {
                f"body_{name}_x": "0.5",
                f"body_{name}_y": "0.4",
                f"body_{name}_z": "0",
                f"body_{name}_vis": "0.9",
            }
        row |= {"body_left_wrist_x": f"{x:.6f}", "body_left_wrist_y": f"{y:.6f}", "body_left_wrist_z": "0",
                "body_left_wrist_vis": "0.9"}  # fmt: skip
        if still_right:
            row |= {"body_right_wrist_x": "0.6", "body_right_wrist_y": "0.6", "body_right_wrist_z": "0",
                    "body_right_wrist_vis": "0.9"}  # fmt: skip
        rows.append(row)
    return rows


def test_p05_schema_dispositions_and_stale_rows(tmp_path):
    clip = _series(parts=((1.0, 1.0, 0.1, 0.0), (3.0, 1.0, -0.1, 0.0)), total=5.0)
    bouts, qc, fields = _run_producer(tmp_path, _landmark_rows(clip))
    expected = [
        "video", "person_idx", "side", "bout_idx", "start_frame", "end_frame", "start_sec",
        "end_sec", "censored",
        *(f"{side}_{f}" for side in ("left", "right") for f in _FEATURES),
    ]  # fmt: skip
    assert fields == expected
    assert [b["side"] for b in bouts] == ["left", "left"]
    assert [b["bout_idx"] for b in bouts] == ["1", "2"]
    assert all(b[f"right_{f}"] == "NA" for b in bouts for f in _FEATURES)
    assert {q["side"]: q["disposition"] for q in qc} == {"left": "bouts", "right": "no_bout"}
    finite = {f: float(bouts[0][f"left_{f}"]) for f in _FEATURES if bouts[0][f"left_{f}"] != "NA"}
    assert finite["bout_duration"] > 0
    assert finite["bout_sparc"] <= 0
    assert finite["bout_normalized_jerk"] >= 0
    assert 0 <= finite["bout_peak_time_share"] <= 1
    assert finite["bout_speed_peaks"] >= 1
    assert finite["bout_path_ratio"] >= 1
    assert finite["bout_dwell"] >= 0
    # A still rerun replaces the file with a header-only one: no stale bout survives.
    still = _series(parts=(), total=5.0)
    bouts, qc, fields = _run_producer(tmp_path, _landmark_rows(still))
    assert fields == expected
    assert bouts == []
    assert {q["disposition"] for q in qc} == {"no_bout"}


def test_p05_directory_mode_does_not_rescan_bout_artifacts(tmp_path):
    clip = _series(parts=((1.0, 1.0, 0.1, 0.0),), total=3.0)
    _run_producer(tmp_path, _landmark_rows(clip))
    proc = subprocess.run(
        ["Rscript", str(_CLINICAL_R), str(tmp_path)], cwd=_ROOT, capture_output=True, text=True,
        timeout=300,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "Clinical features: clip.csv" in proc.stdout
    assert "_bout" not in proc.stdout.split("Clinical features:", 1)[1].split("\n")[0]
    assert not list(tmp_path.glob("*_bouts_clinical*"))


def test_p06_scale_and_translation_invariance():
    parts = ((1.0, 1.0, 0.1, 0.03), (2.8, 0.9, -0.06, 0.05))
    a, b = _detect(_series(parts=parts), _series(parts=parts, scale=3.0, shift=0.2))
    assert a["start"] == b["start"]
    assert a["end"] == b["end"]
    for name in _FEATURES:
        for va, vb in zip(a["features"][name], b["features"][name], strict=True):
            assert (va is None) == (vb is None)
            if va is not None:
                assert va == pytest.approx(vb, abs=1e-9)


def test_p07_rate_consistency():
    parts = ((1.0, 1.2, 0.1, 0.0),)
    outs = _detect(*(_series(fs=fs, parts=parts, phase=0.3) for fs in (25.0, 30.0, 60.0)))
    durations = [_feature(o, "bout_duration") for o in outs]
    assert max(durations) - min(durations) <= 1 / 25 + 1e-9
    shares = [_feature(o, "bout_peak_time_share") for o in outs]
    assert max(shares) - min(shares) <= 0.05
    sparcs = [_feature(o, "bout_sparc") for o in outs]
    assert max(sparcs) - min(sparcs) <= 0.05


def _bq():
    return importlib.import_module("pose_estimation.bout_qualification")


def test_p08_banks_are_seed_determined():
    bq = _bq()

    def digest(bank):
        h = hashlib.sha256()
        for s in bank:
            for arr in (s.t, s.x, s.y, s.frame):
                h.update(np.nan_to_num(np.asarray(arr, float), nan=-1.0).tobytes())
            for b in s.truth:
                h.update(np.asarray([b.start, b.end, b.peak, float(b.supported)]).tobytes())
        return h.hexdigest()

    assert digest(bq.make_bank(5, 1, 3e-4)) == digest(bq.make_bank(5, 1, 3e-4))
    assert digest(bq.make_bank(5, 1, 3e-4)) != digest(bq.make_bank(6, 1, 3e-4))


def test_p08_degenerate_detectors_fail_admissibility_and_score_low():
    bq = _bq()
    bank = bq.make_bank(11, 1, 0.0, scales=(0.0,), artifacts_on=False)
    truth = {s.sid: [{"sparc": -1.5, "peaks": [1, 1, 1]} for _ in s.truth] for s in bank}

    def rows(spans):
        return [{"start_sec": a, "end_sec": b, "censored": False, "sparc_bounds": -1.5,
                 "sparc_extended": -1.5, "peaks": [1, 1, 1]} for a, b in spans]  # fmt: skip

    perfect = {s.sid: rows([(b.start, b.end) for b in s.truth]) for s in bank}
    none = {s.sid: [] for s in bank}
    fragments = {
        s.sid: rows(
            [
                p
                for b in s.truth
                for p in ((b.start, (b.start + b.end) / 2), ((b.start + b.end) / 2, b.end))
            ]
        )
        for s in bank
    }
    assert bq.exact_on_supported(bank, perfect)
    assert not bq.exact_on_supported(bank, none)
    assert not bq.exact_on_supported(bank, fragments)
    assert bq.score(bank, perfect, truth, "extended", 0)["all"]["f1"] == 1.0
    assert bq.score(bank, fragments, truth, "extended", 0)["all"]["f1"] < 0.5
    assert math.isnan(bq.score(bank, none, truth, "extended", 0)["all"]["precision"])


def test_p08_matching_is_one_to_one_within_tolerance():
    bq = _bq()
    assert bq.match([(1.0, 2.0)], [(1.05, 2.05), (1.0, 2.0)]) == [(0, 1)]
    assert bq.match([(1.0, 2.0)], [(1.2, 2.0)]) == []  # 0.2 s > max(0.1, 0.1 x 1 s)
    assert bq.match([(1.0, 4.0)], [(1.25, 4.0)]) == [(0, 0)]  # 0.25 s <= 0.1 x 3 s


def test_p09_run_report_names_the_r_producer(tmp_path, monkeypatch):
    from test_m2u101_hand_gate import _generation

    _, args, _, _ = _generation("corpus_run_2d.py", tmp_path, monkeypatch, enabled=True)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    expected = hashlib.sha256(_CLINICAL_R.read_bytes()).hexdigest()
    assert report["configuration"]["analysis_sha256"] == expected


def test_p09_refresh_clinical_reruns_r_over_complete_events(tmp_path, monkeypatch):
    from test_m2u101_hand_gate import _generation

    driver, args, generated, event_out = _generation(
        "corpus_run_2d.py", tmp_path, monkeypatch, enabled=True
    )
    calls = []
    original = driver["subprocess"].call

    def counting(command, **kwargs):
        calls.append(command[0])
        return original(command, **kwargs)

    monkeypatch.setattr(driver["subprocess"], "call", counting)
    args.analyse_only = True
    args.refresh_clinical = True
    assert driver["main"]() == 0
    assert calls == ["Rscript"]  # one complete event → one R pass, no pose stage
    corpus_run = importlib.import_module("pose_estimation.corpus_run")
    marker = corpus_run.read_marker(event_out) or {}
    assert marker["status"] == corpus_run.MARKER_COMPLETE
    assert marker["stage"] == corpus_run.STAGE_CLINICAL
    assert len(generated) == 1  # the pose stage ran once, in the original generation


def test_c07_sparse_clean_motion_is_still_a_bout():
    """Rest filling > 95 % of a noise-free series zeroes the quantile + floor; the series maximum
    keeps the movement (closing review C07)."""
    short, sparse, still = _detect(
        _series(total=4.0), _series(total=60.0), _series(total=60.0, parts=())
    )
    assert len(short["start"]) == 1
    assert len(sparse["start"]) == 1
    assert len(still["start"]) == 0


def test_c08_noise_proxy_floor_counts_the_selected_centres():
    bq = _bq()
    rng = np.random.default_rng(24)
    t = np.arange(40) / 30.0
    x, y = 0.5 + rng.normal(0, 0.001, (2, 40))  # 34 full stencils, 7 in the slowest 20 %
    assert bq.noise_proxy(t, x, y, 30.0) is None
    t = np.arange(400) / 30.0
    x, y = 0.5 + rng.normal(0, 0.001, (2, 400))
    assert bq.noise_proxy(t, x, y, 30.0) is not None


def test_a08_wrist_reader_matches_the_r_observation_mask():
    """`read_wrist` = R's `adapt_2d_confidence` + `stabilize_camera` on non-finite cells."""
    bq = _bq()
    expected = _r(
        """
        df <- data.frame(body_left_wrist_x = c(0.2, 0.2, 0.2), body_left_wrist_y = c(0.3, 0.3, 0.3),
                         body_left_wrist_vis = c(Inf, 1, 0.5), cam_a = c(1, Inf, 2),
                         cam_b = c(0, 0, 0), cam_tx = c(0, 0, 0.1), cam_ty = c(0, 0, 0))
        df <- stabilize_camera(adapt_2d_confidence(df))
        result <- list(x = as.list(df$body_left_wrist_x), y = as.list(df$body_left_wrist_y))
        """,
        {},
    )
    rows = [
        {"timestamp_sec": str(i / 30), "body_left_wrist_x": "0.2", "body_left_wrist_y": "0.3",
         "body_left_wrist_vis": vis, "cam_a": a, "cam_b": "0", "cam_tx": tx, "cam_ty": "0"}
        for i, (vis, a, tx) in enumerate((("Inf", "1", "0"), ("1", "Inf", "0"), ("0.5", "2", "0.1")))
    ]  # fmt: skip
    _, x, y = bq.read_wrist(rows, "left")
    want = [[math.nan if v is None else v for v in expected[k]] for k in ("x", "y")]
    np.testing.assert_allclose([x, y], want, equal_nan=True)


def test_p08_development_lookup_fails_admissibility():
    """A detector that memorises the development bank's truths fails on the noise-free bank,
    which is drawn from its own seed (contract-m4u5a P08)."""
    bq = _bq()
    dev = bq.make_bank(11, 1, 3e-4)
    free = bq.make_bank(12, 1, 0.0, scales=(0.0,), artifacts_on=False)
    by_slot = {(d.kind, d.rate, d.sid.rsplit("-", 1)[1]): d for d in dev if d.noise_scale == 1.0}
    lookup = {
        f.sid: [
            {"start_sec": b.start, "end_sec": b.end, "censored": False}
            for b in by_slot[(f.kind, f.rate, f.sid.rsplit("-", 1)[1])].truth
        ]
        for f in free
    }
    assert any(lookup.values())  # the lookup answers every series with the memorised truths
    assert not bq.exact_on_supported(free, lookup)


def _driver_generation(tmp_path, monkeypatch, producer):
    import test_m2u101_hand_gate as helpers

    corpus_run = importlib.import_module("pose_estimation.corpus_run")
    monkeypatch.setattr(corpus_run, "ANALYSIS_R", producer)
    original = helpers._driver

    def driver_with(name):
        driver = original(name)
        driver["CLINICAL_R"] = producer
        return driver

    monkeypatch.setattr(helpers, "_driver", driver_with)
    return helpers._generation("corpus_run_2d.py", tmp_path, monkeypatch, enabled=True)


def test_c10_analyse_only_reports_the_producer_the_events_ran(tmp_path, monkeypatch):
    producer = tmp_path / "synthetic_clinical.R"
    producer.write_text("# first producer\n")
    driver, args, _, _ = _driver_generation(tmp_path, monkeypatch, producer)
    old = hashlib.sha256(producer.read_bytes()).hexdigest()
    assert json.loads(args.report.read_text())["configuration"]["analysis_sha256"] == old
    producer.write_text("# changed producer, never run over the event\n")
    args.analyse_only, args.refresh_clinical = True, False
    assert driver["main"]() == 0
    report = json.loads(args.report.read_text())
    assert report["configuration"]["analysis_sha256"] == old
    assert report["verdicts"]["clinical_method_uniform"] is True
    args.refresh_clinical = True
    assert driver["main"]() == 0
    new = hashlib.sha256(producer.read_bytes()).hexdigest()
    assert json.loads(args.report.read_text())["configuration"]["analysis_sha256"] == new


def test_c10_interrupted_refresh_leaves_the_event_failed(tmp_path, monkeypatch):
    producer = tmp_path / "synthetic_clinical.R"
    producer.write_text("# producer\n")
    driver, args, _, event_out = _driver_generation(tmp_path, monkeypatch, producer)
    corpus_run = importlib.import_module("pose_estimation.corpus_run")

    def interrupted(command, **_kwargs):
        (event_out / "partial_clinical.csv").write_text("half")
        raise KeyboardInterrupt

    monkeypatch.setattr(driver["subprocess"], "call", interrupted)
    args.analyse_only, args.refresh_clinical = True, True
    with pytest.raises(KeyboardInterrupt):
        driver["main"]()
    assert (corpus_run.read_marker(event_out) or {})["status"] == corpus_run.MARKER_FAILED


@pytest.mark.parametrize("name", ["corpus_run_2d.py", "pilot_corpus_run.py"])
def test_a12_each_driver_report_records_the_running_build(tmp_path, monkeypatch, name):
    import openvino

    from test_m2u101_hand_gate import _generation

    _, args, _, _ = _generation(name, tmp_path, monkeypatch, enabled=True)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    assert report["configuration"]["openvino_build"] == openvino.get_version()


def test_s12_a_malformed_recorded_digest_never_reaches_the_report(tmp_path, monkeypatch):
    """A marker value joins the report's redaction allowlist only when it is digest-shaped."""
    producer = tmp_path / "synthetic_clinical.R"
    producer.write_text("# producer\n")
    driver, args, _, event_out = _driver_generation(tmp_path, monkeypatch, producer)
    corpus_run = importlib.import_module("pose_estimation.corpus_run")
    marker = corpus_run.read_marker(event_out)
    assert marker is not None
    assert len(marker["analysis_sha256"]) == 64  # positive control: a real digest was recorded
    corpus_run.write_marker(event_out, **(marker | {"analysis_sha256": "PRIVATE-subject/a.mp4"}))
    args.analyse_only, args.refresh_clinical = True, False
    driver["main"]()
    text = args.report.read_text()
    assert "PRIVATE-subject" not in text
    report = json.loads(text)
    assert report["configuration"]["analysis_sha256"] == "mixed"
    assert report["verdicts"]["clinical_method_uniform"] is False
