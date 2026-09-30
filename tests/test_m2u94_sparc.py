"""M2.9.4 P01-P06: diff-blind SPARC oracle, guards, labels and consumers."""

from __future__ import annotations

import csv
import json
import math
import pathlib
import subprocess
import tempfile
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CLINICAL_R = _ROOT / "analysis" / "clinical_features.R"
_ATOL = 1e-9
_RATES = (15, 25, 30, 60)
_PROFILE_KINDS = ("bell", "multi_peak", "noisy", "constant", "near_zero")


def _r_literal(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        return "c(" + ",".join(_r_literal(item) for item in value) + ")"
    number = float(value)
    if math.isnan(number):
        return "NA_real_"
    if math.isinf(number):
        return "Inf" if number > 0 else "-Inf"
    return repr(number)


def _run_r(body: str, *, clinical: bool = True) -> dict[str, Any]:
    source = ""
    if clinical:
        source = (
            f"suppressWarnings(try(source({_r_literal(str(_CLINICAL_R))}), silent=TRUE))\n"
            'stopifnot(exists("spectral_arc_length", mode="function"))\n'
        )
    proc = subprocess.run(
        ["Rscript", "-"],
        input=source
        + body
        + "\njsonlite::write_json(result, stdout(), auto_unbox=TRUE, na='string', digits=17)\n",
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"R harness rc={proc.returncode}: {proc.stderr}"
    payload = next(
        (line for line in reversed(proc.stdout.splitlines()) if line.startswith("{")), ""
    )
    assert payload, f"R harness emitted no result: {proc.stdout}"
    return json.loads(payload)


def _scores(cases: Sequence[dict[str, Any]]) -> list[float]:
    assert cases, "SPARC probe must contain inputs"
    payload = []
    for case in cases:
        row = dict(case)
        row["v"] = [
            float(value) if math.isfinite(value) else str(float(value)) for value in case["v"]
        ]
        for key in row.keys() - {"v"}:
            if not math.isfinite(row[key]):
                row[key] = str(float(row[key]))
        payload.append(row)
    with tempfile.TemporaryDirectory(prefix="sparc-oracle-") as directory:
        request_path = pathlib.Path(directory) / "profiles.json"
        request_path.write_text(
            json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8"
        )
        result = _run_r(
            f"requests <- jsonlite::fromJSON({_r_literal(str(request_path))}, simplifyVector=FALSE)\n"
            """
            result <- list(values=lapply(requests, function(request) {
              # Elementwise: unlist() over numbers mixed with the "nan"/"inf" strings coerces every
              # value to 15-digit text and moves finite samples in their last bits.
              request <- lapply(request, function(value) vapply(
                value, function(x) suppressWarnings(as.numeric(x)), numeric(1)
              ))
              tryCatch({
                value <- do.call(spectral_arc_length, request)
                list(value=value, nan=is.nan(value))
              }, error=function(e) list(error=conditionMessage(e)))
            }))
            """
        )
    rows = result["values"]
    assert len(rows) == len(cases), "R omitted a SPARC input"
    values = []
    for case, row in zip(cases, rows, strict=True):
        context = {key: value for key, value in case.items() if key != "v"}
        context["n"] = len(case["v"])
        assert "error" not in row, f"SPARC raised for {context}: {row}"
        assert row["nan"] is False, f"SPARC returned NaN instead of a scalar or NA for {context}"
        values.append(math.nan if row["value"] == "NA" else float(row["value"]))
    return values


def _spectrum(v: np.ndarray, fs: float, fc: float, pad_level: int) -> tuple[np.ndarray, np.ndarray]:
    nfft = 1 << ((len(v) - 1).bit_length() + pad_level)
    magnitude = np.abs(np.fft.fft(v, n=nfft))
    magnitude /= magnitude.max()
    frequency = np.arange(nfft, dtype=float) * fs / nfft
    band = frequency <= min(fc, fs / 2)
    return frequency[band], magnitude[band]


def _arc(frequency: np.ndarray, magnitude: np.ndarray) -> float:
    if len(frequency) == 1:
        return 0.0
    return -float(
        np.hypot(np.diff(frequency) / (frequency[-1] - frequency[0]), np.diff(magnitude)).sum()
    )


def _reference(
    v: Sequence[float] | np.ndarray,
    fs: float,
    fc: float = 10,
    amp_th: float = 0.05,
    pad_level: int = 4,
) -> float:
    """D01/D02 transcription; no production helper or legacy SAL oracle."""
    values = np.asarray(v, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 4 or fs <= 0:
        return math.nan
    if np.max(np.abs(values)) < 1e-10:
        return 0.0
    frequency, magnitude = _spectrum(values, fs, fc, pad_level)
    above = np.flatnonzero(magnitude >= amp_th)
    assert len(above), "Oracle input has no threshold crossing in the selected band"
    kept = slice(int(above[0]), int(above[-1]) + 1)
    return _arc(frequency[kept], magnitude[kept])


def _assert_reference(cases: Sequence[dict[str, Any]], actual: Sequence[float]) -> None:
    assert len(cases) == len(actual) > 0
    for case, value in zip(cases, actual, strict=True):
        expected = _reference(**case)
        context = {key: item for key, item in case.items() if key != "v"}
        context["n"] = len(case["v"])
        if math.isnan(expected):
            assert math.isnan(value), f"{context}: expected NA, got {value}"
        else:
            assert value == pytest.approx(expected, rel=0, abs=_ATOL), context


def _bell(t: np.ndarray) -> np.ndarray:
    u = np.clip(t, 0, 1)
    return 30 * u**2 * (1 - u) ** 2


def _profile(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0, 1, n)
    bell = _bell(t)
    if kind == "bell":
        return bell * rng.uniform(0.1, 5)
    if kind == "multi_peak":
        return bell + 0.8 * (_bell((t - 0.1) / 0.25) + _bell((t - 0.65) / 0.25))
    if kind == "noisy":
        return np.maximum(0, bell + rng.normal(0, rng.uniform(0.05, 0.8), n))
    if kind == "constant":
        return np.full(n, 10 ** rng.uniform(-4, 4))
    assert kind == "near_zero"
    amplitude = (np.nextafter(1e-10, 0), 1e-10, np.nextafter(1e-10, math.inf))[n % 3]
    return bell / bell.max() * amplitude


@pytest.mark.parametrize("fs", _RATES)
@pytest.mark.parametrize("kind", _PROFILE_KINDS)
def test_p01_generated_profiles_match_reference(kind: str, fs: int) -> None:
    rng = np.random.default_rng(9400 + fs + _PROFILE_KINDS.index(kind))
    cases = [{"v": _profile(kind, n, rng), "fs": fs} for n in range(4, 601)]
    _assert_reference(cases, _scores(cases))


@pytest.mark.parametrize("pad_level", [0, 1, 4, 6])
@pytest.mark.parametrize("amp_th", [0.0, 0.05, 0.3, 1.0])
def test_p01_optional_parameters_are_live(pad_level: int, amp_th: float) -> None:
    rng = np.random.default_rng(9401)
    cases = [
        {
            "v": _profile("multi_peak", n, rng),
            "fs": 30,
            "fc": fc,
            "amp_th": amp_th,
            "pad_level": pad_level,
        }
        for n in (5, 16, 63, 129, 600)
        for fc in (0.0, 1.7, 10.0, 80.0)
    ]
    _assert_reference(cases, _scores(cases))


def test_p01_named_defaults_match_explicit_reference_parameters() -> None:
    result = _run_r(
        "result <- list(amp=SPARC_AMP_THRESHOLD, pad=SPARC_PAD_LEVEL, cutoff=SAL_FREQ_CUTOFF)"
    )
    assert result == {"amp": 0.05, "pad": 4, "cutoff": 10}
    v = _bell(np.linspace(0, 1, 137))
    default, explicit = _scores(
        [
            {"v": v, "fs": 30},
            {"v": v, "fs": 30, "fc": 10, "amp_th": 0.05, "pad_level": 4},
        ]
    )
    assert default == explicit
    assert default == pytest.approx(_reference(v, 30), rel=0, abs=_ATOL)


def test_p01_amplitude_normalization_is_scale_invariant() -> None:
    rng = np.random.default_rng(9402)
    cases = []
    for _ in range(48):
        n = int(rng.integers(4, 601))
        fs = int(rng.choice(_RATES))
        v = rng.uniform(0.01, 4, n)
        cases.extend({"v": v * scale, "fs": fs} for scale in (1e-7, 1, 1e7))
    values = np.asarray(_scores(cases)).reshape(-1, 3)
    assert len(values) == 48
    assert np.max(np.abs(values - values[:, [1]])) <= _ATOL
    _assert_reference(cases, values.ravel().tolist())


@pytest.mark.parametrize("fs", _RATES)
def test_p02_minimum_jerk_bell_is_smoother_than_superimposed_movements(fs: int) -> None:
    t = np.linspace(0, 1, 121)
    bell = _bell(t)
    compound = bell + 0.8 * (_bell((t - 0.1) / 0.25) + _bell((t - 0.65) / 0.25))
    cases: list[dict[str, Any]] = [{"v": v, "fs": fs} for v in (bell, compound)]
    expected = [_reference(**case) for case in cases]
    assert expected[0] > expected[1], "Minimum-jerk fixture does not distinguish the reference"
    actual = _scores(cases)
    assert actual[0] > actual[1]


_GUARDS = [
    pytest.param([], 30, id="empty"),
    pytest.param([0], 30, id="one-zero"),
    pytest.param([1, 2], 30, id="two-finite"),
    pytest.param([1, 2, 3], 30, id="three-finite"),
    pytest.param([0] * 4, 30, id="four-zero"),
    pytest.param([0] * 600, 60, id="six-hundred-zero"),
    pytest.param([1, 2, 3, 4], 0, id="zero-fs"),
    pytest.param([1, 2, 3, 4], -30, id="negative-fs"),
    pytest.param([0] * 4, 0, id="zero-fs-before-no-movement"),
    pytest.param([0] * 4, -30, id="negative-fs-before-no-movement"),
    pytest.param([1, 2, 3, 4], -math.inf, id="negative-infinite-fs"),
    pytest.param([math.nan] * 8, 30, id="all-na"),
    pytest.param([math.inf, -math.inf] * 4, 30, id="all-infinite"),
    pytest.param([1, math.nan, 2, 3, math.nan], 30, id="na-leaves-three"),
    pytest.param([1, math.inf, 2, 3, -math.inf], 30, id="infinity-leaves-three"),
    pytest.param([math.nan, 1, 2, 3, 4, math.nan], 30, id="na-leaves-four"),
    pytest.param([math.inf, 1, 2, 3, 4, -math.inf], 30, id="infinity-leaves-four"),
    pytest.param([np.nextafter(1e-10, 0)] * 4, 30, id="below-movement-floor"),
    pytest.param([1e-10] * 4, 30, id="at-movement-floor"),
    pytest.param([np.nextafter(1e-10, math.inf)] * 4, 30, id="above-movement-floor"),
]


@pytest.mark.parametrize(("v", "fs"), _GUARDS)
def test_p03_guards_apply_after_dropping_nonfinite_samples(v: list[float], fs: float) -> None:
    clean = [value for value in v if math.isfinite(value)]
    actual = _scores([{"v": v, "fs": fs}])[0]
    if len(clean) < 4 or fs <= 0:
        assert math.isnan(actual)
    elif max(abs(value) for value in clean) < 1e-10:
        assert actual == 0
    else:
        assert math.isfinite(actual)
        assert actual == _scores([{"v": clean, "fs": fs}])[0]
        assert actual < 0, "Movement-floor probe was incorrectly treated as no movement"


@pytest.mark.parametrize("fs", _RATES)
def test_p03_cutoff_above_nyquist_is_clamped(fs: int) -> None:
    v = _bell(np.linspace(0, 1, 91)) + 0.2
    cases = [{"v": v, "fs": fs, "fc": fc} for fc in (fs / 2, fs, 1e6)]
    actual = _scores(cases)
    assert actual[0] == actual[1] == actual[2]
    assert math.isfinite(actual[0])
    assert actual[0] < 0


def test_p03_nonfinite_deletion_is_position_independent() -> None:
    rng = np.random.default_rng(9403)
    cases = []
    for _ in range(32):
        v = rng.uniform(0, 2, int(rng.integers(4, 601)))
        fs = int(rng.choice(_RATES))
        decorated = np.insert(v, rng.integers(0, len(v) + 1, 3), [math.nan, math.inf, -math.inf])
        cases.extend([{"v": v, "fs": fs}, {"v": decorated, "fs": fs}])
    actual = _scores(cases)
    assert len(actual) == 64
    assert actual[::2] == actual[1::2]
    assert all(math.isfinite(value) for value in actual)


def test_p04_adaptive_cutoff_excludes_subthreshold_tail() -> None:
    v = _bell(np.linspace(0, 1, 121))
    frequency, magnitude = _spectrum(v, 30, 10, 4)
    last = int(np.flatnonzero(magnitude >= 0.05)[-1])
    expected = _reference(v, 30)
    fixed_band = _arc(frequency, magnitude)
    assert 0 < frequency[last] < frequency[-1]
    assert np.all(magnitude[last + 1 :] < 0.05)
    assert abs(expected - fixed_band) > 0.05, "Adaptive-cutoff probe is not discriminating"
    actual = _scores([{"v": v, "fs": 30}])[0]
    assert abs(actual - fixed_band) > 0.05
    assert actual == pytest.approx(expected, rel=0, abs=_ATOL)


def test_p04_subthreshold_holes_inside_kept_range_are_retained() -> None:
    v = np.ones(100)
    frequency, magnitude = _spectrum(v, 30, 10, 4)
    above = np.flatnonzero(magnitude >= 0.05)
    last = int(above[-1])
    assert np.any(magnitude[:last] < 0.05)
    expected = _reference(v, 30)
    only_crossings = _arc(frequency[above], magnitude[above])
    assert abs(expected - only_crossings) > 0.01
    assert _scores([{"v": v, "fs": 30}])[0] == pytest.approx(expected, rel=0, abs=_ATOL)


def test_p04_flat_spectrum_keeps_the_full_band() -> None:
    v = np.array([1, 0, 0, 0, 0], dtype=float)
    frequency, magnitude = _spectrum(v, 30, 10, 4)
    assert len(frequency) > 1
    assert np.all(magnitude >= 0.05)
    assert _reference(v, 30) == pytest.approx(-1, rel=0, abs=1e-14)
    assert _scores([{"v": v, "fs": 30}])[0] == pytest.approx(-1, rel=0, abs=_ATOL)


def test_p04_threshold_equality_is_included() -> None:
    case: dict[str, Any] = {"v": [1, 0, 0, 0, 0], "fs": 30, "amp_th": 1.0}
    assert _reference(**case) == pytest.approx(-1, rel=0, abs=1e-14)
    assert _scores([case])[0] == pytest.approx(-1, rel=0, abs=_ATOL)


def test_p04_single_kept_frequency_scores_zero() -> None:
    cases = [{"v": np.ones(n), "fs": 30, "amp_th": 1} for n in (4, 5, 32, 129, 600)]
    assert _scores(cases) == [0.0] * len(cases)


def test_p05_method_constant_is_v3() -> None:
    assert _run_r("result <- list(version=METRIC_METHOD_VERSION)")["version"] == "v3"


@pytest.mark.parametrize("language", ["ja", "en"])
def test_p05_cohort_wrist_sal_labels_name_sparc(language: str) -> None:
    from pose_estimation import cohort

    assert pathlib.Path(cohort.__file__).resolve().is_relative_to(_ROOT)
    expected = {
        "ja": {
            "left": "左手関節スペクトルアーク長(SPARC)",
            "right": "右手関節スペクトルアーク長(SPARC)",
        },
        "en": {
            "left": "Left wrist spectral arc length (SPARC)",
            "right": "Right wrist spectral arc length (SPARC)",
        },
    }
    actual = {
        side: getattr(feature, language)
        for side in ("left", "right")
        for feature in cohort.FEATURES
        if feature.column == f"{side}_wrist_sal" and feature.level == "window"
    }
    assert actual == expected[language]
    family = [item for item in cohort.FEATURES if "wrist_sal" in item.column]
    assert family, "SPARC label family is empty"
    base = {"ja": "手関節スペクトルアーク長(SPARC)", "en": "Wrist spectral arc length (SPARC)"}
    assert all(base[language].casefold() in getattr(item, language).casefold() for item in family)


@pytest.fixture(scope="module")
def clinical_inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, pathlib.Path]:
    from test_r_pipeline import _write_reach_grasp_csv, _write_world3d_fixture

    directory = tmp_path_factory.mktemp("sparc-clinical")
    inputs = {"2d": directory / "synthetic_hands-arms.csv", "3d": directory / "world3d.csv"}
    _write_reach_grasp_csv(inputs["2d"], "hands-arms")
    _write_world3d_fixture(inputs["3d"])
    for path in inputs.values():
        proc = subprocess.run(
            ["Rscript", str(_CLINICAL_R), str(path)],
            cwd=_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
    return inputs


@pytest.mark.parametrize(
    "suffix",
    [
        "_clinical_3d.csv",
        "_clinical_3d_windows.csv",
        "_movement_phases_3d.csv",
        "_clinical_3d_window_qc.csv",
    ],
)
def test_p05_emitted_3d_artifacts_stamp_v3(
    clinical_inputs: dict[str, pathlib.Path], suffix: str
) -> None:
    source = clinical_inputs["3d"]
    output = source.with_name(source.stem + suffix)
    with output.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert "metric_method_version" in (reader.fieldnames or []), output.name
        rows = list(reader)
    assert rows, f"Version probe is vacuous for {output.name}"
    assert {row["metric_method_version"] for row in rows} == {"v3"}, output.name


def test_p05_temporal_panel_names_sparc(clinical_inputs: dict[str, pathlib.Path]) -> None:
    temporal = _ROOT / "analysis" / "temporal_clinical.R"
    directory = clinical_inputs["2d"].parent
    result = _run_r(
        f"""
        suppressPackageStartupMessages(library(ggplot2))
        plots <- list()
        capture_plot <- function(filename, plot=last_plot(), ...) {{
          plots[[length(plots) + 1L]] <<- plot
          invisible(NULL)
        }}
        assignInNamespace("ggsave", capture_plot, ns="ggplot2")
        ggsave <- capture_plot
        commandArgs <- function(trailingOnly=FALSE) {{
          if (trailingOnly) return({_r_literal(str(directory))})
          c("Rscript", paste0("--file=", {_r_literal(str(temporal))}),
            {_r_literal(str(directory))})
        }}
        source({_r_literal(str(temporal))})
        grob_labels <- function(grob) {{
          text <- if (inherits(grob, "text")) as.character(grob$label) else character()
          c(text, unlist(lapply(grob$grobs, grob_labels), use.names=FALSE),
            unlist(lapply(grob$children, grob_labels), use.names=FALSE))
        }}
        grDevices::pdf(file=NULL)
        labels <- lapply(plots, function(plot) {{
          grob <- if (inherits(plot, "patchwork")) patchwork::patchworkGrob(plot)
                  else ggplot2::ggplotGrob(plot)
          grob_labels(grob)
        }})
        grDevices::dev.off()
        result <- list(n_plots=length(plots), labels=as.list(unlist(labels, use.names=FALSE)))
        """,
        clinical=False,
    )
    assert result["n_plots"] > 0, "Temporal panel probe captured no emitted plot"
    labels = {
        label
        for label in result["labels"]
        if "wrist" in label.casefold()
        and any(token in label.casefold() for token in ("sal", "sparc", "spectral"))
    }
    assert labels, "Temporal panel probe captured no live wrist-smoothness label"
    assert all("SPARC" in label for label in labels), labels


@pytest.mark.parametrize("consumer", ["2d_windows", "3d_reach"])
def test_p06_both_consumers_use_the_shared_function(
    clinical_inputs: dict[str, pathlib.Path], consumer: str
) -> None:
    if consumer == "2d_windows":
        path = clinical_inputs["2d"]
        setup = """
        df <- adapt_2d_confidence(df)
        frame <- compute_frame_features(df, "hands-arms", is_3d=FALSE)
        consume <- function() compute_window_features(
          df, frame, "hands-arms", is_3d=FALSE)$windows
        sal_columns <- c("left_wrist_sal", "right_wrist_sal")
        """
    else:
        path = clinical_inputs["3d"]
        setup = """
        df <- adapt_world3d(df)
        frame <- compute_frame_features(df, "body", is_3d=TRUE)
        consume <- function() segment_movements(df, frame, "body")
        sal_columns <- "smoothness_sal"
        """
    result = _run_r(
        f"df <- readr::read_csv({_r_literal(str(path))}, show_col_types=FALSE)\n"
        + setup
        + """
        before <- consume()
        calls <- 0L
        sentinel <- -123.456
        spectral_arc_length <- function(v, fs, fc=SAL_FREQ_CUTOFF, ...) {
          calls <<- calls + 1L
          sentinel
        }
        after <- consume()
        result <- list(
          n_before=nrow(before), n_after=nrow(after), calls=calls,
          before=as.list(unname(unlist(before[sal_columns]))),
          after=as.list(unname(unlist(after[sal_columns]))),
          has_reach=if ("phase" %in% names(after)) any(after$phase == "REACH") else TRUE
        )
        """
    )
    assert result["n_before"] == result["n_after"] > 0, consumer
    assert result["has_reach"] is True, consumer
    assert result["calls"] > 0, consumer
    before = result["before"]
    after = result["after"]
    assert len(before) == len(after) > 0, consumer
    assert any(value != "NA" and float(value) != -123.456 for value in before), consumer
    assert after == pytest.approx([-123.456] * len(after), rel=0, abs=1e-12), consumer
