"""Executable M3.1 contract for the R timestamp-aware trajectory kernel."""

from __future__ import annotations

import itertools
import json
import math
import pathlib
import re
import shutil
import subprocess
import textwrap
from typing import Any, cast

import pytest

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CLINICAL_R = _PROJECT_ROOT / "analysis" / "clinical_features.R"
_ARTHROSE_R = _PROJECT_ROOT / "analysis" / "arthrose_diag.R"
_PIPELINE_TEST = _PROJECT_ROOT / "tests" / "test_r_pipeline.py"
_RENV_LOCK = _PROJECT_ROOT / "renv.lock"
_BASELINE = "665b107"
_WINDOW_METRIC_SUFFIXES = {
    "wrist_sal",
    "wrist_velocity_mean",
    "wrist_velocity_peak",
    "wrist_normalized_jerk",
    "wrist_movement_efficiency",
    "fingertip_normalized_jerk",
}


def _r_available() -> bool:
    if not shutil.which("Rscript"):
        return False
    probe = subprocess.run(
        ["Rscript", "-e", "quit(status=!requireNamespace('jsonlite',quietly=TRUE))"],
        cwd=_PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(not _r_available(), reason="R/jsonlite unavailable")


def _r_literal(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return "NA_real_"
        return repr(value)
    if isinstance(value, (list, tuple)):
        return f"c({', '.join(_r_literal(item) for item in value)})"
    raise TypeError(type(value))


def _run_r(body: str, *, clinical: bool = True) -> dict[str, Any]:
    source = ""
    if clinical:
        source = textwrap.dedent(
            f"""
            suppressWarnings(try(source({_r_literal(str(_CLINICAL_R))}), silent=TRUE))
            .m3u1_has_kernel <- exists("trajectory_metrics", mode="function")
            if (!exists("nominal_fs", mode="function")) {{
              nominal_fs <- function(t) 1 / median(diff(t), na.rm=TRUE)
            }}
            if (!exists("trajectory_grid", mode="function")) {{
              trajectory_grid <- function(t, fs) {{
                slot <- round((t - t[1]) * fs)
                list(slot=slot, n_grid=length(slot), residual=0)
              }}
            }}
            if (!.m3u1_has_kernel) {{
              trajectory_metrics <- function(t, x, y, z, fs=NULL, fc=SAL_FREQ_CUTOFF) {{
                if (is.null(fs)) fs <- nominal_fs(t)
                step <- sqrt(diff(x)^2 + diff(y)^2 + diff(z)^2)
                speed <- step * fs
                list(
                  sal=spectral_arc_length(speed, fs, fc),
                  nj=normalized_jerk(x, y, z, fs),
                  v_mean=mean(speed, na.rm=TRUE),
                  v_peak=max(speed, na.rm=TRUE),
                  efficiency=movement_efficiency(x, y, z),
                  dropout=NA_real_, longest_gap_sec=NA_real_
                )
              }}
            }}
            """
        )
    script = (
        source
        + body
        + "\njsonlite::write_json(result, stdout(), auto_unbox=TRUE, na='string', digits=17)\n"
    )
    proc = subprocess.run(
        ["Rscript", "-"],
        input=script,
        cwd=_PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    payload = next(
        (line for line in reversed(proc.stdout.splitlines()) if line.startswith("{")), ""
    )
    assert payload, proc.stdout
    return json.loads(payload)


def _numeric(value: Any) -> float:
    return math.nan if value == "NA" else float(value)


def _minimum_jerk_xyz() -> tuple[list[float], list[float], list[float]]:
    t = [index / 30 for index in range(91)]
    tau = [value / 3 for value in t]
    x = [0.40 * (10 * u**3 - 15 * u**4 + 6 * u**5) for u in tau]
    y = [0.05 * math.sin(math.pi * u) for u in tau]
    return x, y, [0.0] * len(x)


def _masked_nj_expected(dropped: list[int]) -> float:
    x, y, z = _minimum_jerk_xyz()
    for index in dropped:
        x[index] = y[index] = z[index] = math.nan

    step = [
        math.sqrt((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2)
        if all(math.isfinite(value) for value in (ax, ay, az, bx, by, bz))
        else math.nan
        for ax, ay, az, bx, by, bz in zip(x[:-1], y[:-1], z[:-1], x[1:], y[1:], z[1:], strict=True)
    ]

    def derivative(values: list[float]) -> list[float]:
        return [
            (right - left) * 30 if math.isfinite(left) and math.isfinite(right) else math.nan
            for left, right in itertools.pairwise(values)
        ]

    jx = derivative(derivative(derivative(x)))
    jy = derivative(derivative(derivative(y)))
    jz = derivative(derivative(derivative(z)))
    amplitude = sum(value for value in step if math.isfinite(value))
    integral = (
        sum(
            ax**2 + ay**2 + az**2
            for ax, ay, az in zip(jx, jy, jz, strict=True)
            if all(math.isfinite(value) for value in (ax, ay, az))
        )
        / 30
    )
    return math.sqrt(3**5 / (2 * amplitude**2) * integral)


def _trajectory_setup(timestamp: str = "idx") -> str:
    return textwrap.dedent(
        f"""
        fs <- 30
        T <- 3
        idx <- 0:90
        t <- switch({_r_literal(timestamp)},
          idx = idx / fs,
          cumsum = cumsum(c(0, rep(1 / fs, length(idx) - 1))),
          csv4 = round(idx / fs, 4)
        )
        tau <- (idx / fs) / T
        s <- 10 * tau^3 - 15 * tau^4 + 6 * tau^5
        x <- 0.40 * s
        y <- 0.05 * sin(pi * tau)
        z <- rep(0, length(t))
        """
    )


def _legacy_metrics_r() -> str:
    return textwrap.dedent(
        """
        legacy_sal <- function(v, fs, fc=10) {
          v <- v[!is.na(v)]; n <- length(v)
          if (n < 4 || fs <= 0) return(NA_real_)
          v_peak <- max(abs(v)); if (v_peak < 1e-10) return(0)
          V <- Mod(fft(v / v_peak))[seq_len(floor(n / 2) + 1)]
          V <- V / max(V)
          freqs <- seq(0, fs / 2, length.out=length(V))
          fc <- min(fc, fs / 2); keep <- freqs <= fc
          V <- V[keep]; freqs <- freqs[keep]
          if (length(freqs) < 2) return(NA_real_)
          -sum(sqrt((diff(freqs) / fc)^2 + diff(V)^2))
        }
        legacy_nj <- function(x, y, z, fs) {
          ok <- !is.na(x) & !is.na(y) & !is.na(z)
          x <- x[ok]; y <- y[ok]; z <- z[ok]; n <- length(x)
          if (n < 5 || fs <= 0) return(NA_real_)
          dt <- 1 / fs; T_dur <- (n - 1) * dt
          amplitude <- sum(sqrt(diff(x)^2 + diff(y)^2 + diff(z)^2))
          if (amplitude < 1e-10) return(NA_real_)
          vx <- diff(x)*fs; vy <- diff(y)*fs; vz <- diff(z)*fs
          ax <- diff(vx)*fs; ay <- diff(vy)*fs; az <- diff(vz)*fs
          jx <- diff(ax)*fs; jy <- diff(ay)*fs; jz <- diff(az)*fs
          integral <- sum(jx^2 + jy^2 + jz^2) * dt
          sqrt(T_dur^5 / (2 * amplitude^2) * integral)
        }
        legacy_efficiency <- function(x, y, z) {
          ok <- !is.na(x) & !is.na(y) & !is.na(z)
          x <- x[ok]; y <- y[ok]; z <- z[ok]; n <- length(x)
          if (n < 2) return(NA_real_)
          path <- sum(sqrt(diff(x)^2 + diff(y)^2 + diff(z)^2))
          straight <- sqrt((x[n]-x[1])^2 + (y[n]-y[1])^2 + (z[n]-z[1])^2)
          if (straight < 1e-10) return(NA_real_)
          path / straight
        }
        step <- sqrt(diff(x)^2 + diff(y)^2 + diff(z)^2)
        speed <- step * fs
        legacy <- list(
          sal=legacy_sal(speed, fs), nj=legacy_nj(x,y,z,fs),
          v_mean=mean(speed, na.rm=TRUE), v_peak=max(speed, na.rm=TRUE),
          efficiency=legacy_efficiency(x,y,z)
        )
        """
    )


@pytest.mark.parametrize("timestamp", ["idx", "cumsum", "csv4"])
@pytest.mark.parametrize("metric", ["sal", "nj", "v_mean", "v_peak", "efficiency"])
def test_gapfree_metrics_are_bit_identical_to_legacy(timestamp: str, metric: str) -> None:
    result = _run_r(
        _trajectory_setup(timestamp)
        + _legacy_metrics_r()
        + textwrap.dedent(
            f"""
            actual <- trajectory_metrics(t, x, y, z, fs=fs)
            result <- list(identical=identical(actual[[{_r_literal(metric)}]], legacy[[{_r_literal(metric)}]]))
            """
        )
    )
    assert result["identical"] is True


@pytest.mark.parametrize(
    ("pattern", "dropped", "upper_ratio", "lower_ratio"),
    [
        ("drop1_mid", [45], 1.25, 0.75),
        ("drop3_mid", [44, 45, 46], 3.0, 0.0),
        ("drop8_mid", list(range(42, 50)), 3.0, 0.0),
        ("drop15_mid", list(range(38, 53)), 3.0, 0.0),
        ("scatter15", [5, 10, 15, 20, 25, 30, 35, 40, 50, 55, 60, 65, 70, 75, 85], 3.0, 0.0),
    ],
)
def test_normalized_jerk_cannot_explode(
    pattern: str,
    dropped: list[int],
    upper_ratio: float,
    lower_ratio: float,
) -> None:
    result = _run_r(
        _trajectory_setup("csv4")
        + textwrap.dedent(
            f"""
            full <- trajectory_metrics(t, x, y, z, fs=fs)$nj
            dropped <- {_r_literal(dropped)} + 1
            x[dropped] <- NA_real_; y[dropped] <- NA_real_; z[dropped] <- NA_real_
            observed <- trajectory_metrics(t, x, y, z, fs=fs)$nj
            result <- list(pattern={_r_literal(pattern)}, full=full, observed=observed, ratio=observed/full)
            """
        )
    )
    observed = _numeric(result["observed"])
    expected = _masked_nj_expected(dropped)
    ratio = _numeric(result["ratio"])
    assert observed == pytest.approx(expected, rel=1e-11, abs=1e-10), result
    assert math.isfinite(ratio), result
    assert lower_ratio <= ratio < upper_ratio, result


@pytest.mark.parametrize(
    "mutation",
    [
        "t[10] <- Inf",
        "t[10] <- t[9] - 0.01",
        "t[10] <- t[9] + 0.001",
        "t[10] <- t[9] + 0.5 / fs",
    ],
)
def test_grid_mapping_rejects_malformed_timestamps(mutation: str) -> None:
    result = _run_r(
        _trajectory_setup()
        + textwrap.dedent(
            f"""
            {mutation}
            message <- tryCatch({{ trajectory_grid(t, fs); NA_character_ }}, error=function(e) conditionMessage(e))
            result <- list(message=message)
            """
        )
    )
    assert result["message"] != "NA"


def test_skipped_timestamp_becomes_one_missing_nominal_slot() -> None:
    result = _run_r(
        _trajectory_setup()
        + textwrap.dedent(
            """
            keep <- -46
            grid <- trajectory_grid(t[keep], fs)
            metrics <- trajectory_metrics(t[keep], x[keep], y[keep], z[keep], fs=fs)
            result <- list(n_grid=grid$n_grid, skipped=grid$n_grid-length(grid$slot), dropout=metrics$dropout)
            """
        )
    )
    assert result["n_grid"] == 91
    assert result["skipped"] == 1
    assert float(result["dropout"]) == pytest.approx(1 / 91, abs=1e-15)


@pytest.mark.parametrize("n", [91, 1801, 15455])
def test_nominal_fs_recovers_30_hz_from_four_decimal_timestamps(n: int) -> None:
    result = _run_r(
        textwrap.dedent(
            f"""
            idx <- 0:{n - 1}
            t <- round(idx / 30, 4)
            result <- list(fs=nominal_fs(t))
            """
        )
    )
    # Span-based estimation inherits the export's 4-dp rounding: the endpoint
    # carries up to 0.5e-4 s of error spread across n-1 intervals, bounding
    # the rate error at fs^2 * 0.5e-4 / (n-1). Still orders of magnitude
    # tighter than the 30.03003 that 1/median(diff(t)) returns at every n.
    bound = 30.0**2 * 0.5e-4 / (n - 1)
    assert float(result["fs"]) == pytest.approx(30.0, abs=bound)


@pytest.mark.parametrize(("fps", "n"), [(24.0, 2401), (25.0, 2501), (59.94, 2998)])
def test_nominal_fs_recovers_common_rates(fps: float, n: int) -> None:
    result = _run_r(
        textwrap.dedent(
            f"""
            idx <- 0:{n - 1}
            t <- round(idx / {_r_literal(fps)}, 4)
            result <- list(fs=nominal_fs(t))
            """
        )
    )
    assert float(result["fs"]) == pytest.approx(fps, abs=1e-6)


def test_nominal_fs_ignores_absent_rows() -> None:
    result = _run_r(
        textwrap.dedent(
            """
            idx <- 0:1800
            t <- round(idx / 30, 4)
            t <- t[-c(101:103, 800:807, 1400)]
            result <- list(fs=nominal_fs(t))
            """
        )
    )
    assert float(result["fs"]) == pytest.approx(30.0, abs=1e-6)


def test_degenerate_windows_fail_closed() -> None:
    result = _run_r(
        _trajectory_setup()
        + textwrap.dedent(
            """
            all_na <- trajectory_metrics(t, rep(NA_real_,91), rep(NA_real_,91), rep(NA_real_,91), fs=fs)
            short <- trajectory_metrics(t[1:3], x[1:3], y[1:3], z[1:3], fs=fs)
            two <- trajectory_metrics(t, replace(rep(NA_real_,91), 46:47, x[46:47]),
                                      replace(rep(NA_real_,91), 46:47, y[46:47]),
                                      replace(rep(NA_real_,91), 46:47, z[46:47]), fs=fs)
            result <- list(
              all_na=unname(unlist(lapply(all_na[c("sal","nj","v_mean","v_peak","efficiency")], is.na))),
              short_nj=is.na(short$nj),
              two_sal=is.na(two$sal), two_nj=is.na(two$nj),
              two_v_mean=two$v_mean, two_v_peak=two$v_peak, two_efficiency=two$efficiency,
              two_dropout=two$dropout, two_longest=two$longest_gap_sec
            )
            """
        )
    )
    assert result["all_na"] == [True] * 5
    assert result["short_nj"] is True
    assert result["two_sal"] is True
    assert result["two_nj"] is True
    assert math.isfinite(float(result["two_v_mean"]))
    assert math.isfinite(float(result["two_v_peak"]))
    assert math.isfinite(float(result["two_efficiency"]))
    assert float(result["two_dropout"]) == pytest.approx(89 / 91, abs=1e-15)
    assert float(result["two_longest"]) == pytest.approx(45 / 30, abs=1e-15)


@pytest.mark.parametrize("edge", ["leading", "trailing"])
def test_edge_gap_disallows_sal_extrapolation_but_trims_efficiency(edge: str) -> None:
    result = _run_r(
        _trajectory_setup()
        + textwrap.dedent(
            f"""
            missing <- if ({_r_literal(edge)} == "leading") 1 else length(t)
            x[missing] <- NA_real_; y[missing] <- NA_real_; z[missing] <- NA_real_
            metrics <- trajectory_metrics(t, x, y, z, fs=fs)
            result <- list(sal_na=is.na(metrics$sal), efficiency=metrics$efficiency)
            """
        )
    )
    assert result["sal_na"] is True
    assert math.isfinite(float(result["efficiency"]))


def test_interior_gap_makes_efficiency_na() -> None:
    result = _run_r(
        _trajectory_setup()
        + "x[46] <- NA_real_; y[46] <- NA_real_; z[46] <- NA_real_\n"
        + "metrics <- trajectory_metrics(t,x,y,z,fs=fs)\nresult <- list(value=metrics$efficiency)\n"
    )
    assert math.isnan(_numeric(result["value"]))


def test_velocity_estimands_use_only_fully_observed_intervals() -> None:
    result = _run_r(
        textwrap.dedent(
            """
            fs <- 10; t <- 0:6 / fs
            x <- c(0, 1, 3, NA, 103, 107, 112); y <- rep(0,7); z <- rep(0,7)
            metrics <- trajectory_metrics(t,x,y,z,fs=fs)
            result <- list(v_mean=metrics$v_mean, v_peak=metrics$v_peak)
            """
        )
    )
    distances = [1.0, 2.0, 4.0, 5.0]
    expected_mean = sum(distances) / (len(distances) * 0.1)
    assert float(result["v_mean"]) == pytest.approx(expected_mean, abs=1e-12)
    assert float(result["v_peak"]) == pytest.approx(50.0, abs=1e-12)


def test_dropout_fraction_and_longest_gap_duration() -> None:
    result = _run_r(
        _trajectory_setup()
        + textwrap.dedent(
            """
            x[45:47] <- NA_real_; y[45:47] <- NA_real_; z[45:47] <- NA_real_
            metrics <- trajectory_metrics(t,x,y,z,fs=fs)
            result <- list(dropout=metrics$dropout, longest=metrics$longest_gap_sec)
            """
        )
    )
    assert float(result["dropout"]) == pytest.approx(3 / 91, abs=1e-15)
    assert float(result["longest"]) == pytest.approx(3 / 30, abs=1e-15)


def test_gap_bias_probe_corpus_is_bounded_and_gapfree_exact() -> None:
    result = _run_r(
        _trajectory_setup("csv4")
        + _legacy_metrics_r()
        + textwrap.dedent(
            """
            patterns <- list(
              gapfree=integer(0), drop1_mid=45, drop3_mid=44:46,
              drop8_mid=42:49, drop15_mid=38:52,
              scatter15=c(5,10,15,20,25,30,35,40,50,55,60,65,70,75,85)
            )
            rows <- lapply(patterns, function(drop) {
              xx <- x; yy <- y; zz <- z
              at <- drop + 1
              xx[at] <- NA_real_; yy[at] <- NA_real_; zz[at] <- NA_real_
              trajectory_metrics(t,xx,yy,zz,fs=fs)
            })
            result <- list(
              gapfree_identical=unname(vapply(names(legacy), function(k) identical(rows$gapfree[[k]], legacy[[k]]), logical(1))),
              nj=unname(vapply(rows, function(m) m$nj, numeric(1)))
            )
            """
        )
    )
    assert result["gapfree_identical"] == [True] * 5
    nj = [_numeric(value) for value in cast(list[Any], result["nj"])]
    patterns = [
        [],
        [45],
        [44, 45, 46],
        list(range(42, 50)),
        list(range(38, 53)),
        [5, 10, 15, 20, 25, 30, 35, 40, 50, 55, 60, 65, 70, 75, 85],
    ]
    for observed, dropped in zip(nj, patterns, strict=True):
        assert observed == pytest.approx(_masked_nj_expected(dropped), rel=1e-11, abs=1e-10)
    assert 16.455 < nj[0] < 16.456
    assert 0.75 <= nj[1] / nj[0] <= 1.25
    assert all(0 <= value < 3 * nj[0] for value in nj[2:])


def test_m3u1_adds_no_dropout_or_grid_evidence_to_2d_schema() -> None:
    text = _CLINICAL_R.read_text()
    emitted = set(re.findall(r'row\[\[paste0\(side, "_([^"]+)"\)\]\]', text))
    assert {"dropout", "longest_gap_sec", "skipped_slots"}.isdisjoint(emitted)
    assert emitted >= _WINDOW_METRIC_SUFFIXES


def test_arthrose_uses_namespace_qualified_base_filter() -> None:
    text = _ARTHROSE_R.read_text()
    assert "library(zoo)" not in text
    assert re.search(r"stats::filter\s*\(\s*angle_index", text)
    assert "filter(angle_index" not in text.replace("stats::filter(angle_index", "")
    assert '"zoo"' not in _PIPELINE_TEST.read_text()


@pytest.mark.parametrize(
    "values",
    [
        [1, 2, 3, 4, 5, 6, 7],
        [None, 2, 3, 4, 5, 6, 7],
        [1, 2, None, 4, 5, 6, 7],
        [1, 2, 3, 4, 5, 6, None],
        [1, None, 3, 4, None, 6, 7],
    ],
)
def test_base_centered_filter_matches_zoo_na_semantics(values: list[float | None]) -> None:
    result = _run_r(
        textwrap.dedent(
            f"""
            x <- {_r_literal([math.nan if value is None else value for value in values])}
            expected <- zoo::rollmean(x, 5, fill=NA, align="center")
            actual <- as.numeric(stats::filter(x, rep(1/5,5), sides=2))
            finite <- is.finite(expected)
            scale <- pmax(1, abs(expected[finite]))
            delta <- if (any(finite)) max(abs(actual[finite]-expected[finite]) / scale) else 0
            result <- list(mask=identical(is.na(actual),is.na(expected)), delta=delta)
            """
        ),
        clinical=False,
    )
    assert result["mask"] is True
    assert float(result["delta"]) <= 4 * math.ulp(1.0)


def test_renv_project_is_consistent() -> None:
    result = _run_r(
        "status <- capture.output(info <- renv::status())\nresult <- list(synchronized=isTRUE(info$synchronized), status=paste(status,collapse=' | '))\n",
        clinical=False,
    )
    assert result["synchronized"] is True, result["status"]


def test_every_analysis_import_is_recorded_in_renv_lock() -> None:
    lock_packages = set(json.loads(_RENV_LOCK.read_text())["Packages"])
    imported: set[str] = set()
    patterns = [
        re.compile(r"(?:library|require)\s*\(\s*[\"']?([A-Za-z][A-Za-z0-9.]*)"),
        re.compile(r"\b([A-Za-z][A-Za-z0-9.]*):::{0,1}[A-Za-z][A-Za-z0-9._]*"),
    ]
    for path in (_PROJECT_ROOT / "analysis").glob("*"):
        if path.suffix not in {".R", ".Rmd"}:
            continue
        text = re.sub(r"#.*", "", path.read_text())
        for pattern in patterns:
            imported.update(pattern.findall(text))
    imported -= {
        "base",
        "stats",
        "tools",
        "utils",
        "grDevices",
        "graphics",
        "methods",
        "parallel",
    }
    assert imported <= lock_packages, f"missing from renv.lock: {sorted(imported - lock_packages)}"
