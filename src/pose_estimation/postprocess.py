#!/usr/bin/env python3
"""Savitzky-Golay post-processing for batch CSV output.

Applies a second-pass Savitzky-Golay smoothing filter over exported
landmark CSVs.  Unlike the real-time One Euro Filter, Savitzky-Golay
uses both past and future samples and preserves peaks better, making it
ideal for offline refinement.

Standalone usage:
    python -m pose_estimation.postprocess output/video1.csv --window 15 --polyorder 3
"""

import argparse
import pathlib


def _odd_int(value):
    """argparse type that requires an odd integer."""
    try:
        ivalue = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if ivalue % 2 == 0:
        raise argparse.ArgumentTypeError(
            f"window length must be odd (got {ivalue}); try {ivalue + 1} or {ivalue - 1}"
        )
    if ivalue < 3:
        raise argparse.ArgumentTypeError(f"window length must be ≥ 3, got {ivalue}")
    return ivalue


def _lazy_imports():
    """Import pandas and scipy at call time; raise a clear error if missing."""
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("postprocess requires pandas. Install with: pip install pandas") from e
    try:
        from scipy.signal import savgol_filter
    except ImportError as e:
        raise ImportError("postprocess requires scipy. Install with: pip install scipy") from e
    return pd, savgol_filter


# Columns that hold landmark coordinates (to be smoothed).
# Visibility columns are left untouched.
_COORD_SUFFIXES = ("_x", "_y", "_z")

# Gaps longer than this (relative to window) are treated as segment
# boundaries rather than being interpolated through.
_GAP_RATIO = 0.5


def _smoothable_columns(columns):
    """Return the subset of column names that are landmark coordinates."""
    return [c for c in columns if any(c.endswith(s) for s in _COORD_SUFFIXES)]


def _smooth_column(series, window, polyorder, savgol_filter):
    """Apply Savitzky-Golay to one numeric Series, handling NaN gaps.

    Short gaps (< window * _GAP_RATIO) are linearly interpolated before
    filtering.  Long gaps cause the column to be split into contiguous
    segments that are filtered independently.
    """
    if series.isna().all():
        return series

    max_gap = max(1, int(window * _GAP_RATIO))

    # Try interpolating short gaps
    interp = series.copy()
    # Identify gap runs
    is_nan = series.isna()
    if is_nan.any():
        groups = (is_nan != is_nan.shift(fill_value=False)).cumsum()
        gap_lengths = is_nan.groupby(groups).transform("sum")
        short_gap = is_nan & (gap_lengths <= max_gap)
        if short_gap.any():
            safe_limit = min(max_gap, len(series) - 1) or 1
            interp = series.interpolate(limit=safe_limit, limit_area="inside")

    # After interpolation, identify contiguous non-NaN segments
    valid = interp.notna()
    if not valid.any():
        return series

    seg_id = (valid != valid.shift(fill_value=False)).cumsum()
    result = interp.copy()

    for _sid, grp in interp.groupby(seg_id):
        if grp.isna().all():
            continue
        n = len(grp)
        if n < window:
            # Use the largest valid odd window, or skip if too short
            w = n if n % 2 == 1 else n - 1
            if w < polyorder + 2:
                continue
            result.loc[grp.index] = savgol_filter(grp.values, w, polyorder)
        else:
            result.loc[grp.index] = savgol_filter(grp.values, window, polyorder)

    return result


def savgol_smooth_csv(input_path, output_path, window=11, polyorder=3):
    """Apply Savitzky-Golay smoothing to landmark columns in a CSV.

    Parameters
    ----------
    input_path : str or Path
        Path to the source CSV (written by export.py).
    output_path : str or Path
        Path to write the smoothed CSV.
    window : int
        Window length for the filter (must be odd, default 11).
    polyorder : int
        Polynomial order for the filter (default 3).
    """
    pd, savgol_filter = _lazy_imports()

    if window % 2 == 0:
        raise ValueError(f"window must be odd, got {window}")
    if polyorder >= window:
        raise ValueError(f"polyorder ({polyorder}) must be less than window ({window})")

    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)

    df = pd.read_csv(input_path)
    if df.empty:
        df.to_csv(output_path, index=False)
        return

    coord_cols = _smoothable_columns(df.columns)
    if not coord_cols:
        df.to_csv(output_path, index=False)
        return

    # Convert coordinate columns to numeric (blanks become NaN)
    for c in coord_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    group_keys = ["video", "person_idx"]
    present_keys = [k for k in group_keys if k in df.columns]

    if present_keys:
        for _, grp in df.groupby(present_keys):
            idx = grp.index
            for c in coord_cols:
                df.loc[idx, c] = _smooth_column(
                    grp[c],
                    window,
                    polyorder,
                    savgol_filter,
                )
    else:
        for c in coord_cols:
            df[c] = _smooth_column(df[c], window, polyorder, savgol_filter)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Savitzky-Golay post-processing for landmark CSVs",
    )
    parser.add_argument("input", help="Input CSV path")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV path (default: <stem>_smooth.csv alongside input)",
    )
    parser.add_argument(
        "--window",
        type=_odd_int,
        default=11,
        help="Filter window length, must be odd (default: 11)",
    )
    parser.add_argument(
        "--polyorder",
        type=int,
        default=3,
        help="Polynomial order (default: 3)",
    )
    args = parser.parse_args()

    inp = pathlib.Path(args.input)
    if args.output:
        out = pathlib.Path(args.output)
    else:
        out = inp.with_name(f"{inp.stem}_smooth.csv")

    print(f"Input:     {inp}")
    print(f"Output:    {out}")
    print(f"Window:    {args.window}")
    print(f"Polyorder: {args.polyorder}")

    savgol_smooth_csv(inp, out, window=args.window, polyorder=args.polyorder)
    print("Done.")


if __name__ == "__main__":
    main()
