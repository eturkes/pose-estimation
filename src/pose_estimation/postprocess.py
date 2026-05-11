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
import csv
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
    """Import pandas, numpy, and scipy at call time; raise a clear error if missing."""
    try:
        import numpy as np
        import pandas as pd
    except ImportError as e:
        raise ImportError("postprocess requires pandas. Install with: pip install pandas") from e
    try:
        from scipy.signal import savgol_filter
    except ImportError as e:
        raise ImportError("postprocess requires scipy. Install with: pip install scipy") from e
    return np, pd, savgol_filter


# Columns that hold landmark coordinates (to be smoothed).
# Visibility columns are left untouched.
_COORD_SUFFIXES = ("_x", "_y", "_z")


def _format_float_col(arr, fmt, np):
    """Vectorised float→str column using str.format on ``arr.tolist()``.

    Pandas' ``to_csv`` formats cell-by-cell with an inline NaN check;
    that loop dominates wall time on landmark-sized blocks.  This
    helper batches the column: one C-level ``isnan`` reduction up
    front, then ``map(format, arr.tolist())`` which keeps the
    per-element work inside CPython's tight ``map`` loop.
    """
    lst = arr.tolist()
    if not np.isnan(arr).any():
        return list(map(fmt, lst))
    return [fmt(v) if v == v else "" for v in lst]


def _fast_write_csv(df, out_path, np, float_format="%.6f", coord_overrides=None):
    """Write *df* to CSV bypassing pandas' per-cell NaN formatting.

    Pandas' ``DataFrame.to_csv`` dispatches through a Python-level loop
    that calls ``notna`` and the float formatter on every cell, even
    when the column has no missing values.  That dominates runtime on
    landmark-sized frames (≈85 % of the 1800x99 case).

    This helper materialises each column once with ``str.format`` (a
    C-implemented method) using ``arr.tolist()`` to skip per-element
    numpy → Python conversion, then writes everything through
    ``csv.writer.writerows``.  Output is byte-identical to
    ``df.to_csv(out_path, index=False, float_format=float_format)``
    for the dtypes encountered here (numeric + string).

    *coord_overrides*, when given, maps column name → 1-D numpy array
    of values to use *instead* of ``df[c]``.  This lets the caller
    keep smoothed coord data in a standalone numpy block and skip the
    expensive ``df.loc[idx, coord_cols] = smoothed`` write-back.
    """
    # '%.6f' -> '{:.6f}'  for str.format
    spec = "{:" + float_format[1:] + "}"
    fmt = spec.format
    overrides = coord_overrides or {}
    str_cols = []
    for c in df.columns:
        override = overrides.get(c)
        if override is not None:
            str_cols.append(_format_float_col(override, fmt, np))
            continue
        arr = df[c].to_numpy()
        kind = arr.dtype.kind
        if kind == "f":
            str_cols.append(_format_float_col(arr, fmt, np))
        elif kind in ("i", "u", "b"):
            str_cols.append(list(map(str, arr.tolist())))
        else:
            # String / object columns: pandas writes ``str(value)`` and so do we.
            str_cols.append(list(map(str, arr)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(list(df.columns))
        w.writerows(zip(*str_cols, strict=False))

# Gaps longer than this (relative to window) are treated as segment
# boundaries rather than being interpolated through.
_GAP_RATIO = 0.5


def _smoothable_columns(columns):
    """Return the subset of column names that are landmark coordinates."""
    return [c for c in columns if c.endswith(_COORD_SUFFIXES)]


def _largest_valid_window(n_rows, window, polyorder):
    """Return the largest odd window ≤ *window* that fits in *n_rows*.

    Returns ``None`` when no usable window exists.
    """
    if n_rows >= window:
        return window
    w = n_rows if n_rows % 2 == 1 else n_rows - 1
    if w < polyorder + 2:
        return None
    return w


def _smooth_block_nofallback(block, window, polyorder, savgol_filter):
    """Vectorised savgol over a 2-D float array, axis=0.

    Returns the smoothed block, or *block* unchanged when the window
    cannot be adjusted to fit.  Caller guarantees no NaN values.
    """
    n_rows = block.shape[0]
    w = _largest_valid_window(n_rows, window, polyorder)
    if w is None:
        return block
    return savgol_filter(block, w, polyorder, axis=0)


def _smooth_column_with_nan(series, window, polyorder, np, savgol_filter):
    """Apply Savitzky-Golay to one numeric Series, handling NaN gaps.

    Short gaps (< window * _GAP_RATIO) are linearly interpolated before
    filtering.  Long gaps cause the column to be split into contiguous
    segments that are filtered independently.
    """
    values = series.to_numpy(dtype=float, copy=False)
    n = values.shape[0]
    if n == 0:
        return series

    nan_mask = np.isnan(values)
    if nan_mask.all():
        return series

    max_gap = max(1, int(window * _GAP_RATIO))

    # Interpolate short NaN runs in place on a copy
    work = values.copy()
    if nan_mask.any():
        # Locate run boundaries: True where nan_mask changes value
        boundaries = np.empty(n + 1, dtype=bool)
        boundaries[0] = True
        boundaries[1:n] = nan_mask[1:] != nan_mask[:-1]
        boundaries[n] = True
        run_starts = np.flatnonzero(boundaries[:-1])
        run_ends = np.flatnonzero(boundaries[1:])  # exclusive
        for start, end in zip(run_starts, run_ends, strict=False):
            if not nan_mask[start]:
                continue
            run_len = end - start
            if run_len > max_gap or start == 0 or end == n:
                continue
            # Linear interpolation across short interior gap
            left = work[start - 1]
            right = work[end]
            if np.isnan(left) or np.isnan(right):
                continue
            steps = np.arange(1, run_len + 1, dtype=float)
            work[start:end] = left + (right - left) * (steps / (run_len + 1))

    # Process contiguous non-NaN segments
    valid = ~np.isnan(work)
    if not valid.any():
        return series

    boundaries = np.empty(n + 1, dtype=bool)
    boundaries[0] = True
    boundaries[1:n] = valid[1:] != valid[:-1]
    boundaries[n] = True
    seg_starts = np.flatnonzero(boundaries[:-1])
    seg_ends = np.flatnonzero(boundaries[1:])

    for start, end in zip(seg_starts, seg_ends, strict=False):
        if not valid[start]:
            continue
        seg_len = end - start
        w = _largest_valid_window(seg_len, window, polyorder)
        if w is None:
            continue
        work[start:end] = savgol_filter(work[start:end], w, polyorder)

    return series.__class__(work, index=series.index, name=series.name)


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
    np, pd, savgol_filter = _lazy_imports()

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

    # Only run ``pd.to_numeric`` on columns that aren't already numeric.
    # ``pd.read_csv`` on landmark CSVs returns float64 for the coord
    # columns directly, so this apply was a multi-ms identity copy.
    obj_coord_cols = [c for c in coord_cols if df[c].dtype.kind not in "fi"]
    if obj_coord_cols:
        df[obj_coord_cols] = df[obj_coord_cols].apply(pd.to_numeric, errors="coerce")

    # Smooth into a dedicated coord block.  Keeping the smoothed data in
    # a numpy array (rather than writing it back via ``df.loc``) avoids
    # the ~18 ms-per-call setitem cost on the 1800x99 case — pandas'
    # indexer rebuilds the column blocks even for full-slice writes.
    coord_arr = df[coord_cols].to_numpy(dtype=float, copy=True)

    group_keys = [k for k in ("video", "person_idx") if k in df.columns]
    if group_keys:
        groups = df.groupby(group_keys, sort=False)
    else:
        groups = [(None, df)]

    # ``df.index.get_indexer`` is the safe way to map a pandas Index to
    # positional offsets when the index isn't a contiguous RangeIndex.
    base_index = df.index
    for _, grp in groups:
        positional = base_index.get_indexer(grp.index)
        block = coord_arr[positional]

        if not np.isnan(block).any():
            coord_arr[positional] = _smooth_block_nofallback(
                block, window, polyorder, savgol_filter
            )
            continue

        # Mixed case: split columns into all-finite vs has-NaN buckets.
        col_has_nan = np.isnan(block).any(axis=0)
        clean_idx = np.flatnonzero(~col_has_nan)
        dirty_idx = np.flatnonzero(col_has_nan)

        if clean_idx.size:
            sub = block[:, clean_idx]
            smoothed_clean = _smooth_block_nofallback(sub, window, polyorder, savgol_filter)
            coord_arr[np.ix_(positional, clean_idx)] = smoothed_clean

        for j in dirty_idx:
            c = coord_cols[j]
            smoothed_col = _smooth_column_with_nan(
                grp[c], window, polyorder, np, savgol_filter
            )
            coord_arr[positional, j] = smoothed_col.to_numpy()

    # ``_fast_write_csv`` reads non-coord columns from ``df`` and pulls
    # the smoothed coord columns from ``coord_arr`` via the overrides
    # dict — keeps the smoothed data out of pandas entirely.
    overrides = {c: coord_arr[:, j] for j, c in enumerate(coord_cols)}
    _fast_write_csv(df, output_path, np, coord_overrides=overrides)


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
