"""Shared validity-gated linear assignment helpers."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def gated_assignment(cost, *, threshold=None, valid_mask=None):
    """Return a maximum-cardinality, minimum-cost valid assignment.

    ``linear_sum_assignment`` always assigns the smaller side of a rectangular
    matrix.  Filtering those pairs *after* solving can discard a valid
    alternative assignment.  This helper instead marks invalid edges before
    solving and adds one dummy (unmatched) column per row.  The dummy penalty
    dominates the total scaled cost of every real edge, so the solver first
    maximises the number of valid real pairs and then minimises their cost.

    Non-finite costs are always invalid.  ``threshold`` is a strict upper
    bound, matching the trackers' historical ``cost < threshold`` semantics.
    ``valid_mask`` can impose additional pairwise policy such as hand-wrist
    distality.
    """
    costs = np.asarray(cost, dtype=np.float64)
    if costs.ndim != 2:
        raise ValueError(f"cost must be a 2-D matrix, got shape {costs.shape}")

    n_rows, n_cols = costs.shape
    if n_rows == 0 or n_cols == 0:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty.copy()

    valid = np.isfinite(costs)
    if threshold is not None:
        valid &= costs < threshold
    if valid_mask is not None:
        extra_valid = np.asarray(valid_mask, dtype=bool)
        if extra_valid.shape != costs.shape:
            raise ValueError(
                f"valid_mask shape {extra_valid.shape} does not match cost shape {costs.shape}"
            )
        valid &= extra_valid

    if not valid.any():
        empty = np.empty(0, dtype=np.intp)
        return empty, empty.copy()

    # Affinely scale valid costs to [0, 1].  For a fixed cardinality this
    # preserves the minimum-cost assignment, while keeping the lexicographic
    # penalties finite and well-conditioned even for very large input costs.
    valid_costs = costs[valid]
    min_cost = float(valid_costs.min())
    span = float(valid_costs.max() - min_cost)
    scaled = np.zeros_like(costs)
    if span > 0.0:
        scaled[valid] = (valid_costs - min_cost) / span

    max_pairs = min(n_rows, n_cols)
    dummy_cost = float(max_pairs + 1)
    invalid_cost = dummy_cost * float(n_rows + 1)

    augmented = np.full((n_rows, n_cols + n_rows), invalid_cost, dtype=np.float64)
    augmented[:, :n_cols][valid] = scaled[valid]
    augmented[:, n_cols:] = dummy_cost

    row_ind, col_ind = linear_sum_assignment(augmented)
    real = col_ind < n_cols
    row_ind = row_ind[real]
    col_ind = col_ind[real]
    accepted = valid[row_ind, col_ind]
    return row_ind[accepted], col_ind[accepted]
