"""Adjusted p-values for a family of slice tests.

``bonferroni`` and ``holm`` control the family-wise error rate (the probability of flagging any
slice when none differs). ``bh`` (Benjamini & Hochberg 1995) controls the false discovery rate
(the expected share of flagged slices that do not differ), which is the weaker guarantee but the
more useful one when many slices are examined and a few false flags are acceptable.

All three return adjusted p-values in the input order, capped at 1, so a slice is rejected at
level ``alpha`` exactly when its adjusted p-value is ``<= alpha``. ``nan`` inputs (slices
excluded from testing) stay ``nan`` and do not count toward the family size.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from slicestat.metrics import FloatArray

Correction = Literal["none", "bonferroni", "holm", "bh"]


def _prepare(p: ArrayLike) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    arr = np.asarray(p, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("p-values must be one-dimensional")
    finite = np.isfinite(arr)
    vals = arr[finite]
    if np.any((vals < 0) | (vals > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    return arr, vals, finite


def _scatter(arr: FloatArray, finite: NDArray[np.bool_], adjusted: FloatArray) -> FloatArray:
    out = np.full_like(arr, np.nan)
    out[finite] = adjusted
    return out


def bonferroni(p: ArrayLike) -> FloatArray:
    arr, vals, finite = _prepare(p)
    return _scatter(arr, finite, np.minimum(vals * vals.size, 1.0))


def holm(p: ArrayLike) -> FloatArray:
    """Holm (1979) step-down adjustment."""
    arr, vals, finite = _prepare(p)
    m = vals.size
    order = np.argsort(vals, kind="mergesort")
    stepped = (m - np.arange(m)) * vals[order]
    adj_sorted = np.minimum(np.maximum.accumulate(stepped), 1.0)
    adj = np.empty(m)
    adj[order] = adj_sorted
    return _scatter(arr, finite, adj)


def bh(p: ArrayLike) -> FloatArray:
    """Benjamini–Hochberg (1995) step-up adjustment."""
    arr, vals, finite = _prepare(p)
    m = vals.size
    order = np.argsort(vals, kind="mergesort")
    scaled = vals[order] * m / np.arange(1, m + 1)
    adj_sorted = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)
    adj = np.empty(m)
    adj[order] = adj_sorted
    return _scatter(arr, finite, adj)


def adjust(p: ArrayLike, method: Correction) -> FloatArray:
    """Dispatch to one of the corrections; ``"none"`` returns the p-values unchanged."""
    if method == "none":
        arr, _, _ = _prepare(p)
        return np.array(arr, dtype=np.float64)
    if method == "bonferroni":
        return bonferroni(p)
    if method == "holm":
        return holm(p)
    if method == "bh":
        return bh(p)
    raise ValueError(f"unknown correction {method!r}; use none, bonferroni, holm or bh")
