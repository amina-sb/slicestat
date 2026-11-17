"""Stratified bootstrap and the interval constructions built on it.

The resampling is stratified by slice: every replicate draws ``n_g`` rows with replacement from
slice ``g`` and from nowhere else, so each replicate has exactly the slice sizes of the observed
data. An unstratified bootstrap of the whole dataset lets a 30-row slice come back with 20 rows in
one replicate and 41 in the next; that size variation is not sampling error of the metric and it
widens the slice's interval. The overall metric of a replicate is computed on the union of the
per-slice draws.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray

from slicestat.metrics import FloatArray, Metric

IntArray = NDArray[np.intp]

_N = NormalDist()


def stratified_replicates(
    metric: Metric,
    y_true: FloatArray,
    y_pred: FloatArray,
    members: list[IntArray],
    n_boot: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Bootstrap replicates of ``metric`` for every slice and for the whole data.

    ``members[g]`` holds the row indices of slice ``g``. Returns an array of shape
    ``(n_boot, len(members) + 1)``; the last column is the overall metric.
    """
    n_groups = len(members)
    out = np.empty((n_boot, n_groups + 1), dtype=np.float64)
    for b in range(n_boot):
        draws = [idx[rng.integers(0, idx.size, idx.size)] for idx in members]
        for g, d in enumerate(draws):
            out[b, g] = metric(y_true[d], y_pred[d])
        both = np.concatenate(draws)
        out[b, n_groups] = metric(y_true[both], y_pred[both])
    return out


def jackknife(metric: Metric, y_true: FloatArray, y_pred: FloatArray) -> FloatArray:
    """Leave-one-out values of ``metric``. Costs ``n`` metric evaluations."""
    n = y_true.size
    keep = np.ones(n, dtype=bool)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        keep[i] = False
        out[i] = metric(y_true[keep], y_pred[keep])
        keep[i] = True
    return out


def percentile_interval(replicates: FloatArray, alpha: float) -> tuple[float, float]:
    """Equal-tailed percentile interval over the finite replicates."""
    r = replicates[np.isfinite(replicates)]
    if r.size == 0:
        return math.nan, math.nan
    lo, hi = np.quantile(r, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def bca_interval(
    replicates: FloatArray,
    estimate: float,
    jack: FloatArray,
    alpha: float,
) -> tuple[float, float]:
    """Bias-corrected and accelerated interval (Efron 1987; Efron & Tibshirani 1993, ch. 14).

    The bias correction uses the mid-rank proportion ``P(theta* < theta) + P(theta* = theta)/2``,
    which keeps it finite for metrics with a discrete bootstrap distribution such as accuracy on
    a small slice. The acceleration comes from the jackknife values ``jack``. Returns ``nan``
    bounds when the bootstrap distribution lies entirely on one side of the estimate.
    """
    r = replicates[np.isfinite(replicates)]
    if r.size == 0 or not math.isfinite(estimate):
        return math.nan, math.nan
    prop = (np.sum(r < estimate) + 0.5 * np.sum(r == estimate)) / r.size
    if prop <= 0.0 or prop >= 1.0:
        return math.nan, math.nan
    z0 = _N.inv_cdf(float(prop))

    j = jack[np.isfinite(jack)]
    a = 0.0
    if j.size >= 2:
        d = j.mean() - j
        den = 6.0 * float(np.sum(d**2)) ** 1.5
        if den > 0:
            a = float(np.sum(d**3)) / den

    bounds = []
    for q in (alpha / 2, 1 - alpha / 2):
        z = _N.inv_cdf(q)
        adj = _N.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        bounds.append(float(np.quantile(r, adj)))
    return bounds[0], bounds[1]
