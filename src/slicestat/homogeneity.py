"""A single test of whether the metric differs across slices at all.

Per-slice gap tests compare each slice with the rest of the data, and the rest of a healthy slice
contains whichever slice is deviant, so one bad slice makes the healthy ones look good (see
:mod:`slicestat.gaps`). When the question is "is performance heterogeneous?" rather than "which
slice?", one test answers it with exact level and no multiplicity: the size-weighted dispersion
of the slice metrics around the overall metric,

    D = sum_g (n_g / n) * (metric(g) - metric(all))^2

with a null distribution from permuting slice labels across rows. ``D`` reduces to the
between-group sum of squares of a one-way ANOVA when the metric is a mean.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from slicestat.bootstrap import IntArray
from slicestat.metrics import FloatArray, Metric, get_metric
from slicestat.slices import _as_float, encode_groups

_TIE_RTOL = 1e-12


@dataclass(frozen=True)
class HomogeneityResult:
    metric: str
    statistic: float
    p_value: float
    n_perm: int
    n_slices: int
    slices_used: int


def dispersion(
    metric: Metric, y_true: FloatArray, y_pred: FloatArray, members: Sequence[IntArray]
) -> float:
    """Size-weighted dispersion of slice metrics around the overall metric.

    Slices whose metric is undefined are left out and the weights renormalised over the rest.
    """
    overall = metric(y_true, y_pred)
    num = 0.0
    weight = 0
    for idx in members:
        v = metric(y_true[idx], y_pred[idx])
        if math.isfinite(v):
            num += idx.size * (v - overall) ** 2
            weight += idx.size
    return num / weight if weight else math.nan


def homogeneity_test(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    groups: ArrayLike | Sequence[Hashable],
    metric: str | Metric = "accuracy",
    *,
    n_perm: int = 2000,
    min_size: int = 1,
    seed: int | None = 0,
) -> HomogeneityResult:
    """Permutation test of the null that every slice shares one metric value.

    Rows of slices smaller than ``min_size`` are dropped before testing.
    """
    if n_perm < 1:
        raise ValueError("n_perm must be positive")
    name, fn = get_metric(metric)
    yt = _as_float(y_true, "y_true")
    yp = _as_float(y_pred, "y_pred")
    if yt.shape != yp.shape:
        raise ValueError(f"y_true and y_pred differ in length: {yt.size} vs {yp.size}")
    _, members = encode_groups(groups)
    if sum(m.size for m in members) != yt.size:
        raise ValueError("groups must have one label per row")
    n_slices = len(members)
    used = [m for m in members if m.size >= min_size]
    if len(used) < 2:
        raise ValueError("need at least two slices of at least min_size rows")

    keep = np.concatenate(used)
    yt, yp = yt[keep], yp[keep]
    sizes = [m.size for m in used]
    bounds = np.cumsum([0, *sizes])
    contiguous = [np.arange(bounds[i], bounds[i + 1], dtype=np.intp) for i in range(len(used))]

    obs = dispersion(fn, yt, yp, contiguous)
    if not math.isfinite(obs):
        return HomogeneityResult(name, obs, math.nan, n_perm, n_slices, len(used))
    threshold = obs * (1 - _TIE_RTOL) - 1e-15
    rng = np.random.default_rng(seed)
    hits = valid = 0
    for _ in range(n_perm):
        perm = rng.permutation(yt.size)
        null = dispersion(fn, yt[perm], yp[perm], contiguous)
        if math.isfinite(null):
            valid += 1
            hits += null >= threshold
    p = (1 + hits) / (1 + valid) if valid else math.nan
    return HomogeneityResult(name, obs, p, n_perm, n_slices, len(used))
