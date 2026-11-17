"""Does a slice differ from the rest of the data?

For each slice ``g`` the test statistic is the gap ``metric(g) - metric(rest)``. Its null
distribution comes from reshuffling slice membership: draw ``n_g`` rows at random from the whole
dataset to play the slice and let the remaining rows play the rest. Under the null hypothesis that
the slice's ``(y_true, y_pred)`` pairs are exchangeable with everyone else's, the observed gap is
one draw from that distribution. The two-sided p-value is

    (1 + #{ |null gap| >= |observed gap| }) / (1 + n_perm)

which is never zero: with ``n_perm`` permutations the smallest attainable p-value is
``1 / (n_perm + 1)``, and reporting 0 would claim more evidence than the test can give.

The per-slice tests form a family, adjusted with :mod:`slicestat.multitest` (Holm by default).
Slices with fewer than ``min_size`` rows, or whose complement has fewer, are excluded from the
family and reported with status ``"too_small"`` rather than tested with almost no power or dropped
without a trace.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from slicestat.bootstrap import IntArray
from slicestat.metrics import FloatArray, Metric, get_metric
from slicestat.multitest import Correction, adjust
from slicestat.slices import _as_float, encode_groups

# Gaps that are equal in exact arithmetic can differ in the last bit when the same counts are
# reached by a different summation order; without a tolerance such ties are miscounted.
_TIE_RTOL = 1e-12


@dataclass(frozen=True)
class GapRow:
    group: Hashable
    n: int
    estimate: float
    rest_estimate: float
    gap: float
    p_value: float
    p_adjusted: float
    flagged: bool
    status: str  # "tested", "too_small" or "undefined"


@dataclass(frozen=True)
class GapReport:
    metric: str
    correction: Correction
    alpha: float
    n_perm: int
    min_size: int
    rows: list[GapRow] = field(default_factory=list)

    def flagged(self) -> list[GapRow]:
        return [r for r in self.rows if r.flagged]

    def to_records(self) -> list[dict[str, object]]:
        return [r.__dict__.copy() for r in self.rows]

    def to_markdown(self, digits: int = 3) -> str:
        def fmt(x: float) -> str:
            return "—" if math.isnan(x) else f"{x:.{digits}f}"

        lines = [
            f"| slice | n | {self.metric} | rest | gap | p | p ({self.correction}) | flagged |",
            "|---|---:|---:|---:|---:|---:|---:|:---:|",
        ]
        for r in self.rows:
            if r.flagged:
                flag = "yes"
            elif r.status == "too_small":
                flag = "excluded: too small"
            else:
                flag = ""
            lines.append(
                f"| {r.group} | {r.n} | {fmt(r.estimate)} | {fmt(r.rest_estimate)} | "
                f"{fmt(r.gap)} | {fmt(r.p_value)} | {fmt(r.p_adjusted)} | {flag} |"
            )
        return "\n".join(lines)


def permutation_gap_pvalue(
    metric: Metric,
    y_true: FloatArray,
    y_pred: FloatArray,
    in_slice: NDArray[np.bool_],
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Observed gap and its two-sided permutation p-value for one slice mask.

    Permutations in which the gap is undefined are discarded; the p-value is computed over the
    remaining ones. Returns ``(gap, nan)`` when the observed gap itself is undefined.
    """
    n = y_true.size
    k = int(np.sum(in_slice))
    obs = metric(y_true[in_slice], y_pred[in_slice]) - metric(y_true[~in_slice], y_pred[~in_slice])
    if not math.isfinite(obs):
        return obs, math.nan
    threshold = abs(obs) * (1 - _TIE_RTOL) - 1e-15
    hits = 0
    valid = 0
    mask = np.zeros(n, dtype=bool)
    for _ in range(n_perm):
        mask[:] = False
        mask[rng.choice(n, size=k, replace=False)] = True
        null = metric(y_true[mask], y_pred[mask]) - metric(y_true[~mask], y_pred[~mask])
        if math.isfinite(null):
            valid += 1
            hits += abs(null) >= threshold
    if valid == 0:
        return obs, math.nan
    return obs, (1 + hits) / (1 + valid)


_Partial = tuple[Hashable, int, float, float, float, float, str]


def check_test_args(alpha: float, n_perm: int, min_size: int) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if n_perm < 1:
        raise ValueError("n_perm must be positive")
    if min_size < 1:
        raise ValueError("min_size must be at least 1")


def as_pair(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[FloatArray, FloatArray]:
    yt = _as_float(y_true, "y_true")
    yp = _as_float(y_pred, "y_pred")
    if yt.shape != yp.shape:
        raise ValueError(f"y_true and y_pred differ in length: {yt.size} vs {yp.size}")
    return yt, yp


def raw_gap_rows(
    fn: Metric,
    yt: FloatArray,
    yp: FloatArray,
    labels: Sequence[Hashable],
    members: Sequence[IntArray],
    n_perm: int,
    min_size: int,
    rng: np.random.Generator,
) -> list[_Partial]:
    """Unadjusted gap, p-value and status for each slice of one slicing."""
    n = yt.size
    partial: list[_Partial] = []
    for label, idx in zip(labels, members, strict=True):
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        est = fn(yt[mask], yp[mask])
        rest = fn(yt[~mask], yp[~mask])
        if idx.size < min_size or n - idx.size < min_size:
            partial.append((label, idx.size, est, rest, est - rest, math.nan, "too_small"))
            continue
        gap, p = permutation_gap_pvalue(fn, yt, yp, mask, n_perm, rng)
        status = "tested" if math.isfinite(p) else "undefined"
        partial.append((label, idx.size, est, rest, gap, p, status))
    return partial


def finish_rows(partial: Sequence[_Partial], p_adj: FloatArray, alpha: float) -> list[GapRow]:
    return [
        GapRow(
            group=label,
            n=int(size),
            estimate=est,
            rest_estimate=rest,
            gap=gap,
            p_value=p,
            p_adjusted=float(pa),
            flagged=bool(math.isfinite(pa) and pa <= alpha),
            status=status,
        )
        for (label, size, est, rest, gap, p, status), pa in zip(partial, p_adj, strict=True)
    ]


def gap_test(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    groups: ArrayLike | Sequence[Hashable],
    metric: str | Metric = "accuracy",
    *,
    n_perm: int = 2000,
    correction: Correction = "holm",
    alpha: float = 0.05,
    min_size: int = 10,
    seed: int | None = 0,
) -> GapReport:
    """Test every slice against the rest of the data and adjust for the number of slices.

    A slice is ``flagged`` when its adjusted p-value is at most ``alpha``.
    """
    check_test_args(alpha, n_perm, min_size)
    name, fn = get_metric(metric)
    yt, yp = as_pair(y_true, y_pred)
    labels, members = encode_groups(groups)
    if sum(m.size for m in members) != yt.size:
        raise ValueError("groups must have one label per row")

    rng = np.random.default_rng(seed)
    partial = raw_gap_rows(fn, yt, yp, labels, members, n_perm, min_size, rng)
    p_adj = adjust(np.array([row[5] for row in partial]), correction)
    return GapReport(
        metric=name,
        correction=correction,
        alpha=alpha,
        n_perm=n_perm,
        min_size=min_size,
        rows=finish_rows(partial, p_adj, alpha),
    )
