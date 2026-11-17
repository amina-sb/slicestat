"""Per-slice metric estimates with bootstrap intervals."""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from slicestat.bootstrap import (
    IntArray,
    bca_interval,
    jackknife,
    percentile_interval,
    stratified_replicates,
)
from slicestat.metrics import FloatArray, Metric, get_metric

OVERALL = "overall"
IntervalMethod = Literal["percentile", "bca"]


@dataclass(frozen=True)
class SliceRow:
    """One slice's estimate. ``undefined_share`` is the fraction of bootstrap replicates in
    which the metric was undefined (for example a resample of a small slice with no positives);
    when it is large the interval describes only the replicates where the metric existed."""

    group: Hashable
    n: int
    estimate: float
    low: float
    high: float
    undefined_share: float


@dataclass(frozen=True)
class SliceReport:
    metric: str
    method: IntervalMethod
    alpha: float
    n_boot: int
    rows: list[SliceRow] = field(default_factory=list)

    @property
    def overall(self) -> SliceRow:
        return self.rows[-1]

    def slices(self) -> list[SliceRow]:
        """Rows for the slices, excluding the overall row."""
        return self.rows[:-1]

    def to_records(self) -> list[dict[str, object]]:
        return [
            {
                "group": r.group,
                "n": r.n,
                "estimate": r.estimate,
                "low": r.low,
                "high": r.high,
                "undefined_share": r.undefined_share,
            }
            for r in self.rows
        ]

    def to_markdown(self, digits: int = 3) -> str:
        level = round(100 * (1 - self.alpha))
        head = f"| slice | n | {self.metric} | {level}% CI ({self.method}) |"
        lines = [head, "|---|---:|---:|---|"]
        for r in self.rows:
            ci = "undefined" if math.isnan(r.low) else f"{r.low:.{digits}f}–{r.high:.{digits}f}"
            est = "undefined" if math.isnan(r.estimate) else f"{r.estimate:.{digits}f}"
            lines.append(f"| {r.group} | {r.n} | {est} | {ci} |")
        return "\n".join(lines)


def _as_float(a: ArrayLike, name: str) -> FloatArray:
    x = np.asarray(a, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {x.shape}")
    return x


def encode_groups(groups: ArrayLike | Sequence[Hashable]) -> tuple[list[Hashable], list[IntArray]]:
    """Return the distinct slice labels (sorted) and the row indices belonging to each."""
    g = np.asarray(groups, dtype=object)
    if g.ndim != 1:
        raise ValueError(f"groups must be one-dimensional, got shape {g.shape}")
    for v in g:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            raise ValueError("groups contains a missing label; give missing values their own label")
    rows_of: dict[Hashable, list[int]] = {}
    for i, v in enumerate(g.tolist()):
        rows_of.setdefault(v, []).append(i)
    try:
        labels = sorted(rows_of)  # type: ignore[type-var]
    except TypeError:  # mixed label types: fall back to a stable textual order
        labels = sorted(rows_of, key=str)
    members = [np.asarray(rows_of[lab], dtype=np.intp) for lab in labels]
    return labels, members


def slice_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    groups: ArrayLike | Sequence[Hashable],
    metric: str | Metric = "accuracy",
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    method: IntervalMethod = "percentile",
    seed: int | None = 0,
) -> SliceReport:
    """Compute ``metric`` on every slice defined by ``groups`` and on the whole data.

    Intervals come from a bootstrap stratified by slice (see :mod:`slicestat.bootstrap`).
    ``method="bca"`` additionally runs a jackknife per slice, which costs ``n`` metric
    evaluations for a slice of ``n`` rows.

    Returns a :class:`SliceReport` whose last row is the overall population.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    name, fn = get_metric(metric)
    yt = _as_float(y_true, "y_true")
    yp = _as_float(y_pred, "y_pred")
    if yt.shape != yp.shape:
        raise ValueError(f"y_true and y_pred differ in length: {yt.size} vs {yp.size}")
    labels, members = encode_groups(groups)
    if sum(m.size for m in members) != yt.size:
        raise ValueError("groups must have one label per row")

    rng = np.random.default_rng(seed)
    reps = stratified_replicates(fn, yt, yp, members, n_boot, rng)

    everyone = np.arange(yt.size, dtype=np.intp)
    rows: list[SliceRow] = []
    for col, (label, idx) in enumerate([*zip(labels, members, strict=True), (OVERALL, everyone)]):
        est = fn(yt[idx], yp[idx])
        r = reps[:, col]
        if method == "percentile":
            lo, hi = percentile_interval(r, alpha)
        elif method == "bca":
            lo, hi = bca_interval(r, est, jackknife(fn, yt[idx], yp[idx]), alpha)
        else:
            raise ValueError(f"unknown interval method {method!r}")
        if math.isnan(est):
            lo, hi = math.nan, math.nan
        rows.append(
            SliceRow(
                group=label,
                n=int(idx.size),
                estimate=est,
                low=lo,
                high=hi,
                undefined_share=float(np.mean(~np.isfinite(r))),
            )
        )
    return SliceReport(metric=name, method=method, alpha=alpha, n_boot=n_boot, rows=rows)
