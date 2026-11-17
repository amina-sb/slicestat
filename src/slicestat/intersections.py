"""Slices from several attributes and their combinations, tested as one family.

Given attributes such as ``{"region": ..., "age": ..., "device": ...}``, :func:`scan` builds
every slicing of order 1 (each attribute alone) up to ``max_order`` (combinations of that many
attributes), labels each cell ``"region=east & age=young"``, and runs the slice-vs-rest gap test
on every cell.

All cells of all slicings form one family for the multiplicity adjustment. Adjusting within each
slicing separately and then reading the slicings side by side is the common alternative, and it
lets the family-wise error grow with the number of slicings; ``family="per_slicing"`` reproduces
it so the difference can be measured, and the README's evidence table measures it.
"""

from __future__ import annotations

import itertools
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from slicestat.gaps import (
    GapReport,
    GapRow,
    as_pair,
    check_test_args,
    finish_rows,
    raw_gap_rows,
)
from slicestat.metrics import FloatArray, Metric, get_metric
from slicestat.multitest import Correction, adjust
from slicestat.slices import encode_groups

Family = Literal["all", "per_slicing"]
SEP = " & "


def cell_labels(attributes: Mapping[str, ArrayLike], names: tuple[str, ...]) -> list[str]:
    """``"a=x & b=y"`` labels for the combination of the named attributes, one per row."""
    cols = [np.asarray(attributes[n], dtype=object) for n in names]
    return [
        SEP.join(f"{n}={v}" for n, v in zip(names, vals, strict=True))
        for vals in zip(*cols, strict=True)
    ]


def slicings(names: list[str], max_order: int) -> list[tuple[str, ...]]:
    """Attribute combinations of order 1 .. ``max_order``, in a stable order."""
    return [
        combo for order in range(1, max_order + 1) for combo in itertools.combinations(names, order)
    ]


@dataclass(frozen=True)
class ScanReport:
    metric: str
    correction: Correction
    family: Family
    alpha: float
    n_perm: int
    min_size: int
    slicings: list[tuple[str, ...]]
    rows: list[GapRow] = field(default_factory=list)
    slicing_of: list[str] = field(default_factory=list)

    @property
    def family_size(self) -> int:
        return sum(r.status == "tested" for r in self.rows)

    def flagged(self) -> list[GapRow]:
        return [r for r in self.rows if r.flagged]

    def as_gap_report(self) -> GapReport:
        return GapReport(
            metric=self.metric,
            correction=self.correction,
            alpha=self.alpha,
            n_perm=self.n_perm,
            min_size=self.min_size,
            rows=list(self.rows),
        )

    def to_markdown(self, digits: int = 3) -> str:
        return self.as_gap_report().to_markdown(digits)


def scan(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    attributes: Mapping[str, ArrayLike],
    metric: str | Metric = "accuracy",
    *,
    max_order: int = 2,
    n_perm: int = 2000,
    correction: Correction = "holm",
    family: Family = "all",
    alpha: float = 0.05,
    min_size: int = 20,
    seed: int | None = 0,
) -> ScanReport:
    """Gap-test every cell of every slicing up to ``max_order`` attributes.

    ``min_size`` defaults higher than in :func:`slicestat.gap_test` because intersections
    produce many small cells, each of which would add a test with little power to the family.
    """
    check_test_args(alpha, n_perm, min_size)
    if max_order < 1:
        raise ValueError("max_order must be at least 1")
    if family not in ("all", "per_slicing"):
        raise ValueError(f"unknown family {family!r}; use all or per_slicing")
    if not attributes:
        raise ValueError("attributes is empty")
    name, fn = get_metric(metric)
    yt, yp = as_pair(y_true, y_pred)
    names = list(attributes)
    for n in names:
        if np.asarray(attributes[n], dtype=object).shape != yt.shape:
            raise ValueError(f"attribute {n!r} must have one value per row")
    combos = slicings(names, min(max_order, len(names)))

    rng = np.random.default_rng(seed)
    partial: list[tuple[Hashable, int, float, float, float, float, str]] = []
    owner: list[str] = []
    per_slicing_p: list[FloatArray] = []
    for combo in combos:
        labels, members = encode_groups(cell_labels(attributes, combo))
        rows = raw_gap_rows(fn, yt, yp, labels, members, n_perm, min_size, rng)
        partial.extend(rows)
        owner.extend([SEP.join(combo)] * len(rows))
        per_slicing_p.append(adjust(np.array([r[5] for r in rows]), correction))

    raw = np.array([r[5] for r in partial])
    p_adj = adjust(raw, correction) if family == "all" else np.concatenate(per_slicing_p)
    return ScanReport(
        metric=name,
        correction=correction,
        family=family,
        alpha=alpha,
        n_perm=n_perm,
        min_size=min_size,
        slicings=combos,
        rows=finish_rows(partial, p_adj, alpha),
        slicing_of=owner,
    )
