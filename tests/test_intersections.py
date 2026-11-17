"""Intersectional slices tested as one family."""

from __future__ import annotations

import numpy as np
import pytest

from slicestat.gaps import gap_test
from slicestat.intersections import cell_labels, scan, slicings
from slicestat.multitest import holm


def _data(seed: int, n: int = 900, bad_cell: bool = False) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    region = rng.choice(["north", "south", "east"], n)
    age = rng.choice(["young", "mid", "old"], n)
    y = rng.integers(0, 2, n).astype(float)
    acc = np.full(n, 0.85)
    if bad_cell:
        acc[(region == "east") & (age == "old")] = 0.45
    p = np.where(rng.random(n) < acc, y, 1 - y)
    return y, p, region, age


def test_slicings_and_labels() -> None:
    assert slicings(["a", "b", "c"], 2) == [
        ("a",),
        ("b",),
        ("c",),
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    ]
    assert cell_labels({"a": [1, 2], "b": ["x", "y"]}, ("a", "b")) == ["a=1 & b=x", "a=2 & b=y"]


def test_order_one_scan_matches_gap_test() -> None:
    y, p, region, _ = _data(0)
    s = scan(y, p, {"region": region}, max_order=1, n_perm=99, min_size=10, seed=3)
    g = gap_test(y, p, [f"region={r}" for r in region], n_perm=99, min_size=10, seed=3)
    assert s.rows == g.rows


@pytest.mark.filterwarnings("ignore::slicestat.gaps.ResolutionWarning")
def test_one_family_across_slicings() -> None:
    y, p, region, age = _data(1)
    s = scan(y, p, {"region": region, "age": age}, n_perm=99, seed=0)
    assert [", ".join(c) for c in s.slicings] == ["region", "age", "region, age"]
    assert len(s.rows) == 3 + 3 + 9
    assert set(s.slicing_of) == {"region", "age", "region & age"}
    raw = np.array([r.p_value for r in s.rows])
    np.testing.assert_allclose([r.p_adjusted for r in s.rows], holm(raw))
    assert s.family_size == 15


@pytest.mark.filterwarnings("ignore::slicestat.gaps.ResolutionWarning")
def test_per_slicing_adjusts_within_each_slicing_only() -> None:
    y, p, region, age = _data(2)
    s = scan(y, p, {"region": region, "age": age}, n_perm=99, family="per_slicing", seed=0)
    by = {}
    for row, owner in zip(s.rows, s.slicing_of, strict=True):
        by.setdefault(owner, []).append(row)
    for rows in by.values():
        np.testing.assert_allclose([r.p_adjusted for r in rows], holm([r.p_value for r in rows]))
    whole = scan(y, p, {"region": region, "age": age}, n_perm=99, seed=0)
    # Same raw p-values; the whole-family adjustment is never smaller.
    assert [r.p_value for r in s.rows] == [r.p_value for r in whole.rows]
    assert all(w.p_adjusted >= q.p_adjusted for w, q in zip(whole.rows, s.rows, strict=True))


def test_planted_cell_is_found() -> None:
    y, p, region, age = _data(3, n=1800, bad_cell=True)
    s = scan(y, p, {"region": region, "age": age}, n_perm=999, seed=0)
    worst = min((r for r in s.rows if r.status == "tested"), key=lambda r: r.gap)
    assert worst.group == "region=east & age=old"
    assert worst.flagged


@pytest.mark.filterwarnings("ignore::slicestat.gaps.ResolutionWarning")
def test_small_cells_are_excluded() -> None:
    y, p, region, age = _data(4, n=150)
    s = scan(y, p, {"region": region, "age": age}, n_perm=20, min_size=20, seed=0)
    small = [r for r in s.rows if r.n < 20]
    assert small and all(r.status == "too_small" for r in small)
    assert s.family_size == sum(r.n >= 20 for r in s.rows)


def test_errors() -> None:
    y, p, region, age = _data(5, n=60)
    with pytest.raises(ValueError, match="max_order"):
        scan(y, p, {"region": region}, max_order=0)
    with pytest.raises(ValueError, match="family"):
        scan(y, p, {"region": region}, family="each")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        scan(y, p, {})
    with pytest.raises(ValueError, match="one value per row"):
        scan(y, p, {"region": region[:10]})
