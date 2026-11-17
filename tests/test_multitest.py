"""Adjusted p-values against statsmodels, plus the orderings they must satisfy."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from statsmodels.stats.multitest import multipletests

from slicestat.multitest import adjust, bh, bonferroni, holm

pvals = st.lists(st.floats(0, 1, allow_nan=False), min_size=1, max_size=40)
REFERENCE = {"bonferroni": "bonferroni", "holm": "holm", "bh": "fdr_bh"}


@settings(max_examples=300, deadline=None)
@given(pvals, st.sampled_from(sorted(REFERENCE)))
def test_matches_statsmodels(p: list[float], method: str) -> None:
    ours = adjust(p, method)  # type: ignore[arg-type]
    ref = multipletests(p, method=REFERENCE[method])[1]
    np.testing.assert_allclose(ours, ref, rtol=0, atol=1e-12)


def test_matches_statsmodels_with_ties() -> None:
    p = [0.01, 0.01, 0.04, 0.04, 0.04, 0.2, 0.2, 0.9]
    for method, ref_name in REFERENCE.items():
        np.testing.assert_allclose(
            adjust(p, method),  # type: ignore[arg-type]
            multipletests(p, method=ref_name)[1],
            atol=1e-15,
        )


@settings(max_examples=300, deadline=None)
@given(pvals)
def test_ordering_between_corrections(p: list[float]) -> None:
    raw = np.asarray(p)
    b, h, f = bonferroni(raw), holm(raw), bh(raw)
    # Each correction is at least as conservative as the next one down the list.
    assert np.all(b >= h - 1e-15)
    assert np.all(h >= f - 1e-15)
    assert np.all(f >= raw - 1e-15)
    assert np.all((f >= 0) & (b <= 1))


@settings(max_examples=200, deadline=None)
@given(pvals)
def test_adjustment_preserves_rank_order(p: list[float]) -> None:
    raw = np.asarray(p)
    for f in (bonferroni, holm, bh):
        adj = f(raw)
        i, j = np.argsort(raw, kind="mergesort")[[0, -1]]
        assert adj[i] <= adj[j]


def test_nan_entries_are_excluded_from_family() -> None:
    p = [0.01, math.nan, 0.02]
    out = bonferroni(p)
    assert math.isnan(out[1])
    np.testing.assert_allclose(out[[0, 2]], [0.02, 0.04])
    np.testing.assert_allclose(holm(p)[[0, 2]], multipletests([0.01, 0.02], method="holm")[1])


def test_input_validation() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        holm([0.1, 1.2])
    with pytest.raises(ValueError, match="one-dimensional"):
        bh([[0.1]])
    with pytest.raises(ValueError, match="unknown correction"):
        adjust([0.1], "sidak")  # type: ignore[arg-type]
    np.testing.assert_array_equal(adjust([0.3, 0.1], "none"), [0.3, 0.1])
