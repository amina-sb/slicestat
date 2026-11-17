"""Per-slice reports and the bootstrap intervals behind them."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from slicestat.bootstrap import (
    bca_interval,
    jackknife,
    percentile_interval,
    stratified_replicates,
)
from slicestat.slices import OVERALL, encode_groups, slice_report


def _mean_pred(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred))


def _size(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(y_true.size)


def test_replicates_keep_every_slice_size() -> None:
    rng = np.random.default_rng(0)
    groups = np.array(["a"] * 7 + ["b"] * 30 + ["c"] * 3)
    _, members = encode_groups(groups)
    y = np.zeros(groups.size)
    reps = stratified_replicates(_size, y, y, members, 200, rng)
    assert reps.shape == (200, 4)
    np.testing.assert_array_equal(reps[:, 0], 7)
    np.testing.assert_array_equal(reps[:, 1], 30)
    np.testing.assert_array_equal(reps[:, 2], 3)
    np.testing.assert_array_equal(reps[:, 3], 40)


def test_replicates_only_draw_from_their_own_slice() -> None:
    rng = np.random.default_rng(1)
    groups = np.repeat([0, 1, 2], 50)
    _, members = encode_groups(groups)
    # Encode each row's slice in y_pred; a slice's replicate mean must equal its own label.
    pred = groups.astype(np.float64)
    reps = stratified_replicates(_mean_pred, pred, pred, members, 100, rng)
    np.testing.assert_array_equal(reps[:, :3], np.broadcast_to([0.0, 1.0, 2.0], (100, 3)))


def test_pooled_replicates_vary_slice_sizes_and_keep_total() -> None:
    from slicestat.bootstrap import pooled_replicates

    rng = np.random.default_rng(0)
    groups = np.array(["a"] * 5 + ["b"] * 45)
    _, members = encode_groups(groups)
    y = np.zeros(groups.size)
    reps = pooled_replicates(_size, y, y, members, 400, rng)
    np.testing.assert_array_equal(reps[:, 0] + reps[:, 1], 50)
    np.testing.assert_array_equal(reps[:, 2], 50)
    assert reps[:, 0].std() > 1.0  # the small slice's size moves around
    assert reps[:, 0].mean() == pytest.approx(5, abs=0.3)


def test_pooled_resampling_runs_through_report() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 120).astype(float)
    g = np.repeat(["a", "b", "c"], 40)
    rep = slice_report(y, y, g, n_boot=100, resample="pooled")
    assert [r.estimate for r in rep.rows] == [1.0, 1.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="resampling scheme"):
        slice_report(y, y, g, n_boot=10, resample="block")  # type: ignore[arg-type]


def test_encode_groups_sorts_natively_and_keeps_labels() -> None:
    labels, members = encode_groups([10, 2, 2, 10, 3])
    assert labels == [2, 3, 10]
    assert [m.tolist() for m in members] == [[1, 2], [4], [0, 3]]


def test_encode_groups_rejects_missing_labels() -> None:
    with pytest.raises(ValueError, match="missing"):
        encode_groups(["a", None, "b"])
    with pytest.raises(ValueError, match="missing"):
        encode_groups([1.0, float("nan")])


def test_percentile_interval_ignores_undefined_replicates() -> None:
    r = np.array([1.0, 2.0, 3.0, math.nan, 4.0, 5.0])
    lo, hi = percentile_interval(r, 0.5)
    assert (lo, hi) == (2.0, 4.0)
    assert all(math.isnan(v) for v in percentile_interval(np.array([math.nan]), 0.05))


def test_bca_matches_scipy_on_skewed_mean() -> None:
    """BCa of a mean against ``scipy.stats.bootstrap(method="BCa")``.

    Both implementations are Monte Carlo; with 20,000 resamples their bounds should agree to a
    small fraction of the interval width.
    """
    rng = np.random.default_rng(7)
    x = rng.exponential(1.0, 40)
    ours = slice_report(
        np.zeros_like(x), x, np.zeros(x.size), _mean_pred, n_boot=20_000, method="bca", seed=3
    ).overall
    ref = stats.bootstrap(
        (x,), np.mean, method="BCa", n_resamples=20_000, random_state=np.random.default_rng(4)
    ).confidence_interval
    width = ref.high - ref.low
    assert ours.low == pytest.approx(ref.low, abs=0.04 * width)
    assert ours.high == pytest.approx(ref.high, abs=0.04 * width)
    # BCa should shift right of the percentile interval for a right-skewed mean.
    pct = slice_report(np.zeros_like(x), x, np.zeros(x.size), _mean_pred, n_boot=20_000, seed=3)
    assert ours.high > pct.overall.high


def test_bca_is_nan_when_replicates_sit_on_one_side() -> None:
    reps = np.full(50, 2.0)
    assert all(math.isnan(v) for v in bca_interval(reps, 1.0, np.zeros(5), 0.05))


def test_bca_accepts_discrete_metric_with_ties() -> None:
    # A perfect slice: every replicate equals the estimate. Mid-rank correction keeps z0 = 0.
    y = np.ones(20)
    lo, hi = bca_interval(
        np.ones(100), 1.0, jackknife(lambda a, b: float(np.mean(a == b)), y, y), 0.05
    )
    assert (lo, hi) == (1.0, 1.0)


def test_report_rows_and_estimates() -> None:
    y = np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=float)
    p = np.array([1, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    g = ["a", "a", "a", "a", "b", "b", "b", "b"]
    rep = slice_report(y, p, g, "accuracy", n_boot=200)
    assert [r.group for r in rep.rows] == ["a", "b", OVERALL]
    assert [r.n for r in rep.rows] == [4, 4, 8]
    assert [r.estimate for r in rep.rows] == [0.75, 0.75, 0.75]
    for r in rep.rows:
        assert r.low <= r.estimate <= r.high
    assert rep.overall.group == OVERALL
    assert len(rep.slices()) == 2


def test_undefined_slice_is_reported_as_undefined() -> None:
    y = np.array([0, 0, 0, 1, 1, 0], dtype=float)
    p = np.array([0, 1, 0, 1, 0, 0], dtype=float)
    g = ["neg", "neg", "neg", "mix", "mix", "mix"]
    rep = slice_report(y, p, g, "tpr", n_boot=300)
    neg = next(r for r in rep.rows if r.group == "neg")
    assert math.isnan(neg.estimate) and math.isnan(neg.low) and math.isnan(neg.high)
    assert neg.undefined_share == 1.0
    mix = next(r for r in rep.rows if r.group == "mix")
    assert 0 < mix.undefined_share < 1  # some resamples of "mix" draw no positive
    assert "| neg | 3 | undefined | undefined |" in rep.to_markdown()


def test_same_seed_same_report_and_seed_matters() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300).astype(float)
    p = rng.integers(0, 2, 300).astype(float)
    g = rng.integers(0, 4, 300)
    a = slice_report(y, p, g, n_boot=300, seed=11)
    b = slice_report(y, p, g, n_boot=300, seed=11)
    c = slice_report(y, p, g, n_boot=300, seed=12)
    assert a == b
    assert a.rows[0].low != c.rows[0].low or a.rows[0].high != c.rows[0].high


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"alpha": 0.0}, "alpha"),
        ({"n_boot": 0}, "n_boot"),
        ({"method": "basic"}, "interval method"),
    ],
)
def test_bad_arguments(kwargs: dict[str, object], match: str) -> None:
    y = np.array([0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match=match):
        slice_report(y, y, [0, 0, 1], **{"n_boot": 10, **kwargs})  # type: ignore[arg-type]


def test_shape_errors() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        slice_report([0, 1], [0, 1, 1], [0, 0])
    with pytest.raises(ValueError, match="one label per row"):
        slice_report([0, 1], [0, 1], [0, 0, 1])
    with pytest.raises(ValueError, match="one-dimensional"):
        slice_report([[0, 1]], [[0, 1]], [0, 0])


@settings(max_examples=60, deadline=None)
@given(st.integers(10, 120), st.integers(0, 2**32 - 1))
def test_estimates_invariant_to_row_order(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n).astype(float)
    p = rng.integers(0, 2, n).astype(float)
    g = rng.integers(0, 3, n)
    perm = rng.permutation(n)
    a = slice_report(y, p, g, "f1", n_boot=5)
    b = slice_report(y[perm], p[perm], g[perm], "f1", n_boot=5)
    for ra, rb in zip(a.rows, b.rows, strict=True):
        assert ra.group == rb.group and ra.n == rb.n
        assert (math.isnan(ra.estimate) and math.isnan(rb.estimate)) or ra.estimate == rb.estimate
