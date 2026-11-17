"""Slice-vs-rest permutation tests."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from slicestat.gaps import ResolutionWarning, gap_test, permutation_gap_pvalue
from slicestat.metrics import accuracy


def _mean_pred(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred))


def test_pvalue_matches_scipy_exact_permutation_test() -> None:
    """Our Monte Carlo p-value against scipy's exact enumeration of the same test.

    With 6 + 6 observations there are C(12, 6) = 924 splits, so scipy enumerates them all. Its
    one-sided test on |mean difference| is exactly our two-sided statistic.
    """
    rng = np.random.default_rng(5)
    a = rng.normal(0.0, 1.0, 6)
    b = rng.normal(1.0, 1.0, 6)
    x = np.concatenate([a, b])
    mask = np.r_[np.ones(6, bool), np.zeros(6, bool)]
    _, ours = permutation_gap_pvalue(_mean_pred, x, x, mask, 40_000, np.random.default_rng(0))
    ref = stats.permutation_test(
        (a, b),
        lambda u, v: abs(np.mean(u) - np.mean(v)),
        permutation_type="independent",
        alternative="greater",
        n_resamples=10_000,
    ).pvalue
    assert ours == pytest.approx(ref, abs=0.005)


def test_pvalue_is_never_zero() -> None:
    y = np.r_[np.ones(30), np.zeros(30)]
    p = np.r_[np.ones(30), np.ones(30)]  # perfect on the first slice, useless on the second
    mask = np.r_[np.ones(30, bool), np.zeros(30, bool)]
    gap, pv = permutation_gap_pvalue(accuracy, y, p, mask, 99, np.random.default_rng(0))
    assert gap == 1.0
    assert pv == pytest.approx(1 / 100)


@pytest.mark.filterwarnings("ignore::slicestat.gaps.ResolutionWarning")
def test_identical_slices_give_p_of_one() -> None:
    # Every row is identical, so every permuted gap equals the observed gap of zero. The tie
    # tolerance must count all of them.
    y = np.tile([1.0, 0.0, 1.0], 20)
    g = np.repeat(["a", "b", "c"], 20)
    rep = gap_test(y, y, g, n_perm=50)
    assert [r.p_value for r in rep.rows] == [1.0, 1.0, 1.0]


def test_null_pvalues_are_calibrated() -> None:
    """Under the global null a 5% test should reject about 5% of the time, not more."""
    rejections = 0
    trials = 300
    rng = np.random.default_rng(123)
    for t in range(trials):
        n = 200
        y = rng.integers(0, 2, n).astype(float)
        p = np.where(rng.random(n) < 0.8, y, 1 - y)  # 80% accurate everywhere
        g = rng.integers(0, 4, n)
        _, pv = permutation_gap_pvalue(accuracy, y, p, g == 0, 199, np.random.default_rng(t))
        rejections += pv <= 0.05
    # Binomial(300, 0.05) has mean 15 and sd 3.8; 27 is about three sd above the mean.
    assert rejections <= 27
    assert rejections >= 4


def _planted(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 1200
    g = rng.choice(["a", "b", "c", "d", "e"], n)
    y = rng.integers(0, 2, n).astype(float)
    acc = np.where(g == "e", 0.6, 0.9)
    p = np.where(rng.random(n) < acc, y, 1 - y)
    return y, p, g


def test_one_bad_slice_pushes_healthy_slices_above_their_rest() -> None:
    """A deviant slice contaminates the reference of every other slice.

    The naive expectation is that planting a gap in slice "e" flags "e" alone. It does not: the
    "rest" of each healthy slice contains "e", so each healthy slice beats its rest by several
    points, and with a few hundred rows that is significant. Slices a-d are all 90% accurate in
    truth. The test answers its own question correctly -- such a slice really is better than
    the rest of the data -- but "differs from the rest" is not "is anomalous".
    """
    y, p, g = _planted(0)
    by = {r.group: r for r in gap_test(y, p, g, n_perm=999, seed=1).rows}
    assert by["e"].flagged and by["e"].gap < -0.2
    assert all(by[k].gap > 0 for k in "abcd")


def test_contamination_is_systematic_not_a_seed_artifact() -> None:
    """Across 40 datasets: with "e" present most healthy slices are flagged; without it the
    family-wise error rate is back near the nominal 5%."""
    trials = 40
    healthy_flags = 0
    any_flag_without_e = 0
    for s in range(trials):
        y, p, g = _planted(s)
        rep = gap_test(y, p, g, n_perm=199, seed=s)
        healthy_flags += sum(r.flagged for r in rep.rows if r.group != "e")
        keep = g != "e"
        clean = gap_test(y[keep], p[keep], g[keep], n_perm=199, seed=s)
        any_flag_without_e += bool(clean.flagged())
    assert healthy_flags / trials > 2.5  # of 4 healthy slices
    # Binomial(40, 0.05) has mean 2 and sd 1.4.
    assert any_flag_without_e <= 6


@pytest.mark.filterwarnings("ignore::slicestat.gaps.ResolutionWarning")
def test_small_slices_are_excluded_explicitly() -> None:
    rng = np.random.default_rng(2)
    g = np.r_[np.repeat("big", 100), np.repeat("tiny", 4), np.repeat("mid", 50)]
    y = rng.integers(0, 2, g.size).astype(float)
    rep = gap_test(y, y, g, n_perm=20, min_size=10)
    tiny = next(r for r in rep.rows if r.group == "tiny")
    assert tiny.status == "too_small"
    assert math.isnan(tiny.p_value) and math.isnan(tiny.p_adjusted)
    assert not tiny.flagged
    assert tiny.estimate == 1.0  # still reported, just not tested
    # Excluded slices do not count toward the family: Bonferroni over two tests, not three.
    bon = gap_test(y, y, g, n_perm=20, min_size=10, correction="bonferroni")
    tested = [r for r in bon.rows if r.status == "tested"]
    assert len(tested) == 2
    for r in tested:
        assert r.p_adjusted == pytest.approx(min(1.0, 2 * r.p_value))
    assert "excluded: too small" in rep.to_markdown()


def test_undefined_gap_is_not_tested() -> None:
    y = np.r_[np.zeros(20), np.ones(20)]
    p = y.copy()
    g = np.r_[np.repeat("neg", 20), np.repeat("pos", 20)]
    rep = gap_test(y, p, g, "tpr", n_perm=20)
    neg = next(r for r in rep.rows if r.group == "neg")
    assert neg.status == "undefined" and math.isnan(neg.p_value)


def test_reproducible_with_seed() -> None:
    rng = np.random.default_rng(9)
    y = rng.integers(0, 2, 150).astype(float)
    p = rng.integers(0, 2, 150).astype(float)
    g = rng.integers(0, 3, 150)
    assert gap_test(y, p, g, n_perm=100, seed=4) == gap_test(y, p, g, n_perm=100, seed=4)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"alpha": 1.0}, "alpha"), ({"n_perm": 0}, "n_perm"), ({"min_size": 0}, "min_size")],
)
def test_bad_arguments(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        gap_test([0, 1], [0, 1], [0, 1], **kwargs)  # type: ignore[arg-type]


def test_warns_when_no_slice_could_ever_be_flagged() -> None:
    """36 tests at alpha 0.05 need p <= 0.0014 under Bonferroni, but 199 permutations cannot
    produce a p-value below 0.005, so nothing can be flagged whatever the data say."""
    from slicestat.gaps import ResolutionWarning, warn_if_unflaggable

    with pytest.warns(ResolutionWarning, match="n_perm >= 719"):
        warn_if_unflaggable(36, 199, "holm", 0.05)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_unflaggable(36, 999, "holm", 0.05)
        warn_if_unflaggable(36, 199, "none", 0.05)
        warn_if_unflaggable(8, 199, "bonferroni", 0.05)


def test_gap_test_raises_the_resolution_warning() -> None:
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(12), 30)
    y = rng.integers(0, 2, g.size).astype(float)
    with pytest.warns(ResolutionWarning):
        gap_test(y, y, g, n_perm=99)
