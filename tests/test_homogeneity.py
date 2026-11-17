"""The global test of whether slices differ at all."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from slicestat.homogeneity import dispersion, homogeneity_test
from slicestat.slices import encode_groups


def _mean_pred(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred))


def test_dispersion_is_anova_between_group_sum_of_squares() -> None:
    rng = np.random.default_rng(0)
    g = rng.integers(0, 4, 90)
    x = rng.normal(g * 0.3, 1.0)
    _, members = encode_groups(g)
    d = dispersion(_mean_pred, x, x, members)
    ssb = 0.0
    for k in range(4):
        xk = x[g == k]
        ssb += xk.size * (xk.mean() - x.mean()) ** 2
    assert d == pytest.approx(ssb / x.size)


def test_pvalue_agrees_with_one_way_anova_for_a_mean() -> None:
    """For normal data the permutation test of D and the F test should broadly agree."""
    rng = np.random.default_rng(3)
    g = np.repeat([0, 1, 2], 40)
    x = rng.normal(np.array([0.0, 0.0, 0.45])[g], 1.0)
    ours = homogeneity_test(x, x, g, _mean_pred, n_perm=5000, seed=1).p_value
    ref = stats.f_oneway(*(x[g == k] for k in range(3))).pvalue
    assert ours == pytest.approx(ref, abs=0.01)


def _planted(seed: int, bad: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 1000
    g = rng.choice(list("abcde"), n)
    y = rng.integers(0, 2, n).astype(float)
    acc = np.where(g == "e", bad, 0.9)
    p = np.where(rng.random(n) < acc, y, 1 - y)
    return y, p, g


def test_level_and_power() -> None:
    trials = 60
    null_rejections = sum(
        homogeneity_test(*_planted(s, 0.9), n_perm=199, seed=s).p_value <= 0.05
        for s in range(trials)
    )
    alt_rejections = sum(
        homogeneity_test(*_planted(s, 0.8), n_perm=199, seed=s).p_value <= 0.05
        for s in range(trials)
    )
    # Binomial(60, 0.05): mean 3, sd 1.7.
    assert null_rejections <= 8
    assert alt_rejections >= 45


def test_small_slices_dropped_and_errors() -> None:
    g = np.r_[np.repeat("a", 30), np.repeat("b", 30), np.repeat("c", 2)]
    y = np.ones(g.size)
    res = homogeneity_test(y, y, g, n_perm=20, min_size=5)
    assert (res.n_slices, res.slices_used) == (3, 2)
    assert res.p_value == 1.0  # identical slices: every permutation ties
    with pytest.raises(ValueError, match="two slices"):
        homogeneity_test(y, y, g, n_perm=20, min_size=31)
    with pytest.raises(ValueError, match="n_perm"):
        homogeneity_test(y, y, g, n_perm=0)
