"""Metrics checked against scikit-learn and against their defining properties."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn import metrics as skm

from slicestat import metrics as m


def _labels(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    y = rng.integers(0, 2, n).astype(np.float64)
    p = rng.integers(0, 2, n).astype(np.float64)
    return y, p


@pytest.mark.parametrize("seed", range(20))
def test_label_metrics_match_sklearn(seed: int) -> None:
    rng = np.random.default_rng(seed)
    y, p = _labels(rng, int(rng.integers(5, 200)))
    assert m.accuracy(y, p) == pytest.approx(skm.accuracy_score(y, p))
    assert m.error_rate(y, p) == pytest.approx(1 - skm.accuracy_score(y, p))
    if y.sum() > 0:
        assert m.tpr(y, p) == pytest.approx(skm.recall_score(y, p))
    if p.sum() > 0:
        assert m.precision(y, p) == pytest.approx(skm.precision_score(y, p))
    if y.sum() + p.sum() > 0:
        assert m.f1(y, p) == pytest.approx(skm.f1_score(y, p))
    if (y == 0).sum() > 0:
        tn, fp, _, _ = skm.confusion_matrix(y, p, labels=[0, 1]).ravel()
        assert m.fpr(y, p) == pytest.approx(fp / (fp + tn))


@pytest.mark.parametrize("seed", range(20))
def test_score_metrics_match_sklearn(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 300))
    y = rng.integers(0, 2, n).astype(np.float64)
    y[0], y[1] = 0.0, 1.0
    # Rounded scores force ties, which is where hand-rolled AUCs usually go wrong.
    s = np.round(rng.random(n), 1)
    assert m.roc_auc(y, s) == pytest.approx(skm.roc_auc_score(y, s))
    assert m.brier(y, s) == pytest.approx(skm.brier_score_loss(y, s))


def test_undefined_metrics_are_nan() -> None:
    zeros = np.zeros(5)
    ones = np.ones(5)
    assert math.isnan(m.tpr(zeros, ones))  # no positives
    assert math.isnan(m.fpr(ones, ones))  # no negatives
    assert math.isnan(m.precision(ones, zeros))  # nothing predicted positive
    assert math.isnan(m.f1(zeros, zeros))
    assert math.isnan(m.roc_auc(ones, np.linspace(0, 1, 5)))
    empty = np.array([], dtype=np.float64)
    for f in (m.accuracy, m.error_rate, m.brier):
        assert math.isnan(f(empty, empty))


def test_get_metric_by_name_and_callable() -> None:
    assert m.get_metric("f1") == ("f1", m.f1)
    assert m.get_metric(m.brier)[1] is m.brier
    with pytest.raises(ValueError, match="known metrics"):
        m.get_metric("acuracy")


scores = st.lists(st.floats(0, 1, allow_nan=False), min_size=2, max_size=60)


@settings(max_examples=200, deadline=None)
@given(scores, st.integers(0, 2**32 - 1))
def test_auc_bounded_and_flips_under_negation(s: list[float], seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = np.asarray(s)
    y = rng.integers(0, 2, x.size).astype(np.float64)
    y[0], y[1] = 0.0, 1.0
    a = m.roc_auc(y, x)
    assert 0.0 <= a <= 1.0
    # Reversing the score ordering maps AUC to 1 - AUC exactly, ties included.
    assert m.roc_auc(y, -x) == pytest.approx(1.0 - a, abs=1e-12)


@settings(max_examples=200, deadline=None)
@given(st.integers(1, 80), st.integers(0, 2**32 - 1))
def test_metrics_invariant_to_row_order(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    y, p = _labels(rng, n)
    perm = rng.permutation(n)
    for name, f in m.METRICS.items():
        a, b = f(y, p), f(y[perm], p[perm])
        assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b), name
