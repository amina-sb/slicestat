"""Metric functions.

Every metric takes ``(y_true, y_pred)`` as one-dimensional arrays and returns a float. A metric
that is undefined on its input -- the true-positive rate of a slice with no positives, the AUC of
a slice containing a single class -- returns ``nan`` instead of raising or returning a default.
Silently reporting 0.0 for an undefined rate is the most common way a subgroup table lies.

Label metrics (``accuracy``, ``error_rate``, ``tpr``, ``fpr``, ``precision``, ``f1``) expect
``y_pred`` to hold 0/1 predictions. Score metrics (``roc_auc``, ``brier``) expect ``y_pred`` to
hold scores or probabilities.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Metric = Callable[[FloatArray, FloatArray], float]

_NAN = float("nan")


def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else _NAN


def accuracy(y_true: FloatArray, y_pred: FloatArray) -> float:
    """Fraction of predictions equal to the label."""
    if y_true.size == 0:
        return _NAN
    return float(np.mean(y_true == y_pred))


def error_rate(y_true: FloatArray, y_pred: FloatArray) -> float:
    """Fraction of predictions different from the label."""
    if y_true.size == 0:
        return _NAN
    return float(np.mean(y_true != y_pred))


def tpr(y_true: FloatArray, y_pred: FloatArray) -> float:
    """True-positive rate (recall, sensitivity). ``nan`` when there are no positives."""
    pos = y_true == 1
    return _ratio(float(np.sum(y_pred[pos] == 1)), float(np.sum(pos)))


def fpr(y_true: FloatArray, y_pred: FloatArray) -> float:
    """False-positive rate. ``nan`` when there are no negatives."""
    neg = y_true == 0
    return _ratio(float(np.sum(y_pred[neg] == 1)), float(np.sum(neg)))


def precision(y_true: FloatArray, y_pred: FloatArray) -> float:
    """Positive predictive value. ``nan`` when nothing is predicted positive."""
    pred_pos = y_pred == 1
    return _ratio(float(np.sum(y_true[pred_pos] == 1)), float(np.sum(pred_pos)))


def f1(y_true: FloatArray, y_pred: FloatArray) -> float:
    """F1 score, ``2TP / (2TP + FP + FN)``. ``nan`` when there are no positives of either kind."""
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    return _ratio(2 * tp, 2 * tp + fp + fn)


def _average_ranks(x: FloatArray) -> FloatArray:
    """1-based ranks with ties given their average rank (``scipy.stats.rankdata`` default)."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    # Boundaries of runs of equal values.
    starts = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1]])
    ends = np.r_[starts[1:], xs.size]
    avg = (starts + ends + 1) / 2.0  # mean of 1-based positions start+1 .. end
    ranks_sorted = np.repeat(avg, ends - starts)
    ranks = np.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    return ranks


def roc_auc(y_true: FloatArray, y_pred: FloatArray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic, ties counted as one half.

    ``nan`` when the input does not contain both classes.
    """
    pos = y_true == 1
    n_pos = int(np.sum(pos))
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return _NAN
    ranks = _average_ranks(y_pred.astype(np.float64))
    u = float(np.sum(ranks[pos])) - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def brier(y_true: FloatArray, y_pred: FloatArray) -> float:
    """Mean squared difference between predicted probability and the 0/1 label."""
    if y_true.size == 0:
        return _NAN
    return float(np.mean((y_pred - y_true) ** 2))


METRICS: dict[str, Metric] = {
    "accuracy": accuracy,
    "error_rate": error_rate,
    "tpr": tpr,
    "fpr": fpr,
    "precision": precision,
    "f1": f1,
    "roc_auc": roc_auc,
    "brier": brier,
}


def get_metric(metric: str | Metric) -> tuple[str, Metric]:
    """Resolve a metric given by name or as a callable to ``(name, function)``."""
    if callable(metric):
        return getattr(metric, "__name__", "metric"), metric
    try:
        return metric, METRICS[metric]
    except KeyError:
        known = ", ".join(sorted(METRICS))
        raise ValueError(f"unknown metric {metric!r}; known metrics: {known}") from None
