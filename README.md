# slicestat

Subgroup performance metrics for machine-learning models.

"The model is 94% accurate" and "the model is 94% accurate on four fifths of the data and 61% on
the rest" are different sentences. `slicestat` computes a metric per slice of the data, puts an
interval on each one, tests each slice against the rest, and adjusts those tests for the number of
slices you looked at.

## Install

```bash
pip install -e ".[dev]"   # from a clone
```

The library itself depends only on NumPy. SciPy, scikit-learn and statsmodels are used by the
test suite as reference implementations.

## Per-slice estimates

```python
from slicestat import slice_report

report = slice_report(y_true, y_pred, groups, metric="tpr", n_boot=2000, method="bca", seed=0)
print(report.to_markdown())
```

`groups` holds one slice label per row. The report has one row per slice and a final `overall`
row, each with the estimate, an interval, and the share of bootstrap replicates in which the
metric was undefined.

Metrics: `accuracy`, `error_rate`, `tpr`, `fpr`, `precision`, `f1` (on 0/1 predictions) and
`roc_auc`, `brier` (on scores). Any callable `f(y_true, y_pred) -> float` also works.

## Decisions that matter

**The bootstrap is stratified by slice.** Each replicate redraws every slice from itself, so a
slice of 30 rows has 30 rows in every replicate. Resampling the whole dataset instead lets a small
slice come back with 20 rows in one replicate and 41 in the next; that variation in size is not
sampling error of the metric, and it inflates the slice's interval.

**Undefined is not zero.** The true-positive rate of a slice with no positives does not exist.
`slicestat` reports it as undefined rather than as 0.0, and reports how often a slice's metric was
undefined across bootstrap replicates, because an interval computed from the replicates where a
small slice happened to contain a positive describes a conditional quantity.

**BCa uses the mid-rank bias correction.** For metrics with a discrete bootstrap distribution —
accuracy on a 25-row slice takes only 26 values — the usual `P(θ* < θ̂)` correction is biased by
the mass sitting exactly at `θ̂`; counting ties as one half fixes that.

See [ROADMAP.md](ROADMAP.md) for what is planned.
