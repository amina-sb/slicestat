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

## Which slices differ?

```python
from slicestat import gap_test, homogeneity_test

print(homogeneity_test(y_true, y_pred, groups, metric="accuracy").p_value)
print(gap_test(y_true, y_pred, groups, metric="accuracy", correction="holm").to_markdown())
```

`homogeneity_test` is one permutation test of the null that every slice has the same metric. Its
statistic is the size-weighted dispersion of slice metrics around the overall metric (the ANOVA
between-group sum of squares when the metric is a mean).

`gap_test` tests each slice against the rest of the data by reshuffling slice membership with the
slice size fixed, then adjusts the p-values for the number of slices: `holm` (default) or
`bonferroni` control the chance of flagging any slice when none differs, `bh` controls the
expected share of false flags. Slices smaller than `min_size` (default 10), or whose complement is,
are listed as excluded instead of being tested with no power. Permutation p-values are never
reported as zero.

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

**"Differs from the rest" is not "is anomalous".** The rest of a healthy slice contains whichever
slice is deviant. In a test with five slices of about 240 rows, four of them 90% accurate and one
60% accurate, the healthy slices beat their rest by several points and on average 3.2 of the 4 are
flagged (40 simulated datasets, Holm at 5%); with the deviant slice removed, the family-wise rate
returns to about 5%. Read a positive gap next to a large negative one accordingly, and use
`homogeneity_test` when the question is whether performance varies at all. The test for this is
`tests/test_gaps.py::test_contamination_is_systematic_not_a_seed_artifact`.

See [ROADMAP.md](ROADMAP.md) for what is planned.
