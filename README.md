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

## Command line

```bash
slicestat predictions.csv --y-true label --y-pred pred --group region --metric tpr
```

prints the homogeneity test, the per-slice table and the slice-vs-rest table as markdown.

## Decisions that matter

**The bootstrap is stratified by slice.** Each replicate redraws every slice from itself, so a
slice of 30 rows has 30 rows in every replicate. Resampling the whole dataset instead (available
as `resample="pooled"`, for comparison) lets a small slice come back with 20 rows in one replicate
and 41 in the next. That variation in size is not sampling error of the metric. In the coverage
table below it makes the pooled intervals wider for the 15-row slice while covering no more often;
at 150 rows the two schemes are indistinguishable.

**Undefined is not zero.** The true-positive rate of a slice with no positives does not exist.
`slicestat` reports it as undefined rather than as 0.0, and reports how often a slice's metric was
undefined across bootstrap replicates, because an interval computed from the replicates where a
small slice happened to contain a positive describes a conditional quantity.

**BCa uses the mid-rank bias correction.** For metrics with a discrete bootstrap distribution —
accuracy on a 25-row slice takes only 26 values — the usual `P(θ* < θ̂)` correction is biased by
the mass sitting exactly at `θ̂`; counting ties as one half fixes that.

**"Differs from the rest" is not "is anomalous".** The rest of a healthy slice contains whichever
slice is deviant, so one bad slice makes the healthy slices beat their rest, and with enough rows
those gaps are significant. The last table below shows the size of the effect: 1,200 rows split at
random across five slices, four of them 90% accurate and one 60% accurate. Read a positive gap next
to a large negative one accordingly, and use `homogeneity_test` when the question is whether
performance varies at all.

## Evidence

Every number in this section is printed by [`scripts/evidence.py`](scripts/evidence.py) from
simulated data whose true values are known, and CI fails if this section and the script's output
differ. Rates carry a 95% Wilson interval over the simulated datasets and the range across five
independent seed blocks. Intervals use 400 bootstrap resamples and tests 199 permutations, to keep
the script to a few minutes; the defaults in the library are 2,000 of each.

The coverage simulation puts the slice of interest next to two 200-row slices; the true accuracy
is 0.8 everywhere, and 400 datasets are drawn per row. The family-wise error simulation uses eight
100-row slices with identical true accuracy, 300 datasets.

<!-- evidence:start -->

#### Interval coverage

| slice size | interval | coverage of true accuracy 0.8 | mean width |
|---:|---|---|---:|
| 15 | percentile, stratified by slice | 0.948 (0.921–0.965) · blocks 0.925–0.988 | 0.3588 |
| 15 | BCa, stratified by slice | 0.940 (0.912–0.959) · blocks 0.912–0.988 | 0.3933 |
| 15 | percentile, pooled bootstrap (baseline) | 0.930 (0.901–0.951) · blocks 0.900–0.975 | 0.3797 |
| 40 | percentile, stratified by slice | 0.955 (0.930–0.971) · blocks 0.938–1.000 | 0.2376 |
| 40 | BCa, stratified by slice | 0.975 (0.955–0.986) · blocks 0.963–1.000 | 0.2437 |
| 40 | percentile, pooled bootstrap (baseline) | 0.932 (0.904–0.953) · blocks 0.875–0.988 | 0.2406 |
| 150 | percentile, stratified by slice | 0.932 (0.904–0.953) · blocks 0.875–0.963 | 0.1261 |
| 150 | BCa, stratified by slice | 0.932 (0.904–0.953) · blocks 0.900–0.963 | 0.1268 |
| 150 | percentile, pooled bootstrap (baseline) | 0.922 (0.892–0.945) · blocks 0.875–0.963 | 0.1264 |

#### Family-wise error with no real difference

| correction | P(any slice flagged), 8 slices with no real difference |
|---|---|
| unadjusted (baseline) | 0.290 (0.242–0.344) · blocks 0.267–0.300 |
| Bonferroni | 0.037 (0.021–0.064) · blocks 0.017–0.067 |
| Holm | 0.037 (0.021–0.064) · blocks 0.017–0.067 |
| Benjamini–Hochberg | 0.037 (0.021–0.064) · blocks 0.017–0.067 |

#### One deviant slice and slice-vs-rest gaps

| four healthy slices at 90% accuracy, plus | result (Holm, 5%) |
|---|---|
| one slice at 60% | deviant slice flagged: 1.000 (0.940–1.000) · blocks 1.000–1.000 |
| one slice at 60% | any healthy slice flagged: 0.967 (0.886–0.991) · blocks 0.917–1.000 |
| one slice at 60% | healthy slices flagged per dataset, of 4: 2.83 · blocks 2.42–3.33 |
| nothing (deviant slice removed) | any slice flagged: 0.067 (0.026–0.159) · blocks 0.000–0.167 |

<!-- evidence:end -->

See [ROADMAP.md](ROADMAP.md) for what is planned.
