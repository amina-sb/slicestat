# Roadmap

## Phase 1 — per-slice estimates with intervals
- [x] Metric functions for labels and scores (accuracy, error rate, TPR, FPR, precision, F1, ROC AUC, Brier)
- [x] Bootstrap resampling stratified by slice, so every replicate keeps each slice's size
- [x] Percentile and BCa intervals
- [x] `slice_report` returning one row per slice plus the overall row

## Phase 2 — gap tests and multiple-comparison control
- [x] Permutation test of each slice against the rest of the data
- [x] Bonferroni, Holm and Benjamini–Hochberg adjustment
- [x] Minimum-slice-size handling that is explicit rather than silent
- [x] Global homogeneity test whose reference is not contaminated by a deviant slice

## Phase 3 — evidence and tooling
- [ ] Simulation evidence: interval coverage, family-wise error under the null, power under a planted gap
- [ ] Command-line interface over a CSV
- [ ] CI running tests, lint, type check and the evidence check
- [ ] Release workflow for PyPI trusted publishing

## Phase 4 — intersectional slices
- [ ] A slice-localisation procedure that does not inherit the contamination of slice-vs-rest gaps, with its error control shown by simulation
- [ ] Slices defined by combinations of attributes, with the multiplicity they imply
- [ ] Documentation site
