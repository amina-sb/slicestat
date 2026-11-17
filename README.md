# slicestat

Subgroup performance metrics for machine-learning models.

"The model is 94% accurate" and "the model is 94% accurate on four fifths of the data and 61% on
the rest" are different sentences. `slicestat` computes a metric per slice of the data, puts an
interval on each one, tests each slice against the rest, and adjusts those tests for the number of
slices you looked at.

Status: under construction — see [ROADMAP.md](ROADMAP.md).
