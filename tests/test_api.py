"""The documented public surface."""

from __future__ import annotations

import slicestat


def test_public_names_are_exported() -> None:
    for name in slicestat.__all__:
        assert hasattr(slicestat, name), name


def test_readme_example_runs() -> None:
    y = [1, 1, 0, 1, 0, 1, 1, 0]
    p = [1, 0, 0, 1, 1, 1, 1, 0]
    g = ["a", "a", "a", "a", "b", "b", "b", "b"]
    report = slicestat.slice_report(y, p, g, metric="tpr", n_boot=200, method="bca", seed=0)
    table = report.to_markdown()
    assert table.splitlines()[0] == "| slice | n | tpr | 95% CI (bca) |"
    assert table.splitlines()[-1].startswith("| overall | 8 | 0.800 |")


def test_readme_gap_example_runs() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    g = rng.choice(["a", "b", "c"], 300)
    y = rng.integers(0, 2, 300)
    p = np.where(rng.random(300) < 0.85, y, 1 - y)
    assert 0 < slicestat.homogeneity_test(y, p, g, n_perm=50).p_value <= 1
    table = slicestat.gap_test(y, p, g, metric="accuracy", correction="holm", n_perm=50)
    assert table.to_markdown().startswith("| slice | n | accuracy | rest | gap | p | p (holm) |")
