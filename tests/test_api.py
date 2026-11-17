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
