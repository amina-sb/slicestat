"""The command-line interface, run on a small synthetic CSV written by the test itself."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from slicestat.cli import main


def _write(path: Path, n: int = 300) -> None:
    rng = np.random.default_rng(0)
    g = rng.choice(["north", "south", "east"], n)
    y = rng.integers(0, 2, n)
    acc = np.where(g == "east", 0.55, 0.9)
    p = np.where(rng.random(n) < acc, y, 1 - y)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "pred", "region"])
        w.writerows(zip(y, p, g, strict=True))


def test_cli_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = tmp_path / "preds.csv"
    _write(f)
    args = [str(f), "--y-true", "label", "--y-pred", "pred", "--group", "region"]
    assert main([*args, "--n-boot", "200", "--n-perm", "199"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# accuracy by region")
    assert "Homogeneity test over 3 slices" in out
    assert "| east |" in out and "| overall |" in out
    east = next(line for line in out.splitlines() if line.startswith("| east |") and "yes" in line)
    assert "-" in east.split("|")[5]  # negative gap


def test_cli_errors(tmp_path: Path) -> None:
    f = tmp_path / "preds.csv"
    _write(f)
    with pytest.raises(SystemExit, match="not found"):
        main([str(f), "--y-true", "y", "--y-pred", "pred", "--group", "region"])
    with pytest.raises(SystemExit, match="not numeric"):
        main([str(f), "--y-true", "region", "--y-pred", "pred", "--group", "label"])
