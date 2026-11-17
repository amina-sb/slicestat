"""The README's evidence section and the script that produces it."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evidence  # noqa: E402


@pytest.mark.parametrize(("k", "n"), [(0, 10), (3, 10), (57, 60), (380, 400), (400, 400)])
def test_wilson_matches_statsmodels(k: int, n: int) -> None:
    lo, hi = evidence.wilson(k, n)
    ref_lo, ref_hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    assert lo == pytest.approx(ref_lo, abs=1e-6)
    assert hi == pytest.approx(ref_hi, abs=1e-6)


def test_rate_cell_format() -> None:
    assert evidence.rate_cell([10, 8], 10).startswith("0.900 (")
    assert evidence.rate_cell([10, 8], 10).endswith("· blocks 0.800–1.000")


def test_readme_has_exactly_one_evidence_block() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert text.count(evidence.START) == 1 and text.count(evidence.END) == 1


@pytest.mark.slow
def test_readme_evidence_is_reproduced() -> None:
    """Runs the full simulation (a few minutes) and compares it with the README."""
    assert evidence.main(["--check"]) == 0
