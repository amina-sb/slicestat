"""Produce every number the README quotes.

    python scripts/evidence.py            # print the evidence tables as markdown
    python scripts/evidence.py --check    # exit 1 if README.md differs from the printed tables

All data are simulated from generators defined below, so the true value each method is meant to
recover is known. Every rate is reported with a 95% Wilson interval over the simulated datasets,
and with the range across five independent seed blocks, so a reader can see both the Monte Carlo
error and whether a result depends on the seed.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from slicestat import gap_test, slice_report
from slicestat.metrics import FloatArray
from slicestat.multitest import Correction, adjust
from slicestat.slices import IntervalMethod, Resampling

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
START = "<!-- evidence:start -->"
END = "<!-- evidence:end -->"
BLOCKS = 5


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return math.nan, math.nan
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def rate_cell(hits: list[int], per_block: int) -> str:
    """'0.950 (0.927–0.966) · blocks 0.925–0.975' for per-block hit counts."""
    k, n = sum(hits), per_block * len(hits)
    lo, hi = wilson(k, n)
    rates = [h / per_block for h in hits]
    return f"{k / n:.3f} ({lo:.3f}–{hi:.3f}) · blocks {min(rates):.3f}–{max(rates):.3f}"


def labelled(n: int, accuracy: float, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """Balanced 0/1 labels and predictions that are correct with probability ``accuracy``."""
    y = rng.integers(0, 2, n).astype(np.float64)
    p = np.where(rng.random(n) < accuracy, y, 1 - y)
    return y, p


# --- 1. interval coverage -------------------------------------------------------------------

COVERAGE_SIZES = (15, 40, 150)
COVERAGE_TRIALS_PER_BLOCK = 80
COVERAGE_TRUE = 0.8
COVERAGE_SCHEMES: tuple[tuple[IntervalMethod, Resampling, str], ...] = (
    ("percentile", "slice", "percentile, stratified by slice"),
    ("bca", "slice", "BCa, stratified by slice"),
    ("percentile", "pooled", "percentile, pooled bootstrap (baseline)"),
)


def coverage_table() -> str:
    lines = [
        f"| slice size | interval | coverage of true accuracy {COVERAGE_TRUE} | mean width |",
        "|---:|---|---|---:|",
    ]
    for size in COVERAGE_SIZES:
        for method, resample, label in COVERAGE_SCHEMES:
            hits: list[int] = []
            widths: list[float] = []
            for block in range(BLOCKS):
                h = 0
                for t in range(COVERAGE_TRIALS_PER_BLOCK):
                    seed = 1_000_000 * block + 1_000 * size + t
                    rng = np.random.default_rng(seed)
                    g = np.repeat([0, 1, 2], [size, 200, 200])
                    y, p = labelled(g.size, COVERAGE_TRUE, rng)
                    row = slice_report(
                        y, p, g, n_boot=400, method=method, resample=resample, seed=seed
                    ).rows[0]
                    h += row.low <= COVERAGE_TRUE <= row.high
                    widths.append(row.high - row.low)
                hits.append(h)
            cell = rate_cell(hits, COVERAGE_TRIALS_PER_BLOCK)
            lines.append(f"| {size} | {label} | {cell} | {np.mean(widths):.4f} |")
    return "\n".join(lines)


# --- 2. family-wise error under the global null ---------------------------------------------

FWER_SLICES = 8
FWER_SIZE = 100
FWER_TRIALS_PER_BLOCK = 60


def fwer_table() -> str:
    corrections: tuple[Correction, ...] = ("none", "bonferroni", "holm", "bh")
    hits = {c: [0] * BLOCKS for c in corrections}
    for block in range(BLOCKS):
        for t in range(FWER_TRIALS_PER_BLOCK):
            seed = 7_000_000 + 10_000 * block + t
            rng = np.random.default_rng(seed)
            g = np.repeat(np.arange(FWER_SLICES), FWER_SIZE)
            y, p = labelled(g.size, 0.8, rng)
            raw = np.array(
                [
                    r.p_value
                    for r in gap_test(y, p, g, n_perm=199, correction="none", seed=seed).rows
                ]
            )
            for c in corrections:
                hits[c][block] += bool(np.any(adjust(raw, c) <= 0.05))
    names = {
        "none": "unadjusted (baseline)",
        "bonferroni": "Bonferroni",
        "holm": "Holm",
        "bh": "Benjamini–Hochberg",
    }
    lines = [
        f"| correction | P(any slice flagged), {FWER_SLICES} slices with no real difference |",
        "|---|---|",
    ]
    for c in corrections:
        lines.append(f"| {names[c]} | {rate_cell(hits[c], FWER_TRIALS_PER_BLOCK)} |")
    return "\n".join(lines)


# --- 3. contamination of slice-vs-rest gaps -------------------------------------------------

CONTAM_TRIALS_PER_BLOCK = 12


def _planted(
    rng: np.random.Generator, with_bad: bool
) -> tuple[FloatArray, FloatArray, NDArray[np.str_]]:
    n = 1200
    g = rng.choice(np.array(list("abcde")), n)
    acc = np.where(g == "e", 0.6, 0.9)
    y = rng.integers(0, 2, n).astype(np.float64)
    p = np.where(rng.random(n) < acc, y, 1 - y)
    if not with_bad:
        keep = g != "e"
        return y[keep], p[keep], g[keep]
    return y, p, g


def contamination_table() -> str:
    flagged_healthy: list[int] = [0] * BLOCKS
    any_with = [0] * BLOCKS
    any_without = [0] * BLOCKS
    bad_flagged = [0] * BLOCKS
    for block in range(BLOCKS):
        for t in range(CONTAM_TRIALS_PER_BLOCK):
            seed = 9_000_000 + 10_000 * block + t
            rows = gap_test(
                *_planted(np.random.default_rng(seed), True), n_perm=199, seed=seed
            ).rows
            healthy = sum(r.flagged for r in rows if r.group != "e")
            flagged_healthy[block] += healthy
            any_with[block] += healthy > 0
            bad_flagged[block] += any(r.flagged for r in rows if r.group == "e")
            clean = gap_test(*_planted(np.random.default_rng(seed), False), n_perm=199, seed=seed)
            any_without[block] += any(r.flagged for r in clean.rows)
    per = CONTAM_TRIALS_PER_BLOCK
    means = [f / per for f in flagged_healthy]
    mean_all = sum(flagged_healthy) / (per * BLOCKS)
    lines = [
        "| four healthy slices at 90% accuracy, plus | result (Holm, 5%) |",
        "|---|---|",
        f"| one slice at 60% | deviant slice flagged: {rate_cell(bad_flagged, per)} |",
        f"| one slice at 60% | any healthy slice flagged: {rate_cell(any_with, per)} |",
        f"| one slice at 60% | healthy slices flagged per dataset, of 4: {mean_all:.2f}"
        f" · blocks {min(means):.2f}–{max(means):.2f} |",
        f"| nothing (deviant slice removed) | any slice flagged: {rate_cell(any_without, per)} |",
    ]
    return "\n".join(lines)


SECTIONS: list[tuple[str, Callable[[], str]]] = [
    ("Interval coverage", coverage_table),
    ("Family-wise error with no real difference", fwer_table),
    ("One deviant slice and slice-vs-rest gaps", contamination_table),
]


def render() -> str:
    parts = [START]
    for title, fn in SECTIONS:
        parts.append(f"#### {title}\n\n{fn()}")
    parts.append(END)
    return "\n\n".join(parts)


def readme_block(text: str) -> str:
    start, end = text.index(START), text.index(END) + len(END)
    return text[start:end]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail if README.md is out of date")
    ap.add_argument("--write", action="store_true", help="rewrite the block in README.md")
    args = ap.parse_args(argv)
    out = render()
    readme = README.read_text(encoding="utf-8")
    if args.check:
        current = readme_block(readme)
        if current != out:
            sys.stdout.write("README evidence block differs from scripts/evidence.py output:\n\n")
            sys.stdout.write(out + "\n")
            return 1
        sys.stdout.write("README evidence block is up to date.\n")
        return 0
    if args.write:
        README.write_text(readme.replace(readme_block(readme), out), encoding="utf-8")
    sys.stdout.write(out + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
