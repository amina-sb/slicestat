"""Command-line interface: ``slicestat data.csv --y-true y --y-pred p --group region``.

Prints a markdown report: the homogeneity test, per-slice estimates with intervals, and the
slice-vs-rest gap tests. Reads the CSV with the standard library, so the CLI needs nothing beyond
the library's own NumPy dependency.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import cast

import numpy as np

from slicestat.gaps import gap_test
from slicestat.homogeneity import homogeneity_test
from slicestat.metrics import METRICS, FloatArray
from slicestat.multitest import Correction
from slicestat.slices import IntervalMethod, slice_report


def read_columns(path: Path, names: list[str]) -> dict[str, list[str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [n for n in names if n not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"error: column(s) not found in {path}: {', '.join(missing)}")
        cols: dict[str, list[str]] = {n: [] for n in names}
        for row in reader:
            for n in names:
                cols[n].append(row[n])
    return cols


def _numeric(values: list[str], name: str) -> FloatArray:
    try:
        return np.array([float(v) for v in values], dtype=np.float64)
    except ValueError as exc:
        raise SystemExit(f"error: column {name!r} is not numeric ({exc})") from None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="slicestat", description=__doc__.splitlines()[0])
    ap.add_argument("csv", type=Path, help="input CSV with a header row")
    ap.add_argument("--y-true", required=True, help="column of 0/1 labels")
    ap.add_argument("--y-pred", required=True, help="column of predictions or scores")
    ap.add_argument("--group", required=True, help="column defining the slices")
    ap.add_argument("--metric", default="accuracy", choices=sorted(METRICS))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--interval", default="percentile", choices=["percentile", "bca"])
    ap.add_argument("--correction", default="holm", choices=["none", "bonferroni", "holm", "bh"])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cols = read_columns(args.csv, [args.y_true, args.y_pred, args.group])
    y = _numeric(cols[args.y_true], args.y_true)
    p = _numeric(cols[args.y_pred], args.y_pred)
    g = cols[args.group]

    homog = homogeneity_test(
        y, p, g, args.metric, n_perm=args.n_perm, min_size=args.min_size, seed=args.seed
    )
    report = slice_report(
        y,
        p,
        g,
        args.metric,
        n_boot=args.n_boot,
        alpha=args.alpha,
        method=cast(IntervalMethod, args.interval),
        seed=args.seed,
    )
    gaps = gap_test(
        y,
        p,
        g,
        args.metric,
        n_perm=args.n_perm,
        correction=cast(Correction, args.correction),
        alpha=args.alpha,
        min_size=args.min_size,
        seed=args.seed,
    )
    out = sys.stdout
    out.write(f"# {args.metric} by {args.group}\n\n")
    out.write(
        f"Homogeneity test over {homog.slices_used} slices: dispersion {homog.statistic:.3g}, "
        f"p = {homog.p_value:.4f} ({homog.n_perm} permutations).\n\n"
    )
    out.write("## Per-slice estimates\n\n" + report.to_markdown() + "\n\n")
    out.write("## Slice vs rest\n\n" + gaps.to_markdown() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
