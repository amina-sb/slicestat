"""Subgroup performance metrics with intervals, gap tests and multiplicity control."""

from slicestat.metrics import METRICS, get_metric
from slicestat.slices import OVERALL, SliceReport, SliceRow, slice_report

__all__ = [
    "METRICS",
    "OVERALL",
    "SliceReport",
    "SliceRow",
    "get_metric",
    "slice_report",
]

__version__ = "0.1.0"
