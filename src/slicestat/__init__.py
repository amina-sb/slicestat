"""Subgroup performance metrics with intervals, gap tests and multiplicity control."""

from slicestat.gaps import GapReport, GapRow, gap_test
from slicestat.homogeneity import HomogeneityResult, homogeneity_test
from slicestat.intersections import ScanReport, scan
from slicestat.metrics import METRICS, get_metric
from slicestat.multitest import adjust
from slicestat.slices import OVERALL, SliceReport, SliceRow, slice_report

__all__ = [
    "METRICS",
    "OVERALL",
    "ScanReport",
    "GapReport",
    "GapRow",
    "HomogeneityResult",
    "SliceReport",
    "SliceRow",
    "adjust",
    "gap_test",
    "get_metric",
    "homogeneity_test",
    "scan",
    "slice_report",
]

__version__ = "0.1.0"
