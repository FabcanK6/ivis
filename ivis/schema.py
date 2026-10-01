"""Label schema shared by the data generator, the model, and the spec builder.

iVIS frames "natural language -> visual spec" as two jointly learned tasks:

1. Chart-type classification (one label per request, predicted from [CLS]).
2. Slot filling (one BIO tag per word), which marks the spans that become
   measures, axes, legends, filters, time windows, top-N and sort order.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Chart types (intent labels)
# ---------------------------------------------------------------------------
CHART_TYPES: list[str] = [
    "bar",
    "stacked_bar",
    "line",
    "area",
    "pie",
    "donut",
    "table",
    "matrix",
    "card",
    "scatter",
    "map",
]

# Power BI visual type identifiers (as used in report JSON / PBIR visual.json).
POWERBI_VISUALS: dict[str, str] = {
    "bar": "clusteredBarChart",
    "stacked_bar": "stackedBarChart",
    "line": "lineChart",
    "area": "areaChart",
    "pie": "pieChart",
    "donut": "donutChart",
    "table": "tableEx",
    "matrix": "pivotTable",
    "card": "card",
    "scatter": "scatterChart",
    "map": "filledMap",
}

# Power BI field-well role names per visual, used when laying out the spec.
VISUAL_ROLES: dict[str, dict[str, str]] = {
    "bar": {"group_by": "Category", "series": "Series", "measure": "Y"},
    "stacked_bar": {"group_by": "Category", "series": "Series", "measure": "Y"},
    "line": {"group_by": "Category", "series": "Series", "measure": "Y"},
    "area": {"group_by": "Category", "series": "Series", "measure": "Y"},
    "pie": {"group_by": "Category", "series": "Series", "measure": "Y"},
    "donut": {"group_by": "Category", "series": "Series", "measure": "Y"},
    "table": {"group_by": "Values", "series": "Values", "measure": "Values"},
    "matrix": {"group_by": "Rows", "series": "Columns", "measure": "Values"},
    "card": {"group_by": "Values", "series": "Values", "measure": "Values"},
    "scatter": {"group_by": "Details", "series": "Series", "measure": "X/Y"},
    "map": {"group_by": "Location", "series": "Legend", "measure": "Values"},
}

# ---------------------------------------------------------------------------
# Slot types (span labels)
# ---------------------------------------------------------------------------
SLOT_TYPES: list[str] = [
    "METRIC",    # what is measured: "open queries", "SAEs", "enrollment"
    "AGG",       # explicit aggregation: "average", "total", "number of"
    "GROUP_BY",  # primary axis / rows: "by site", "per month"
    "SERIES",    # legend / columns / breakdown: "split by severity"
    "FILTER",    # a filter value: "Germany", "Site 104", "open", "ONC-301"
    "TIME",      # time window: "last 30 days", "this quarter", "in 2025"
    "TOPN",      # "top 10", "5 worst"
    "SORT",      # "highest", "fewest", "descending"
]

BIO_LABELS: list[str] = ["O"] + [f"{p}-{s}" for s in SLOT_TYPES for p in ("B", "I")]

CHART2ID = {c: i for i, c in enumerate(CHART_TYPES)}
ID2CHART = {i: c for c, i in CHART2ID.items()}
LABEL2ID = {lab: i for i, lab in enumerate(BIO_LABELS)}
ID2LABEL = {i: lab for lab, i in LABEL2ID.items()}
