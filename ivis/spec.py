"""Turn (chart type, labelled spans) into a Power BI visual spec.

This layer is deterministic: the model decides *what* was asked for, and this
module maps it onto the semantic model in :mod:`ivis.catalog`, fills sensible
defaults, and records anything it could not resolve under ``warnings``.
"""

from __future__ import annotations

import re
from collections import OrderedDict

from ivis import catalog
from ivis.schema import POWERBI_VISUALS, VISUAL_ROLES
from ivis.text import Span

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20, "fifty": 50, "hundred": 100,
}
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
UNIT_MAP = {"day": "Days", "week": "Weeks", "month": "Months", "quarter": "Quarters", "year": "Years"}


def _to_int(tok: str) -> int | None:
    tok = tok.lower()
    if tok.isdigit():
        return int(tok)
    return NUMBER_WORDS.get(tok)


# ---------------------------------------------------------------------------
# Slot normalizers
# ---------------------------------------------------------------------------
def parse_time(phrase: str) -> dict:
    """Normalize a time phrase into a Power BI-style relative/absolute date filter."""
    p = phrase.lower().strip()
    out: dict = {"text": phrase, "field": catalog.DATE_FIELD}

    m = re.search(r"(?:last|past|previous|trailing)\s+(\w+)\s+(day|week|month|quarter|year)s?", p)
    if m and _to_int(m.group(1)):
        n, unit = _to_int(m.group(1)), m.group(2)
        out.update(filterType="RelativeDate", operator="InLast", timeUnitsCount=n,
                   timeUnit=UNIT_MAP[unit], includeToday=True)
        return out
    m = re.search(r"(this|current|last|previous|past)\s+(day|week|month|quarter|year)", p)
    if m:
        this = m.group(1) in ("this", "current")
        out.update(filterType="RelativeDate", operator="InThis" if this else "InLastCalendar",
                   timeUnitsCount=1, timeUnit=UNIT_MAP[m.group(2)])
        return out
    if p in ("ytd", "year to date") or "year to date" in p:
        out.update(filterType="RelativeDate", operator="InThis", timeUnitsCount=1, timeUnit="Years",
                   toDate=True)
        return out
    if "mtd" == p or "month to date" in p:
        out.update(filterType="RelativeDate", operator="InThis", timeUnitsCount=1, timeUnit="Months",
                   toDate=True)
        return out
    m = re.search(r"q([1-4])\s*(\d{4})?", p)
    if m:
        q = int(m.group(1))
        year = m.group(2)
        out.update(filterType="Advanced", operator="Quarter", quarter=q)
        if year:
            out["year"] = int(year)
        return out
    m = re.search(r"since\s+(\w+)(?:\s+(\d{4}))?", p)
    if m:
        word = m.group(1)
        if word in MONTHS:
            out.update(filterType="Advanced", operator="GreaterThanOrEqual", month=MONTHS.index(word) + 1)
            if m.group(2):
                out["year"] = int(m.group(2))
        else:
            out.update(filterType="Advanced", operator="GreaterThanOrEqual", anchor=word.upper())
        return out
    m = re.search(r"\b(19|20)\d{2}\b", p)
    if m:
        out.update(filterType="Advanced", operator="Year", year=int(m.group(0)))
        return out
    out.update(filterType="Unresolved")
    return out


def parse_topn(phrase: str) -> dict:
    p = phrase.lower()
    n = next((_to_int(t) for t in p.split() if _to_int(t)), 10)
    direction = "asc" if any(w in p for w in ("bottom", "lowest", "least", "fewest")) else "desc"
    return {"n": n, "direction": direction, "text": phrase}


def parse_sort(phrase: str) -> str:
    p = phrase.lower()
    for direction, ws in catalog.SORT_WORDS.items():
        if any(w in p for w in ws):
            return direction
    return "desc"


def parse_agg(phrase: str) -> str | None:
    p = phrase.lower().strip()
    best = None
    for agg, ws in catalog.AGGREGATIONS.items():
        for w in ws:
            if w in p and (best is None or len(w) > best[1]):
                best = (agg, len(w))
    return best[0] if best else None


# ---------------------------------------------------------------------------
# Spec builder
# ---------------------------------------------------------------------------
def build_spec(
    text: str,
    tokens: list[str],
    chart_type: str,
    spans: list[Span],
    chart_confidence: float | None = None,
    chart_alternatives: list[tuple[str, float]] | None = None,
) -> dict:
    warnings: list[str] = []
    measures: list[dict] = []
    group_by: list[dict] = []
    series: list[dict] = []
    filters: "OrderedDict[str, dict]" = OrderedDict()
    time_filter = None
    top_n = None
    sort_dir = None
    pending_agg = None

    for sp in sorted(spans, key=lambda s: s.start):
        if sp.label == "AGG":
            pending_agg = parse_agg(sp.text)
        elif sp.label == "METRIC":
            m = catalog.resolve_measure(sp.text)
            if m is None:
                warnings.append(f"Unknown measure '{sp.text}' - add it to the catalog or map it manually.")
                measures.append({"name": sp.text, "field": None, "aggregation": pending_agg or "count",
                                 "source_span": sp.text})
            else:
                measures.append({"name": m.display, "field": m.field, "aggregation": pending_agg or m.default_agg,
                                 "unit": m.unit, "source_span": sp.text})
            pending_agg = None
        elif sp.label in ("GROUP_BY", "SERIES"):
            d = catalog.resolve_dimension(sp.text)
            target = group_by if sp.label == "GROUP_BY" else series
            if d is None:
                warnings.append(f"Unknown dimension '{sp.text}'.")
                target.append({"name": sp.text, "field": None, "source_span": sp.text})
            elif not any(x["field"] == d.field for x in group_by + series):
                target.append({"name": d.display, "field": d.field, "source_span": sp.text})
        elif sp.label == "FILTER":
            context = tokens[max(0, sp.start - 3):sp.start]
            d, value = catalog.resolve_filter_value(sp.text, context)
            key = d.field if d else f"?{sp.text}"
            if d is None:
                warnings.append(f"Could not tell which field the filter value '{sp.text}' belongs to.")
            f = filters.setdefault(key, {"field": d.field if d else None, "name": d.display if d else None,
                                         "operator": "In", "values": []})
            if value not in f["values"]:
                f["values"].append(value)
        elif sp.label == "TIME":
            time_filter = parse_time(sp.text)
            if time_filter["filterType"] == "Unresolved":
                warnings.append(f"Time window '{sp.text}' needs a manual date range.")
        elif sp.label == "TOPN":
            top_n = parse_topn(sp.text)
        elif sp.label == "SORT":
            sort_dir = parse_sort(sp.text)

    if pending_agg and measures:
        measures[0]["aggregation"] = pending_agg

    # --- chart-specific defaults -------------------------------------------
    if not measures:
        warnings.append("No measure detected; defaulting to a count of records.")
        measures.append({"name": "Record Count", "field": None, "aggregation": "count", "source_span": None})
    if chart_type in ("line", "area") and not any(
        g["field"] and g["field"].startswith("Date[") for g in group_by
    ):
        if group_by:
            series = series or group_by
        group_by = [{"name": "Month", "field": "Date[Month]", "source_span": None, "defaulted": True}]
    if chart_type == "map":
        geo = [g for g in group_by if g["field"] in ("Site[Country]", "Site[Region]")]
        if not geo:
            group_by = [{"name": "Country", "field": "Site[Country]", "source_span": None, "defaulted": True}]
    if chart_type in ("pie", "donut", "bar", "stacked_bar", "matrix") and not group_by:
        warnings.append(f"A {chart_type} chart needs a category; none was found in the request.")
    if chart_type == "scatter" and len(measures) < 2:
        warnings.append("A scatter plot needs two measures (X and Y); only one was found.")
    if chart_type in ("stacked_bar", "matrix") and not series:
        warnings.append(f"A {chart_type} usually needs a second dimension (legend/columns).")
    if chart_type == "card":
        if group_by:
            warnings.append("Card visuals show a single value; the grouping was dropped.")
        group_by, series = [], []

    if top_n and not sort_dir:
        sort_dir = top_n["direction"]
    if top_n:
        top_n["by"] = measures[0]["name"]
        top_n["direction"] = sort_dir or top_n["direction"]
    sort = None
    if sort_dir:
        sort = {"by": measures[0]["name"], "direction": sort_dir}
    elif chart_type in ("bar", "table", "pie", "donut") and group_by:
        sort = {"by": measures[0]["name"], "direction": "desc", "defaulted": True}

    roles = VISUAL_ROLES[chart_type]
    field_wells: dict[str, list[str]] = {}

    def _add(role: str, f: str | None) -> None:
        if f:
            field_wells.setdefault(role, []).append(f)

    for g in group_by:
        _add(roles["group_by"], g["field"])
    for s in series:
        _add(roles["series"], s["field"])
    if chart_type == "scatter":
        if measures:
            _add("X", measures[0]["field"])
        if len(measures) > 1:
            _add("Y", measures[1]["field"])
    else:
        for m in measures:
            _add(roles["measure"], m["field"])

    title = _title(measures, group_by, series, chart_type)
    return {
        "source_text": text,
        "title": title,
        "chart_type": chart_type,
        "powerbi_visual": POWERBI_VISUALS[chart_type],
        "measures": measures,
        "group_by": group_by,
        "series": series,
        "filters": list(filters.values()),
        "time_filter": time_filter,
        "top_n": top_n,
        "sort": sort,
        "field_wells": field_wells,
        "confidence": {
            "chart_type": None if chart_confidence is None else round(float(chart_confidence), 4),
            "alternatives": [{"chart_type": c, "p": round(float(p), 4)} for c, p in (chart_alternatives or [])],
        },
        "spans": [{"label": s.label, "text": s.text, "start": s.start, "end": s.end} for s in spans],
        "warnings": warnings,
    }


def _title(measures, group_by, series, chart_type) -> str:
    names = [m["name"] for m in measures]
    t = " vs ".join(names) if chart_type == "scatter" else ", ".join(names)
    if group_by:
        t += " by " + " and ".join(g["name"] for g in group_by)
    if series:
        t += (" and " if chart_type == "matrix" else " split by ") + " and ".join(s["name"] for s in series)
    return t
