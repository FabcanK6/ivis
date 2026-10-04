"""Power BI visual file: turn an iVIS spec into a PBIR ``visual.json``.

PBIR is Power BI's folder-based report format (Power BI Desktop: save as a Power BI Project, .pbip, with the
enhanced report format on). Each visual lives in ``<Report>/definition/pages/<page>/visuals/<visual>/visual.json``.
This module writes that file: the visual type, the fields in each role, the sort, the title and simple filters.

What is written:
* measures as model measures (``Table[Measure]`` in the data model), columns as columns
* categorical filters ("Country is Germany") as visual-level filters
* the sort order and the title

What is left to set in Power BI (listed in ``notes``): relative date windows ("last 30 days") and top N, whose
filter definitions depend on the report's date table, and any part iVIS could not map to a field.
"""

from __future__ import annotations

import hashlib
import json

from ivis import catalog

SCHEMA = "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/visualContainer/2.4.0/schema.json"

# PBIR visual type and the role each part goes to.
VISUALS: dict[str, dict] = {
    "bar": {"type": "clusteredBarChart", "axis": "Category", "legend": "Series", "measure": "Y"},
    "stacked_bar": {"type": "stackedBarChart", "axis": "Category", "legend": "Series", "measure": "Y"},
    "line": {"type": "lineChart", "axis": "Category", "legend": "Series", "measure": "Y"},
    "area": {"type": "areaChart", "axis": "Category", "legend": "Series", "measure": "Y"},
    "pie": {"type": "pieChart", "axis": "Category", "legend": None, "measure": "Y"},
    "donut": {"type": "donutChart", "axis": "Category", "legend": None, "measure": "Y"},
    "table": {"type": "tableEx", "axis": "Values", "legend": "Values", "measure": "Values"},
    "matrix": {"type": "pivotTable", "axis": "Rows", "legend": "Columns", "measure": "Values"},
    "card": {"type": "card", "axis": None, "legend": None, "measure": "Values"},
    "scatter": {"type": "scatterChart", "axis": "Category", "legend": "Series", "measure": ("X", "Y")},
    "map": {"type": "map", "axis": "Category", "legend": "Series", "measure": "Size"},
}
SIZES = {"card": (240, 140), "table": (640, 400), "matrix": (720, 400)}


def _literal(value: str) -> dict:
    return {"Literal": {"Value": "'" + str(value).replace("'", "''") + "'"}}


def _ref(field: str, kind: str, source: str | None = None) -> dict | None:
    parts = catalog.split_field(field or "")
    if not parts:
        return None
    table, col = parts
    expr = {"SourceRef": {"Source": source}} if source else {"SourceRef": {"Entity": table}}
    return {kind: {"Expression": expr, "Property": col}}


def _projection(field: str, kind: str) -> dict | None:
    ref = _ref(field, kind)
    if ref is None:
        return None
    table, col = catalog.split_field(field)
    return {"field": ref, "queryRef": f"{table}.{col}", "nativeQueryRef": col}


def visual_name(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec.get("source_text", spec["title"])).encode()).hexdigest()[:20]


def to_visual_json(spec: dict, cat: catalog.Catalog | None = None) -> tuple[dict, list[str]]:
    """The PBIR ``visual.json`` for a spec, plus notes on what to finish in Power BI."""
    cat = cat or catalog.DEFAULT
    layout = VISUALS[spec["chart_type"]]
    notes: list[str] = []
    roles: dict[str, list[dict]] = {}

    def put(role: str | None, proj: dict | None, what: str) -> None:
        if role and proj:
            roles.setdefault(role, []).append({"projection": proj, "kind": what})

    for g in spec["group_by"]:
        put(layout["axis"], _projection(g["field"], "Column"), "axis")
    if layout["legend"]:
        for s in spec["series"]:
            put(layout["legend"], _projection(s["field"], "Column"), "legend")
    elif spec["series"]:
        notes.append(f"A {spec['chart_type']} has no legend, so {', '.join(s['name'] for s in spec['series'])} was "
                     "left out.")
    measures = [m for m in spec["measures"] if m.get("field")]
    for m in spec["measures"]:
        if not m.get("field"):
            notes.append(f"'{m['name']}' is not in the data model, so it is not in the visual.")
            continue
        model_measure = cat.measure_by_field(m["field"])
        pie_share = spec["chart_type"] in ("pie", "donut") and m.get("aggregation") == "percent_of_total"
        if model_measure and m.get("aggregation") not in (None, model_measure.default_agg) and not pie_share:
            notes.append(f"{m['name']} is a model measure, so Power BI uses its own calculation; the request asked "
                         f"for {m['aggregation'].replace('_', ' ')}.")
    if isinstance(layout["measure"], tuple):
        for role, m in zip(layout["measure"], measures):
            put(role, _projection(m["field"], "Measure"), "measure")
    else:
        for m in measures:
            put(layout["measure"], _projection(m["field"], "Measure"), "measure")
    for x in [*spec["group_by"], *spec["series"]]:
        if not x.get("field"):
            notes.append(f"'{x['name']}' is not in the data model, so it is not in the visual.")

    query: dict = {"queryState": {role: {"projections": [i["projection"] for i in items]}
                                  for role, items in roles.items()}}
    if spec.get("sort") and measures and spec["chart_type"] not in ("line", "area"):
        ref = _ref(measures[0]["field"], "Measure")
        if ref:
            query["sortDefinition"] = {"sort": [{"field": ref, "direction": "Descending"
                                                 if spec["sort"]["direction"] == "desc" else "Ascending"}],
                                       "isDefaultSort": False}

    filters = []
    for f in spec["filters"]:
        parts = catalog.split_field(f.get("field") or "")
        if not parts or not f["values"]:
            notes.append(f"The filter {', '.join(f['values'])} has no field, so it was left out.")
            continue
        table, _col = parts
        filters.append({
            "name": hashlib.sha256(f"{f['field']}{f['values']}".encode()).hexdigest()[:20],
            "field": _ref(f["field"], "Column"),
            "type": "Categorical",
            "filter": {
                "Version": 2,
                "From": [{"Name": "t", "Entity": table, "Type": 0}],
                "Where": [{"Condition": {"In": {
                    "Expressions": [_ref(f["field"], "Column", source="t")],
                    "Values": [[_literal(v)] for v in f["values"]],
                }}}],
            },
        })
    if spec.get("time_filter"):
        t = spec["time_filter"]
        kind = "a relative date filter" if t.get("filterType") == "RelativeDate" else "a date filter"
        notes.append(f"Time window \"{t['text']}\": add {kind} on {t.get('field') or 'your date column'} in Power BI's "
                     "filter pane.")
    if spec.get("top_n"):
        t = spec["top_n"]
        axis = spec["group_by"][0]["name"] if spec["group_by"] else "the axis"
        notes.append(f"Top {t['n']}: in the filter pane, set {axis} to Top N = {t['n']} by {t['by']}.")

    width, height = SIZES.get(spec["chart_type"], (640, 360))
    visual: dict = {"visualType": layout["type"], "query": query,
                    "visualContainerObjects": {"title": [{"properties": {
                        "show": {"expr": {"Literal": {"Value": "true"}}},
                        "text": {"expr": _literal(spec["title"])}}}]},
                    "drillFilterOtherVisuals": True}
    out = {"$schema": SCHEMA, "name": visual_name(spec),
           "position": {"x": 40, "y": 40, "z": 0, "width": width, "height": height, "tabOrder": 0},
           "visual": visual}
    if filters:
        out["filterConfig"] = {"filters": filters}
    if spec["chart_type"] == "map":
        notes.append("The map is a bubble map (the measure sets the bubble size); switch to a filled map in Power BI "
                     "if you prefer shaded countries.")
    return out, notes


HOW_TO = """How to add it to a report:
1. In Power BI Desktop, turn on File > Options > Preview features > "Power BI Project (.pbip) save option" and
   "Store reports using enhanced metadata format (PBIR)", then save your report as a Power BI Project (.pbip).
2. Close Power BI Desktop. In the project folder, open <Report>.Report/definition/pages/ and the folder of the page
   you want; inside its visuals folder, make a new folder (any name, e.g. the file's "name") and put visual.json in it.
3. Reopen the .pbip file. The visual appears on that page; finish the items listed under "Finish in Power BI".
The field names come from the data model in iVIS, so they must match your Power BI model exactly."""
