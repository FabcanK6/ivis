"""AI reading mode: an AI model (Gemini) reads the request; plain code checks what it says.

The model returns the chart type and the parts of the request (measure, axis, legend, filters, time window, top N,
sort), each with the exact words from the request and, where it applies, the Power BI field it chose from the data
model. Before anything reaches the spec, code checks that:

* the words really are in the request (a part whose words are not there is dropped and listed), and
* the field really is in the data model, and is the right kind (a measure for a measure, a column for an axis,
  legend or filter); otherwise the field is ignored and the words are matched against the data model instead.

The checked parts then go through the same spec builder as the BERT and keyword readers, so the output is the same
Power BI spec whichever reader was used.
"""

from __future__ import annotations

import json

from ivis import catalog
from ivis.schema import CHART_TYPES
from ivis.spec import build_spec
from ivis.text import Span, spans_to_bio, words

ROLES = {"measure": "METRIC", "aggregation": "AGG", "axis": "GROUP_BY", "legend": "SERIES", "filter": "FILTER",
         "time": "TIME", "top_n": "TOPN", "sort": "SORT"}
FIELD_ROLES = {"measure": "measure", "axis": "column", "legend": "column", "filter": "column"}

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "chart_type": {"type": "STRING", "enum": CHART_TYPES},
        "parts": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "role": {"type": "STRING", "enum": list(ROLES)},
                    "text": {"type": "STRING"},
                    "field": {"type": "STRING"},
                },
                "required": ["role", "text"],
            },
        },
    },
    "required": ["chart_type", "parts"],
}

SYSTEM = """You turn a plain-English request for a Power BI chart into its parts. Answer only with JSON.

chart_type, one of:
- bar: compare a measure across categories; rankings, top N
- stacked_bar: a measure by category, split into segments by a second column (legend)
- line: a measure over time (trend, by month/week)
- area: a cumulative or running total over time
- pie / donut: shares or proportions of a whole (donut only when asked for)
- table: a list or listing of rows
- matrix: a measure with one column as rows and another as columns (crosstab, heatmap, pivot)
- card: one number (how many, total, KPI, headline figure)
- scatter: two measures against each other (vs, correlation, relationship)
- map: a measure by country or region on a map

parts: every part of the request, in the order it appears. Each part has:
- role: measure (what is counted or added up), aggregation (words like average, total, how many, number of),
  axis (the main grouping: x axis, rows, categories, pie slices), legend (a second split: series, columns,
  segments), filter (one value to filter on, e.g. a country, a site, a status), time (the time window, e.g. last
  30 days, this quarter, since January), top_n (e.g. top 10), sort (e.g. highest, fewest)
- text: the exact words from the request, copied as written, without words around them (for a filter just the
  value, e.g. "Germany", not "in Germany"; for an axis just the column word, e.g. "site", not "by site")
- field: for measure, axis, legend and filter parts, the Power BI field from the data model below, written exactly
  as listed (Table[Column]); leave it empty if nothing in the data model fits

Never add words that are not in the request. A request may have no time window, filter or legend: leave them out.
A scatter plot has two measures.

Data model:
"""


def _locate(tokens: list[str], text: str, used: set[int]) -> tuple[int, int] | None:
    want = [w.lower() for w in words(text)]
    low = [t.lower() for t in tokens]
    k = len(want)
    if not k:
        return None
    for i in range(len(low) - k + 1):
        if low[i:i + k] == want and not used & set(range(i, i + k)):
            return i, i + k
    return None


class AIParser:
    name = "ai"

    def __init__(self, client, cat: catalog.Catalog | None = None):
        self.client = client
        self.cat = cat or catalog.DEFAULT

    def ask(self, text: str) -> dict:
        system = SYSTEM + self.cat.prompt_text()
        return self.client.generate_json(system, f"Request: {text}", SCHEMA)

    def check(self, text: str, answer: dict) -> dict:
        """The AI's answer, checked against the request and the data model, as (chart type, spans, notes)."""
        tokens = words(text)
        notes: list[str] = []
        chart = answer.get("chart_type") if answer.get("chart_type") in CHART_TYPES else None
        if chart is None:
            notes.append(f"The AI gave no usable chart type ({answer.get('chart_type')!r}); using a bar chart.")
            chart = "bar"
        spans: list[Span] = []
        used: set[int] = set()
        for part in answer.get("parts") or []:
            role, said = part.get("role"), str(part.get("text") or "").strip()
            if role not in ROLES or not said:
                continue
            loc = _locate(tokens, said, used)
            if loc is None:
                notes.append(f'The AI read "{said}" as the {role.replace("_", " ")}, but those words are not in the '
                             "request, so it was left out.")
                continue
            field = str(part.get("field") or "").strip() or None
            if field and role in FIELD_ROLES:
                ok = (self.cat.measure_by_field(field) if FIELD_ROLES[role] == "measure"
                      else self.cat.dimension_by_field(field))
                if not ok:
                    notes.append(f'The AI chose {field} for "{said}", which is not a '
                                 f'{"measure" if FIELD_ROLES[role] == "measure" else "column"} in the data model; '
                                 "the words were matched to the data model instead.")
                    field = None
            elif role not in FIELD_ROLES:
                field = None
            used |= set(range(*loc))
            spans.append(Span(ROLES[role], loc[0], loc[1], " ".join(tokens[loc[0]:loc[1]]), field))
        spans.sort(key=lambda s: s.start)
        return {"tokens": tokens, "spans": spans, "tags": spans_to_bio(len(tokens), [s.as_tuple() for s in spans]),
                "chart_type": chart, "chart_confidence": None, "chart_alternatives": [], "notes": notes}

    def predict(self, text: str) -> dict:
        pred = self.check(text, self.ask(text))
        pred["model"] = getattr(self.client, "model", None)
        return pred

    def parse(self, text: str, answer: dict | None = None) -> dict:
        pred = self.check(text, answer if answer is not None else self.ask(text))
        spec = build_spec(text, pred["tokens"], pred["chart_type"], pred["spans"], None, [], cat=self.cat)
        spec["backend"] = "ai"
        spec["ai_model"] = getattr(self.client, "model", None)
        spec["warnings"] = pred["notes"] + spec["warnings"]
        spec["fields_from_ai"] = [{"text": s.text, "field": s.field} for s in pred["spans"] if s.field]
        return spec


def answer_key(text: str, cat: catalog.Catalog) -> str:
    """Cache key for an AI answer: the request and the data model it was read against."""
    return json.dumps([" ".join(text.split()).lower(), cat.fingerprint()])
