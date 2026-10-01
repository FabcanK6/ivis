"""Rule-based parser: the original keyword/lexicon version of iVIS.

It needs no model or GPU, so it serves as (a) a baseline for evaluation and
(b) a fallback in the app when no trained checkpoint is available. It emits
the same (chart type, spans) output as the BERT model, so both share the spec
builder.
"""

from __future__ import annotations

import re

from ivis import catalog
from ivis.text import Span, spans_to_bio, words

_TIME_RE = re.compile(
    r"^(?:(?:last|past|previous|trailing)\s+(?:\w+\s+)?(?:days?|weeks?|months?|quarters?|years?)"
    r"|(?:this|current)\s+(?:day|week|month|quarter|year)"
    r"|year to date|ytd|month to date|mtd"
    r"|q[1-4](?:\s+\d{4})?"
    r"|(?:in\s+)?(?:19|20)\d{2}"
    r"|since\s+\w+(?:\s+\d{4})?)$"
)
_TOPN_RE = re.compile(r"^(?:top|bottom)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|twenty)$")

CHART_RULES: list[tuple[str, list[str]]] = [
    ("matrix", ["matrix", "crosstab", "cross tab", "heatmap", "heat map", "pivot", "grid of", "as rows"]),
    ("map", [" map", "geograph", "where in the world"]),
    ("donut", ["donut", "doughnut", "ring chart"]),
    ("scatter", [" vs ", "versus", " against ", "correlation", "relationship between", "scatter"]),
    ("pie", ["pie", "share of", "percentage", "proportion", "split between", "contribution", "distribution"]),
    ("area", ["cumulative", "running total", "area chart", "build-up", "build up"]),
    ("stacked_bar", ["stacked", "broken down by", "split by", "legend", "segmented", "composition", "break down"]),
    ("table", ["table", "list ", "listing", "list them", "list of"]),
    ("card", ["how many", "kpi", "card", "single number", "headline", "what's our", "what is our",
              "what is the total", "total number"]),
    ("line", ["trend", "over time", "changed", "going up", "going down", "week over week", "track ",
              "monitor", "over the"]),
]


def _build_lexicon() -> list[tuple[tuple[str, ...], str, object]]:
    lex: list[tuple[tuple[str, ...], str, object]] = []

    def add(phrase: str, label: str, payload=None):
        toks = tuple(t.lower() for t in words(phrase))
        if toks:
            lex.append((toks, label, payload))

    for m in catalog.MEASURES:
        for s in m.synonyms + [m.display]:
            add(s, "METRIC", m)
    for d in catalog.DIMENSIONS:
        for s in d.synonyms:
            add(s, "DIM", d)
        for v in d.values:
            add(v, "FILTER", d)
    for agg, ws in catalog.AGGREGATIONS.items():
        for w in ws:
            add(w, "AGG", agg)
    for direction, ws in catalog.SORT_WORDS.items():
        for w in ws:
            add(w, "SORT", direction)
    # longest phrases first so "open queries"/"query status" beat "queries"/"status"
    lex.sort(key=lambda x: -len(x[0]))
    return lex


_LEXICON = _build_lexicon()


class RuleParser:
    name = "rules"

    def classify(self, text: str, spans: list[Span]) -> tuple[str, float]:
        t = " " + text.lower() + " "
        for chart, cues in CHART_RULES:
            if any(c in t for c in cues):
                return chart, 0.6
        groups = [catalog.resolve_dimension(s.text) for s in spans if s.label == "GROUP_BY"]
        if any(g and g.is_time for g in groups):
            return "line", 0.6
        if not groups:
            return "card", 0.4
        return "bar", 0.5

    def tag(self, tokens: list[str]) -> list[Span]:
        low = [t.lower() for t in tokens]
        spans: list[Span] = []
        i = 0
        seen_dim = False
        while i < len(low):
            # regex-based slots first (time windows, top-N, open-ended values)
            matched = False
            for k in range(min(4, len(low) - i), 0, -1):
                chunk = " ".join(low[i:i + k])
                label = None
                if _TIME_RE.match(chunk):
                    label = "TIME"
                elif _TOPN_RE.match(chunk):
                    label = "TOPN"
                elif k <= 2 and re.match(r"^site\s*#?\s*\d+$", chunk) or k == 1 and re.match(r"^[a-z]{2,5}-\d{2,4}$", chunk):
                    label = "FILTER"
                if label:
                    start = i + 1 if label == "TIME" and low[i] == "in" else i
                    spans.append(Span(label, start, i + k, " ".join(tokens[start:i + k])))
                    i += k
                    matched = True
                    break
            if matched:
                continue
            for phrase, label, payload in _LEXICON:
                k = len(phrase)
                if tuple(low[i:i + k]) == phrase:
                    if label == "DIM":
                        label = "SERIES" if seen_dim else "GROUP_BY"
                        seen_dim = True
                    spans.append(Span(label, i, i + k, " ".join(tokens[i:i + k])))
                    i += k
                    matched = True
                    break
            if not matched:
                i += 1
        return spans

    def predict(self, text: str) -> dict:
        tokens = words(text)
        spans = self.tag(tokens)
        chart, conf = self.classify(text, spans)
        tags = spans_to_bio(len(tokens), [s.as_tuple() for s in spans])
        return {"tokens": tokens, "tags": tags, "spans": spans, "chart_type": chart, "chart_confidence": conf,
                "chart_alternatives": []}
