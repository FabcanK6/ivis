"""iVIS Streamlit front end.

    streamlit run app/streamlit_app.py
    IVIS_MODEL_DIR=models/ivis-bert streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivis.catalog import DIMENSIONS  # noqa: E402
from ivis.devops import to_json_patch, to_markdown  # noqa: E402
from ivis.predict import RuleBasedParser, load_parser  # noqa: E402
from ivis.text import words  # noqa: E402

MODEL_DIR = os.environ.get("IVIS_MODEL_DIR", "models/ivis-bert")
EXAMPLES = [
    "Can we get a chart of top 10 sites with the most open queries in Germany last 30 days?",
    "Trend of SAEs by month for ONC-301 this year",
    "How many screen failures do we have at Site 104 since January?",
    "Protocol deviations by country broken down by deviation category in 2025",
    "Share of adverse events by severity for the placebo arm",
    "Map of enrollment by country YTD",
    "Matrix of missing pages by site and visit for CRA Okafor",
    "Query aging vs SDV completion by site this quarter",
    "List of sites with outstanding missing pages in EMEA",
]

st.set_page_config(page_title="iVIS - Visualization Intent Engine", page_icon="📊", layout="wide")


@st.cache_resource
def get_parser(backend: str):
    return RuleBasedParser() if backend == "rules" else load_parser(MODEL_DIR)


def _dim_values(field: str | None, n: int) -> list[str]:
    for d in DIMENSIONS:
        if d.field == field:
            if d.key in ("month", "week", "quarter", "day", "year"):
                return {
                    "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"],
                    "quarter": ["Q1", "Q2", "Q3", "Q4"],
                    "week": [f"W{i}" for i in range(1, 13)],
                    "year": ["2023", "2024", "2025", "2026"],
                    "day": [f"Day {i}" for i in range(1, 15)],
                }[d.key]
            vals = d.values or [f"{d.display} {chr(65 + i)}" for i in range(6)]
            return vals[:n]
    return [f"Item {i}" for i in range(1, n + 1)]


def mock_frame(spec: dict) -> pd.DataFrame:
    """Placeholder data shaped like the spec, so stakeholders can sanity-check the layout."""
    rng = random.Random(hash(spec["source_text"]) % 10_000)
    m = spec["measures"][0]["name"]
    g = spec["group_by"][0] if spec["group_by"] else None
    s = spec["series"][0] if spec["series"] else None
    groups = _dim_values(g["field"], 8) if g else ["All"]
    if spec.get("top_n"):
        groups = groups[: spec["top_n"]["n"]]
    series = _dim_values(s["field"], 4) if s else [None]
    rows = []
    for gv in groups:
        for sv in series:
            row = {g["name"] if g else "Group": gv, m: rng.randint(5, 120)}
            if s:
                row[s["name"]] = sv
            if len(spec["measures"]) > 1:
                row[spec["measures"][1]["name"]] = rng.randint(5, 120)
            rows.append(row)
    df = pd.DataFrame(rows)
    if spec["chart_type"] == "area":
        df[m] = df[m].cumsum()
    return df


def preview(spec: dict):
    df = mock_frame(spec)
    m = spec["measures"][0]["name"]
    g = spec["group_by"][0]["name"] if spec["group_by"] else "Group"
    s = spec["series"][0]["name"] if spec["series"] else None
    ct = spec["chart_type"]
    order = "-x" if (spec.get("sort") or {}).get("direction") == "desc" else "x"
    color = alt.Color(f"{s}:N") if s else alt.value("#4C78A8")
    if ct == "card":
        st.metric(spec["title"], f"{int(df[m].sum()):,}")
        return
    if ct in ("table", "map"):
        if ct == "map":
            st.caption("Map preview is shown as a table; the spec targets a filled map in Power BI.")
        st.dataframe(df, use_container_width=True, hide_index=True)
        return
    if ct == "matrix" and s:
        st.dataframe(df.pivot_table(index=g, columns=s, values=m, aggfunc="sum"), use_container_width=True)
        return
    if ct in ("pie", "donut"):
        chart = alt.Chart(df).mark_arc(innerRadius=60 if ct == "donut" else 0).encode(
            theta=f"{m}:Q", color=f"{g}:N")
    elif ct == "scatter" and len(spec["measures"]) > 1:
        m2 = spec["measures"][1]["name"]
        chart = alt.Chart(df).mark_circle(size=90).encode(x=f"{m}:Q", y=f"{m2}:Q", tooltip=[g])
    elif ct in ("line", "area"):
        mark = alt.Chart(df).mark_line(point=True) if ct == "line" else alt.Chart(df).mark_area(opacity=0.7)
        chart = mark.encode(x=alt.X(f"{g}:N", sort=None), y=f"{m}:Q", color=color)
    else:
        chart = alt.Chart(df).mark_bar().encode(
            y=alt.Y(f"{g}:N", sort=order if not s else None), x=f"{m}:Q", color=color)
    st.altair_chart(chart.properties(height=360, title=spec["title"]), use_container_width=True)


# ---------------------------------------------------------------------------
st.title("📊 iVIS")
st.caption("Intelligent Visualization Insight Synthesizer · turn plain-English dashboard requests into Power BI specs")

with st.sidebar:
    has_model = (Path(MODEL_DIR) / "ivis_model.pt").exists()
    backend = st.radio("Parser", ["bert", "rules"], index=0 if has_model else 1,
                       help="'bert' needs a trained checkpoint (python -m ivis.train).")
    if backend == "bert" and not has_model:
        st.warning(f"No checkpoint at `{MODEL_DIR}`, so the rule-based parser is used instead.")
    st.markdown("**Examples**")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state["request"] = ex

text = st.text_area("Stakeholder request", key="request", height=90,
                    placeholder="e.g. Can we get a chart of top queries by site?")
if text.strip():
    parser = get_parser(backend)
    spec = parser.parse(text)

    c1, c2, c3 = st.columns(3)
    c1.metric("Visual", spec["powerbi_visual"])
    conf = spec["confidence"]["chart_type"]
    c2.metric("Chart confidence", f"{conf:.0%}" if conf is not None else "n/a")
    c3.metric("Backend", spec["backend"])
    if spec["confidence"]["alternatives"]:
        st.caption("Alternatives: " + ", ".join(f"{a['chart_type']} ({a['p']:.0%})"
                                                for a in spec["confidence"]["alternatives"]))
    for w in spec["warnings"]:
        st.warning(w)

    tab_prev, tab_spec, tab_ticket, tab_spans = st.tabs(["Preview", "Spec JSON", "DevOps ticket", "Detected spans"])
    with tab_prev:
        st.caption("Mock data, for checking the layout only.")
        preview(spec)
    with tab_spec:
        st.json(spec)
        st.download_button("Download spec.json", json.dumps(spec, indent=2), "ivis_spec.json", "application/json")
    with tab_ticket:
        md = to_markdown(spec)
        st.markdown(md)
        st.download_button("Download work item (JSON Patch)", json.dumps(to_json_patch(spec), indent=2),
                           "ivis_workitem.json", "application/json")
    with tab_spans:
        colors = {"METRIC": "#4C78A8", "GROUP_BY": "#F58518", "SERIES": "#E45756", "FILTER": "#54A24B",
                  "TIME": "#B279A2", "TOPN": "#9D755D", "SORT": "#72B7B2", "AGG": "#EECA3B"}
        html = []
        tokens = words(text)
        starts = {sp["start"]: sp for sp in spec["spans"]}
        i = 0
        while i < len(tokens):
            if i in starts:
                sp = starts[i]
                html.append(f"<span style='background:{colors.get(sp['label'], '#ccc')};color:white;"
                            f"padding:2px 6px;border-radius:6px;margin:0 2px'>{sp['text']} "
                            f"<sub>{sp['label']}</sub></span>")
                i = sp["end"]
            else:
                html.append(tokens[i])
                i += 1
        st.markdown(" ".join(html), unsafe_allow_html=True)
