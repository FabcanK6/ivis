"""iVIS Streamlit front end.

    streamlit run app/streamlit_app.py
    IVIS_MODEL_DIR=models/ivis-bert streamlit run app/streamlit_app.py

Optional: GEMINI_API_KEY (environment or Streamlit secrets) turns on the AI reader.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivis import catalog as C
from ivis.ai import AIParser, answer_key
from ivis.benchmark import sample, score, test_split
from ivis.devops import to_json_patch, to_markdown
from ivis.llm import GeminiClient, LLMError, get_api_key
from ivis.pbir import HOW_TO, to_visual_json
from ivis.predict import BertParser, HybridParser, RuleBasedParser, to_spec
from ivis.text import words

MODEL_DIR = os.environ.get("IVIS_MODEL_DIR", "models/ivis-bert")
# Public Hugging Face repo with the trained checkpoint; downloaded on first run if MODEL_DIR is empty.
HF_MODEL_REPO = os.environ.get("IVIS_HF_MODEL", "FabcanK6/ivis-bert")
AI_LIMIT = 30  # new AI readings per browser session on the shared free key
BROWSER_KEY = "ivis.datamodel.v1"
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
READERS = {
    "hybrid": "BERT + keyword cues",
    "ai": "AI (Gemini)",
    "bert": "BERT only",
    "rules": "Keywords only",
}
BUILT_IN = "Clinical operations (built-in)"
MINE = "My data model"
BLANK_ROWS = [
    {"Kind": "Measure", "Field": "Sales[Revenue]", "Also called": "sales, turnover", "Values": "",
     "Default aggregation": "sum"},
    {"Kind": "Column", "Field": "Product[Category]", "Also called": "product line", "Values": "Bikes, Accessories",
     "Default aggregation": ""},
    {"Kind": "Location column", "Field": "Store[Country]", "Also called": "", "Values": "Germany, France",
     "Default aggregation": ""},
    {"Kind": "Date column", "Field": "Calendar[Month]", "Also called": "", "Values": "", "Default aggregation": ""},
    {"Kind": "Date column", "Field": "Calendar[Date]", "Also called": "", "Values": "", "Default aggregation": ""},
]
ROW_COLUMNS = ["Kind", "Field", "Also called", "Values", "Default aggregation"]

st.set_page_config(page_title="iVIS - Visualization Intent Engine", page_icon="📊", layout="wide")

try:  # keeps the user's own data model in their browser between visits (nothing stored on the server)
    from streamlit_js_eval import streamlit_js_eval as _browser_js
except ImportError:  # pragma: no cover - the app still works; the data model then lasts for the session only
    _browser_js = None


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Downloading the iVIS BERT model (first run only)...")
def ensure_model() -> bool:
    """Make sure a checkpoint exists locally, fetching it from the Hugging Face Hub if needed."""
    if (Path(MODEL_DIR) / "ivis_model.pt").exists():
        return True
    if not HF_MODEL_REPO:
        return False
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(HF_MODEL_REPO, local_dir=MODEL_DIR)
    except Exception as exc:  # noqa: BLE001 - network issues, missing repo, ...: fall back to keywords
        st.sidebar.warning(f"Could not download `{HF_MODEL_REPO}`: {exc}")
    return (Path(MODEL_DIR) / "ivis_model.pt").exists()


@st.cache_resource(show_spinner="Loading the BERT model...")
def get_bert() -> BertParser | None:
    try:
        return BertParser(MODEL_DIR) if ensure_model() else None
    except Exception as exc:  # noqa: BLE001 - torch missing, broken checkpoint, ...
        st.sidebar.warning(f"The BERT model could not be loaded ({exc}); the keyword reader is used instead.")
        return None


def api_key() -> str | None:
    return get_api_key(st.secrets) if hasattr(st, "secrets") else get_api_key()


def ai_client() -> GeminiClient | None:
    key = api_key()
    if not key:
        return None
    if "ai_client" not in st.session_state:
        st.session_state["ai_client"] = GeminiClient(key)
    return st.session_state["ai_client"]


@st.cache_data(ttl=24 * 3600, show_spinner=False, max_entries=2000)
def ai_answer(key: str, text: str, _parser: AIParser) -> dict:
    """The AI's answer for a request and data model, shared by everyone for a day (same request, same answer)."""
    st.session_state["ai_used"] = st.session_state.get("ai_used", 0) + 1  # only counts answers not in the cache
    return _parser.ask(text)


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False, max_entries=500)
def ai_answer_check(key: str, text: str, _parser: AIParser) -> dict:
    """Answers for the accuracy check (kept a week, not counted against the session's limit)."""
    return _parser.ask(text)


def read_request(text: str, reader: str, cat: C.Catalog) -> dict:
    bert = get_bert()
    if reader in ("hybrid", "bert") and bert is None:
        reader = "rules"
    if reader == "rules":
        return RuleBasedParser(cat).parse(text)
    if reader == "bert":
        return to_spec(text, bert.predict(text), "bert", cat)
    if reader == "hybrid":
        return to_spec(text, HybridParser(bert, cat).predict(text), "hybrid", cat)
    client = ai_client()
    parser = AIParser(client, cat)
    if st.session_state.get("ai_used", 0) >= AI_LIMIT:
        raise LLMError(f"This session has used its {AI_LIMIT} AI readings on the shared free key. Pick another reader "
                       "in the sidebar, or come back later.")
    t0 = time.monotonic()
    client.trail = []
    answer = ai_answer(answer_key(text, cat), text, parser)
    spec = parser.parse(text, answer=answer)
    spec["ai_seconds"] = round(time.monotonic() - t0, 1)
    spec["ai_trail"] = list(client.trail)
    return spec


# ---------------------------------------------------------------------------
# Data model (built-in, or the user's own, kept in their browser)
# ---------------------------------------------------------------------------
def browser_load() -> None:
    if _browser_js is None or st.session_state.get("dm_loaded"):
        return
    raw = _browser_js(js_expressions=f"localStorage.getItem('{BROWSER_KEY}') || 'null'", key="ivis_dm_read")
    if raw is None:
        return
    try:
        rows = json.loads(raw)
    except ValueError:
        rows = None
    if isinstance(rows, list) and rows and "my_rows" not in st.session_state:
        st.session_state["my_rows"] = [r for r in rows if isinstance(r, dict)]
    st.session_state["dm_loaded"] = True


def browser_save(rows: list[dict] | None) -> None:
    if _browser_js is None:
        return
    payload = json.dumps(rows) if rows else "null"
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    _browser_js(js_expressions=f"localStorage.setItem('{BROWSER_KEY}', {json.dumps(payload)}); 'saved'",
                key=f"ivis_dm_write_{digest}")


@st.cache_resource(max_entries=50)
def catalog_from(rows_json: str) -> tuple[C.Catalog, list[str]]:
    return C.Catalog.from_rows(json.loads(rows_json), name=MINE)


def active_catalog(choice: str) -> C.Catalog:
    rows = st.session_state.get("my_rows")
    if choice == MINE and rows:
        return catalog_from(json.dumps(rows, sort_keys=True))[0]
    return C.DEFAULT


def data_model_tab() -> None:
    st.subheader("Your data model")
    st.markdown(
        "iVIS maps each request onto a **data model**: the measures and columns in your Power BI dataset. "
        "Use the built-in clinical-operations model, or describe your own below. Write each field exactly as it "
        "appears in Power BI, as `Table[Column]`. iVIS adds the obvious other names itself (\"Site Name\" also "
        "matches \"site\" and \"sites\"); add any others people use under **Also called**.")
    st.caption("Kinds: **Measure** = something to count or add up · **Column** = something to group by or filter on "
               "· **Date column** = a date or a time grain (Month, Week) · **Location column** = country or region, "
               "for maps. **Values** are example values people filter on (\"Germany\", \"open\").")
    c1, c2, c3 = st.columns(3)
    if c1.button("Start from the built-in model", width="stretch"):
        st.session_state["draft_rows"] = C.DEFAULT.to_rows()
    if c2.button("Start blank (with examples)", width="stretch"):
        st.session_state["draft_rows"] = list(BLANK_ROWS)
    up = c3.file_uploader("Upload a CSV", type=["csv"], label_visibility="collapsed")
    if up is not None and st.session_state.get("uploaded") != up.file_id:
        try:
            df = pd.read_csv(up).fillna("")
            missing = [c for c in ("Kind", "Field") if c not in df.columns]
            if missing:
                st.error(f"The CSV needs the columns {', '.join(ROW_COLUMNS)} (missing: {', '.join(missing)}).")
            else:
                st.session_state["draft_rows"] = df.reindex(columns=ROW_COLUMNS).fillna("").to_dict("records")
                st.session_state["uploaded"] = up.file_id
        except (ValueError, UnicodeDecodeError) as exc:  # pandas' parser errors are ValueErrors
            st.error(f"Could not read that CSV ({exc}).")
    draft = st.session_state.get("draft_rows") or st.session_state.get("my_rows") or list(BLANK_ROWS)
    edited = st.data_editor(
        pd.DataFrame(draft).reindex(columns=ROW_COLUMNS).fillna(""), num_rows="dynamic", width="stretch",
        hide_index=True, key="dm_editor",
        column_config={
            "Kind": st.column_config.SelectboxColumn(options=list(C.Catalog.ROW_KINDS), required=True),
            "Field": st.column_config.TextColumn(help="As in Power BI: Table[Column], e.g. Sales[Revenue]"),
            "Also called": st.column_config.TextColumn(help="Other words people use, separated by commas"),
            "Values": st.column_config.TextColumn(help="Example values people filter on, separated by commas"),
            "Default aggregation": st.column_config.SelectboxColumn(options=["", *C.Catalog.AGG_CHOICES]),
        })
    rows = [r for r in edited.fillna("").to_dict("records") if str(r.get("Field") or "").strip()]
    cat, problems = C.Catalog.from_rows(rows, name=MINE)
    for p in problems:
        st.warning(p)
    n_cols = len([d for d in cat.dimensions if not d.is_time])
    st.caption(f"{len(cat.measures)} measure{'s' * (len(cat.measures) != 1)} · {n_cols} column{'s' * (n_cols != 1)} · "
               f"date column: {cat.date_field or 'none'} · "
               f"location columns: {', '.join(d.field for d in cat.geo_dimensions()) or 'none'}")
    b1, b2, b3 = st.columns(3)
    if b1.button("Use this data model", type="primary", width="stretch", disabled=not cat.measures):
        st.session_state["my_rows"] = rows
        st.session_state["draft_rows"] = rows
        st.session_state["pending_dm"] = MINE
        browser_save(rows)
        st.success("Saved in this browser. Requests are now read against your data model (sidebar: Data model).")
    b2.download_button("Download as CSV", pd.DataFrame(rows, columns=ROW_COLUMNS).to_csv(index=False),
                       "ivis_data_model.csv", "text/csv", width="stretch")
    if st.session_state.get("my_rows") and b3.button("Forget my data model", width="stretch"):
        st.session_state.pop("my_rows", None)
        st.session_state.pop("draft_rows", None)
        st.session_state["pending_dm"] = BUILT_IN
        browser_save(None)
        st.rerun()
    st.info("The BERT model learned the built-in model's wording. With your own data model, the **AI (Gemini)** "
            "reader usually maps requests best, because it reads your field list.")


# ---------------------------------------------------------------------------
# Preview with mock data
# ---------------------------------------------------------------------------
TIME_VALUES = {
    "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"],
    "quarter": ["Q1", "Q2", "Q3", "Q4"],
    "week": [f"W{i}" for i in range(1, 13)],
    "year": ["2023", "2024", "2025", "2026"],
}


def _dim_values(cat: C.Catalog, field: str | None, n: int) -> list[str]:
    d = cat.dimension_by_field(field)
    if d is None:
        return [f"Item {i}" for i in range(1, n + 1)]
    if d.is_time:
        name = d.display.lower()
        return next((v for k, v in TIME_VALUES.items() if k in name), [f"Day {i}" for i in range(1, 15)])
    vals = d.values or [f"{d.display} {chr(65 + i)}" for i in range(6)]
    return vals[:n]


def mock_frame(spec: dict, cat: C.Catalog) -> pd.DataFrame:
    """Placeholder data shaped like the spec, so stakeholders can sanity-check the layout."""
    rng = random.Random(hash(spec["source_text"]) % 10_000)
    m = spec["measures"][0]["name"]
    g = spec["group_by"][0] if spec["group_by"] else None
    s = spec["series"][0] if spec["series"] else None
    groups = _dim_values(cat, g["field"], 8) if g else ["All"]
    if spec.get("top_n"):
        groups = groups[: spec["top_n"]["n"]]
    series = _dim_values(cat, s["field"], 4) if s else [None]
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


def preview(spec: dict, cat: C.Catalog):
    df = mock_frame(spec, cat)
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
            st.caption("Map preview is shown as a table.")
        st.dataframe(df, width="stretch", hide_index=True)
        return
    if ct == "matrix" and s:
        st.dataframe(df.pivot_table(index=g, columns=s, values=m, aggfunc="sum"), width="stretch")
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
    st.altair_chart(chart.properties(height=360, title=spec["title"]), width="stretch")


def spans_html(text: str, spec: dict) -> str:
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
    return " ".join(html)


def chosen_by(spec: dict) -> str:
    if spec["backend"] == "ai":
        return spec.get("ai_model") or "AI"
    if spec.get("chart_source", "BERT") != "BERT":
        return spec["chart_source"]
    conf = spec["confidence"]["chart_type"]
    name = {"hybrid": "BERT", "bert": "BERT", "rules": "keywords"}[spec["backend"]]
    return f"{name} ({conf:.0%})" if conf is not None else name


def request_tab(reader: str, cat: C.Catalog) -> None:
    text = st.text_area("Stakeholder request", key="request", height=90,
                        placeholder="e.g. Can we get a chart of top queries by site?")
    if not text.strip():
        st.info("Type a request, or pick an example in the sidebar. iVIS turns it into a Power BI visual: a preview, "
                "a ready visual file for your report, and a work item.")
        return
    try:
        spec = read_request(text, reader, cat)
    except LLMError as exc:
        st.error(str(exc))
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Visual", spec["powerbi_visual"])
    c2.metric("Chart chosen by", chosen_by(spec))
    c3.metric("Reader", READERS.get(spec["backend"], spec["backend"]))
    if spec["confidence"]["alternatives"]:
        st.caption("Other charts considered: " + ", ".join(
            f"{a['chart_type']} ({a['p']:.0%})" for a in spec["confidence"]["alternatives"] if a["p"] is not None))
    if spec["backend"] == "ai":
        st.caption("How this was read: " + (" → ".join(spec["ai_trail"]) if spec.get("ai_trail")
                                            else "from the cache (this request was read earlier today)"))
    for w in spec["warnings"]:
        st.warning(w)

    visual, notes = to_visual_json(spec, cat)
    t_prev, t_file, t_ticket, t_spec, t_spans = st.tabs(
        ["Preview", "Power BI visual file", "DevOps ticket", "Spec JSON", "Detected parts"])
    with t_prev:
        st.caption("Mock data, for checking the layout only.")
        preview(spec, cat)
    with t_file:
        st.markdown(f"A **{visual['visual']['visualType']}** visual for a Power BI report saved in the PBIR format. "
                    "Download it, put it in your report's folder, and reopen the report.")
        st.download_button("Download visual.json", json.dumps(visual, indent=2), "visual.json", "application/json",
                           type="primary")
        if notes:
            st.markdown("**Finish in Power BI:**\n" + "\n".join(f"- {n}" for n in notes))
        with st.expander("How to add it to a report"):
            st.text(HOW_TO)
        with st.expander("Show the file"):
            st.json(visual)
    with t_ticket:
        st.markdown(to_markdown(spec))
        st.download_button("Download work item (JSON Patch)", json.dumps(to_json_patch(spec), indent=2),
                           "ivis_workitem.json", "application/json")
    with t_spec:
        st.json(spec)
        st.download_button("Download spec.json", json.dumps(spec, indent=2), "ivis_spec.json", "application/json")
    with t_spans:
        st.markdown(spans_html(text, spec), unsafe_allow_html=True)
        if spec.get("fields_from_ai"):
            st.caption("Fields the AI chose (each checked against the data model): " + "; ".join(
                f"\"{f['text']}\" → {f['field']}" for f in spec["fields_from_ai"]))


# ---------------------------------------------------------------------------
# Accuracy check
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=2)
def bert_predictions() -> list[dict]:
    """BERT's readings of the whole test split (about a minute on the app's CPU; kept for the session's server)."""
    bert = get_bert()
    texts = [r["text"] for r in test_split()]
    preds = []
    for i in range(0, len(texts), 64):
        preds += bert.predict_batch(texts[i:i + 64])
    return preds


@st.cache_data(show_spinner=False, max_entries=8)
def run_check(reader: str, n: int) -> dict:
    rows = test_split()
    if reader in ("bert", "hybrid"):
        preds = bert_predictions()
        if reader == "hybrid":
            preds = [HybridParser.apply_cues(r["text"], p) for r, p in zip(rows, preds)]
    elif reader == "rules":
        p = RuleBasedParser()
        preds = [p.predict(r["text"]) for r in rows]
    else:
        rows = sample(rows, n)
        parser = AIParser(ai_client())
        preds = []
        for r in rows:
            for attempt in range(3):  # the free tier allows only a few requests a minute per model
                try:
                    answer = ai_answer_check(answer_key(r["text"], C.DEFAULT), r["text"], parser)
                    break
                except LLMError:
                    if attempt == 2:
                        raise
                    time.sleep(30)
            preds.append(parser.check(r["text"], answer))
    report = score(rows, preds)
    report["reader"] = reader
    return report


def accuracy_tab() -> None:
    st.subheader("Accuracy check")
    st.markdown(
        "Runs a reader on the standard test set: 2,000 requests made by iVIS's generator (built-in data model). "
        "Half use phrasings the BERT model saw in training; the other half use **new wording** it never saw. "
        "**Same visual** compares the Power BI visual each reading produces with the one the labels describe "
        "(chart, measures, axis, legend, filters, time window, top N).")
    st.caption("These are generated requests, so the scores overstate real-world accuracy. The AI reader is "
               "scored on a fixed sample, half new wording, because each request uses free AI quota.")
    has_bert = get_bert() is not None
    options = [r for r in ("rules", "bert", "hybrid", "ai") if (r != "ai" or api_key())
               and (r not in ("bert", "hybrid") or has_bert)]
    picked = st.multiselect("Readers", options, default=[r for r in options if r != "ai"],
                            format_func=lambda r: READERS[r])
    ai_n = st.select_slider("AI sample size", [20, 40, 100], value=40) if "ai" in picked else 0
    if not st.button("Run", type="primary"):
        if not st.session_state.get("check_results"):
            return
    else:
        results = []
        for r in picked:
            with st.spinner(f"Running {READERS[r]}..."):
                try:
                    results.append(run_check(r, ai_n if r == "ai" else 0))
                except LLMError as exc:
                    st.error(f"{READERS[r]}: {exc}")
        st.session_state["check_results"] = results
    results = st.session_state.get("check_results") or []
    table = []
    for rep in results:
        new = rep.get("unseen_templates", {})
        table.append({
            "Reader": READERS[rep["reader"]], "Requests": rep["n"],
            "Chart type right": f"{rep['chart_accuracy']:.1%}", "Same visual": f"{rep['same_visual']:.1%}",
            "New wording: chart right": f"{new.get('chart_accuracy', 0):.1%}",
            "New wording: same visual": f"{new.get('same_visual', 0):.1%}",
            "Word-level F1": f"{rep['slots']['micro']['f1']:.3f}", "Exact frame": f"{rep['frame_exact_match']:.1%}",
        })
    if table:
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
        st.download_button("Download results (JSON)",
                           json.dumps([{k: v for k, v in r.items() if k != "misses"} for r in results], indent=2),
                           "ivis_accuracy.json", "application/json")
        for rep in results:
            with st.expander(f"{READERS[rep['reader']]}: {len(rep['misses'])} requests with a different visual"):
                st.dataframe(pd.DataFrame(rep["misses"][:200]), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
st.title("📊 iVIS")
st.caption("Intelligent Visualization Insight Synthesizer · plain-English dashboard requests in, Power BI visuals out")
browser_load()

with st.sidebar:
    has_model = ensure_model()
    has_ai = bool(api_key())
    if "pending_dm" in st.session_state:
        st.session_state["dm_choice"] = st.session_state.pop("pending_dm")
    dm_options = [BUILT_IN] + ([MINE] if st.session_state.get("my_rows") else [])
    if st.session_state.get("dm_choice") not in dm_options:
        st.session_state["dm_choice"] = dm_options[0]
    dm_choice = st.radio("Data model", dm_options, key="dm_choice",
                         help="Edit or add your own in the Data model tab.")
    reader_options = [r for r in READERS if (r != "ai" or has_ai) and (r not in ("hybrid", "bert") or has_model)]
    default = "ai" if (dm_choice == MINE and has_ai) else reader_options[0]
    if st.session_state.get("reader_for") != dm_choice:  # switching data model resets the suggested reader
        st.session_state["reader"] = default
        st.session_state["reader_for"] = dm_choice
    reader = st.radio("Reader", reader_options, key="reader", format_func=lambda r: READERS[r],
                      help="BERT + keyword cues: fast and free. AI (Gemini): reads your data model's field list; "
                           "uses the free AI tier. Keywords only: the original rule-based reader.")
    if not has_model:
        st.caption("The BERT model isn't available, so the keyword reader stands in for it.")
    if not has_ai:
        st.caption("The AI reader appears when a Gemini API key is set in the app's secrets.")
    st.markdown("**Examples**" + (" (built-in data model)" if dm_choice == MINE else ""))
    for ex in EXAMPLES:
        if st.button(ex, width="stretch"):
            st.session_state["request"] = ex

cat = active_catalog(dm_choice)
t_req, t_dm, t_acc = st.tabs(["Request", "Data model", "Accuracy check"])
with t_req:
    request_tab(reader, cat)
with t_dm:
    data_model_tab()
with t_acc:
    accuracy_tab()
