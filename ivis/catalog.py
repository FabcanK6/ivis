"""Semantic model for a clinical-operations Power BI dataset.

The model only learns *where* a measure, dimension, or filter is mentioned.
This catalog maps those spans to concrete Power BI fields ("Table[Column]").
Swap these entries for your own dataset's tables and columns; the generator
uses the same synonyms to build training data, so retrain after editing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Measure:
    key: str
    display: str
    field: str
    default_agg: str
    synonyms: list[str]
    unit: str = "count"


@dataclass
class Dimension:
    key: str
    display: str
    field: str
    synonyms: list[str]
    values: list[str] = field(default_factory=list)
    value_pattern: str | None = None  # regex for open-ended values (e.g. site numbers)
    is_time: bool = False
    is_geo: bool = False


MEASURES: list[Measure] = [
    Measure("queries", "Query Count", "Queries[Query Count]", "count",
            ["queries", "data queries", "DM queries", "EDC queries", "query volume"]),
    Measure("query_age", "Query Age (days)", "Queries[Days Open]", "average",
            ["query aging", "query age", "days open", "query turnaround", "time to resolve queries"],
            unit="days"),
    Measure("adverse_events", "Adverse Events", "Safety[AE Count]", "count",
            ["adverse events", "AEs", "AE reports", "safety events"]),
    Measure("saes", "Serious Adverse Events", "Safety[SAE Count]", "count",
            ["SAEs", "serious adverse events", "serious AEs"]),
    Measure("deviations", "Protocol Deviations", "Deviations[Deviation Count]", "count",
            ["protocol deviations", "deviations", "PDs", "major deviations"]),
    Measure("enrolled", "Enrolled Subjects", "Enrollment[Enrolled Subjects]", "count",
            ["enrollment", "enrolled subjects", "enrolled patients", "subjects enrolled", "recruitment"]),
    Measure("screen_failures", "Screen Failures", "Enrollment[Screen Failures]", "count",
            ["screen failures", "screen fails", "screening failures"]),
    Measure("screen_fail_rate", "Screen Failure Rate", "Enrollment[Screen Failure Rate]", "average",
            ["screen failure rate", "screen fail rate"], unit="percent"),
    Measure("randomized", "Randomized Subjects", "Enrollment[Randomized Subjects]", "count",
            ["randomized patients", "randomizations", "randomized subjects"]),
    Measure("missing_pages", "Missing Pages", "CRF[Missing Pages]", "count",
            ["missing pages", "missing CRFs", "missing eCRF pages", "outstanding pages"]),
    Measure("sdv", "SDV Completion %", "Monitoring[SDV Completion %]", "average",
            ["SDV completion", "SDV rate", "source data verification", "SDV percentage"], unit="percent"),
    Measure("visits", "Completed Visits", "Visits[Completed Visits]", "count",
            ["completed visits", "subject visits", "patient visits"]),
    Measure("dropouts", "Early Terminations", "Enrollment[Early Terminations]", "count",
            ["dropouts", "early terminations", "discontinuations", "withdrawals"]),
    Measure("monitoring_visits", "Monitoring Visits", "Monitoring[Visit Count]", "count",
            ["monitoring visits", "SMVs", "site visits", "IMVs"]),
]

DIMENSIONS: list[Dimension] = [
    Dimension("site", "Site", "Site[Site Name]", ["site", "sites", "site number", "investigator site"],
              values=["Site 101", "Site 104", "Site 112", "Site 203", "Site 215", "Site 310", "Site 342"],
              value_pattern=r"^site\s*#?\s*\d+$"),
    Dimension("country", "Country", "Site[Country]", ["country", "countries"],
              values=["Germany", "USA", "United States", "Japan", "Brazil", "Poland", "Spain",
                      "Canada", "India", "UK", "France", "Australia", "Mexico", "South Korea"],
              is_geo=True),
    Dimension("region", "Region", "Site[Region]", ["region", "regions", "geography"],
              values=["EMEA", "APAC", "North America", "LATAM", "Europe", "Asia Pacific"], is_geo=True),
    Dimension("study", "Study", "Study[Protocol Number]", ["study", "studies", "protocol", "protocols", "trial", "trials"],
              values=["ONC-301", "CARD-210", "NEU-114", "IMM-450", "RES-022", "DERM-118"],
              value_pattern=r"^[a-z]{2,5}-\d{2,4}$"),
    Dimension("cra", "CRA", "Monitoring[CRA Name]", ["CRA", "CRAs", "monitor", "monitors", "clinical research associate"]),
    Dimension("investigator", "Principal Investigator", "Site[Principal Investigator]",
              ["PI", "PIs", "investigator", "investigators", "principal investigator"]),
    Dimension("visit", "Visit", "Visits[Visit Name]", ["visit", "visit name", "visit type"],
              values=["Screening", "Baseline", "Week 4", "Week 12", "End of Treatment", "Follow-up"]),
    Dimension("form", "CRF Form", "CRF[Form Name]", ["form", "forms", "CRF page", "eCRF form", "CRF form"],
              values=["Demographics", "Vital Signs", "Concomitant Medications", "Labs", "Medical History"]),
    Dimension("query_status", "Query Status", "Queries[Status]", ["status", "query status"],
              values=["open", "closed", "answered", "overdue", "unresolved", "resolved"]),
    Dimension("severity", "Severity", "Safety[Severity]", ["severity", "grade", "CTCAE grade", "seriousness"],
              values=["severe", "moderate", "mild", "grade 3", "grade 4", "life-threatening"]),
    Dimension("arm", "Treatment Arm", "Randomization[Arm]", ["treatment arm", "arm", "arms", "cohort", "cohorts"],
              values=["placebo", "active arm", "high dose", "low dose", "cohort A", "cohort B"]),
    Dimension("deviation_category", "Deviation Category", "Deviations[Category]",
              ["deviation category", "deviation type", "category", "categories"],
              values=["informed consent", "eligibility", "dosing", "visit window", "IP handling"]),
    Dimension("query_type", "Query Type", "Queries[Query Type]", ["query type", "query source"],
              values=["manual", "auto-generated", "SDV-raised", "medical review"]),
    Dimension("therapeutic_area", "Therapeutic Area", "Study[Therapeutic Area]", ["therapeutic area", "TA", "indication"],
              values=["oncology", "cardiology", "neurology", "immunology", "respiratory", "dermatology"]),
    Dimension("site_status", "Site Status", "Site[Site Status]", ["site status", "activation status"],
              values=["active", "activated", "closed out", "on hold"]),
    # Time grains (used as axes, e.g. "per month")
    Dimension("day", "Date", "Date[Date]", ["day", "days", "daily", "date"], is_time=True),
    Dimension("week", "Week", "Date[Week]", ["week", "weeks", "weekly"], is_time=True),
    Dimension("month", "Month", "Date[Month]", ["month", "months", "monthly"], is_time=True),
    Dimension("quarter", "Quarter", "Date[Quarter]", ["quarter", "quarters", "quarterly"], is_time=True),
    Dimension("year", "Year", "Date[Year]", ["year", "years", "yearly", "annual"], is_time=True),
]

AGGREGATIONS: dict[str, list[str]] = {
    "count": ["number of", "count of", "count", "how many", "#"],
    "sum": ["total", "sum of", "overall"],
    "average": ["average", "avg", "mean"],
    "median": ["median"],
    "max": ["maximum", "max", "peak"],
    "min": ["minimum", "min"],
    "distinct_count": ["distinct", "unique number of", "unique"],
    "percent_of_total": ["percentage of", "percent of", "% of", "share of", "proportion of"],
}

SORT_WORDS: dict[str, list[str]] = {
    "desc": ["highest", "most", "largest", "biggest", "worst", "descending", "top performing", "greatest"],
    "asc": ["lowest", "fewest", "least", "smallest", "best", "ascending", "slowest"],
}

DATE_FIELD = "Date[Date]"


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w%#\- ]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _singular(s: str) -> str:
    if s.endswith("ies") and len(s) > 4:
        return s[:-3] + "y"
    if s.endswith("s") and not s.endswith("ss") and len(s) > 3:
        return s[:-1]
    return s


_MEASURE_INDEX: dict[str, Measure] = {}
for _m in MEASURES:
    for _syn in [_m.display, _m.key, *_m.synonyms]:
        _MEASURE_INDEX[_norm(_syn)] = _m

_DIM_INDEX: dict[str, Dimension] = {}
_VALUE_INDEX: dict[str, Dimension] = {}
for _d in DIMENSIONS:
    for _syn in [_d.display, _d.key, *_d.synonyms]:
        _DIM_INDEX[_norm(_syn)] = _d
    for _v in _d.values:
        _VALUE_INDEX[_norm(_v)] = _d

DIM_BY_KEY = {d.key: d for d in DIMENSIONS}
MEASURE_BY_KEY = {m.key: m for m in MEASURES}


def _fuzzy(text: str, index: dict, cutoff: float = 0.82):
    import difflib

    match = difflib.get_close_matches(text, list(index), n=1, cutoff=cutoff)
    return index[match[0]] if match else None


def resolve_measure(span: str) -> Measure | None:
    n = _norm(span)
    if n in _MEASURE_INDEX:
        return _MEASURE_INDEX[n]
    # strip leading qualifiers the model sometimes includes ("open queries" -> "queries")
    words = n.split()
    for i in range(1, len(words)):
        tail = " ".join(words[i:])
        if tail in _MEASURE_INDEX:
            return _MEASURE_INDEX[tail]
    return _fuzzy(n, _MEASURE_INDEX) or _fuzzy(_singular(n), _MEASURE_INDEX)


def resolve_dimension(span: str) -> Dimension | None:
    n = _norm(span)
    for cand in (n, _singular(n), re.sub(r"^(each|every|per|by)\s+", "", n)):
        if cand in _DIM_INDEX:
            return _DIM_INDEX[cand]
    return _fuzzy(n, _DIM_INDEX)


def resolve_filter_value(span: str, context: list[str] | None = None) -> tuple[Dimension | None, str]:
    """Map a filter value span ("germany", "site 104") to (dimension, canonical value).

    ``context`` holds a few words that preceded the span ("for the study ..."),
    which helps place values that are not in the catalog.
    """
    n = _norm(span)
    if n in _VALUE_INDEX:
        dim = _VALUE_INDEX[n]
        canonical = next(v for v in dim.values if _norm(v) == n)
        return dim, canonical
    for dim in DIMENSIONS:
        if dim.value_pattern and re.match(dim.value_pattern, n):
            return dim, span.strip().title() if dim.key == "site" else span.strip().upper()
    # "<dimension word> <value>", e.g. "CRA Smith" or "arm C"
    words = n.split()
    for i in range(len(words) - 1, 0, -1):
        head = " ".join(words[:i])
        if head in _DIM_INDEX:
            return _DIM_INDEX[head], " ".join(span.split()[i:])
    if context:
        for w in reversed(context):
            d = _DIM_INDEX.get(_norm(w)) or _DIM_INDEX.get(_singular(_norm(w)))
            if d:
                return d, span.strip()
    fuzzy = _fuzzy(n, _VALUE_INDEX, cutoff=0.85)
    if fuzzy:
        return fuzzy, span.strip()
    return None, span.strip()
