"""Semantic model for a clinical-operations Power BI dataset.

The model only learns *where* a measure, dimension, or filter is mentioned.
This catalog maps those spans to concrete Power BI fields ("Table[Column]").
The built-in catalog below is a clinical-operations model. Users can bring their own data model in the
app (a table of measures and columns, see :class:`Catalog` and :func:`Catalog.from_rows`); the spec builder, the
keyword parser and the AI reader all work against whichever catalog is active. The BERT model was trained on the
built-in catalog's synonyms, so it finds the parts of a request best when the wording is similar.
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

DATE_FIELD = "Date[Date]"  # the built-in catalog's date column (relative date filters)


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------
FIELD_RE = re.compile(r"^\s*'?([^\[\]']+?)'?\s*\[\s*([^\[\]]+?)\s*\]\s*$")
TIME_WORDS = {"day": ["daily"], "date": ["daily"], "week": ["weekly"], "month": ["monthly"], "quarter": ["quarterly"],
              "year": ["yearly", "annual"]}
_STEM_SUFFIXES = (" name", " count", " number", " id", " code", " #", " amount", " total", " value")


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


def _plural(s: str) -> str:
    if s.endswith("y") and len(s) > 2 and s[-2] not in "aeiou":
        return s[:-1] + "ies"
    return s if s.endswith("s") else s + "s"


def split_field(field: str) -> tuple[str, str] | None:
    """``"Site[Site Name]"`` -> ``("Site", "Site Name")``; None if it is not in Table[Column] form."""
    m = FIELD_RE.match(field or "")
    return (m.group(1).strip(), m.group(2).strip()) if m else None


def auto_synonyms(name: str, is_time: bool = False) -> list[str]:
    """Words people are likely to use for a field, from its name: "Site Name" -> site name, site names, site, sites."""
    n = _norm(name)
    out = [n, _plural(n)]
    for suf in _STEM_SUFFIXES:
        if n.endswith(suf) and len(n) > len(suf) + 1:
            stem = n[: -len(suf)].strip()
            out += [stem, _plural(_singular(stem))]
    if is_time:
        out += TIME_WORDS.get(_singular(n), [])
    return [x for x in dict.fromkeys(out) if x]


class Catalog:
    """A data model iVIS maps requests onto: measures, dimensions (with example values) and a date column."""

    def __init__(self, measures: list[Measure], dimensions: list[Dimension],
                 name: str = "Clinical operations (built-in)", date_field: str | None = None):
        self.name = name
        self.measures = list(measures)
        self.dimensions = list(dimensions)
        times = [d for d in self.dimensions if d.is_time]
        self.date_field = date_field or next((d.field for d in times if _norm(d.display) in ("date", "day")),
                                             times[0].field if times else None)
        self.measure_index: dict[str, Measure] = {}
        for m in self.measures:
            for syn in [m.display, m.key, *m.synonyms]:
                self.measure_index[_norm(syn)] = m
        self.dim_index: dict[str, Dimension] = {}
        self.value_index: dict[str, Dimension] = {}
        for d in self.dimensions:
            for syn in [d.display, d.key, *d.synonyms]:
                self.dim_index[_norm(syn)] = d
            for v in d.values:
                self.value_index[_norm(v)] = d
        self.by_field = {x.field: x for x in [*self.measures, *self.dimensions]}

    # -- lookups ----------------------------------------------------------
    def measure_by_field(self, field: str | None) -> Measure | None:
        x = self.by_field.get(field or "")
        return x if isinstance(x, Measure) else None

    def dimension_by_field(self, field: str | None) -> Dimension | None:
        x = self.by_field.get(field or "")
        return x if isinstance(x, Dimension) else None

    def resolve_measure(self, span: str) -> Measure | None:
        n = _norm(span)
        if n in self.measure_index:
            return self.measure_index[n]
        # strip leading qualifiers the model sometimes includes ("open queries" -> "queries")
        words = n.split()
        for i in range(1, len(words)):
            tail = " ".join(words[i:])
            if tail in self.measure_index:
                return self.measure_index[tail]
        return _fuzzy(n, self.measure_index) or _fuzzy(_singular(n), self.measure_index)

    def resolve_dimension(self, span: str) -> Dimension | None:
        n = _norm(span)
        for cand in (n, _singular(n), re.sub(r"^(each|every|per|by)\s+", "", n)):
            if cand in self.dim_index:
                return self.dim_index[cand]
        return _fuzzy(n, self.dim_index)

    def canonical_value(self, dim: Dimension, span: str) -> str:
        n = _norm(span)
        return next((v for v in dim.values if _norm(v) == n), span.strip())

    def resolve_filter_value(self, span: str, context: list[str] | None = None) -> tuple[Dimension | None, str]:
        """Map a filter value span ("germany", "site 104") to (dimension, canonical value).

        ``context`` holds a few words that preceded the span ("for the study ..."),
        which helps place values that are not in the catalog.
        """
        n = _norm(span)
        if n in self.value_index:
            dim = self.value_index[n]
            return dim, self.canonical_value(dim, span)
        for dim in self.dimensions:
            if dim.value_pattern and re.match(dim.value_pattern, n):
                return dim, span.strip().title() if dim.key == "site" else span.strip().upper()
        # "<dimension word> <value>", e.g. "CRA Smith" or "arm C"
        words = n.split()
        for i in range(len(words) - 1, 0, -1):
            head = " ".join(words[:i])
            if head in self.dim_index:
                return self.dim_index[head], " ".join(span.split()[i:])
        if context:
            for w in reversed(context):
                d = self.dim_index.get(_norm(w)) or self.dim_index.get(_singular(_norm(w)))
                if d:
                    return d, span.strip()
        fuzzy = _fuzzy(n, self.value_index, cutoff=0.85)
        if fuzzy:
            return fuzzy, span.strip()
        return None, span.strip()

    def default_time_axis(self) -> Dimension | None:
        """The axis a trend chart gets when the request names none: Month if there is one, else the first date
        column."""
        times = [d for d in self.dimensions if d.is_time]
        return next((d for d in times if "month" in _norm(d.display)), times[0] if times else None)

    def geo_dimensions(self) -> list[Dimension]:
        return [d for d in self.dimensions if d.is_geo]

    # -- import / export --------------------------------------------------
    ROW_KINDS = ("Measure", "Column", "Date column", "Location column")
    AGG_CHOICES = ("count", "sum", "average", "median", "max", "min", "distinct_count")

    def to_rows(self) -> list[dict]:
        rows = []
        for m in self.measures:
            rows.append({"Kind": "Measure", "Field": m.field, "Also called": ", ".join(m.synonyms),
                         "Values": "", "Default aggregation": m.default_agg})
        for d in self.dimensions:
            kind = "Date column" if d.is_time else "Location column" if d.is_geo else "Column"
            rows.append({"Kind": kind, "Field": d.field, "Also called": ", ".join(d.synonyms),
                         "Values": ", ".join(d.values), "Default aggregation": ""})
        return rows

    @classmethod
    def from_rows(cls, rows: list[dict], name: str = "My data model") -> tuple[Catalog, list[str]]:
        """Build a catalog from table rows (Kind, Field, Also called, Values, Default aggregation).

        Returns the catalog and a list of problems (rows skipped and why). Field names must be ``Table[Column]``, as
        they appear in Power BI; synonyms are added automatically from each field's name."""
        measures: list[Measure] = []
        dims: list[Dimension] = []
        problems: list[str] = []
        seen: set[str] = set()
        for i, r in enumerate(rows, 1):
            field = str(r.get("Field") or "").strip()
            if not field:
                continue
            parts = split_field(field)
            if not parts:
                problems.append(f"Row {i}: '{field}' is not in Table[Column] form, e.g. Sales[Revenue].")
                continue
            table, col = parts
            field = f"{table}[{col}]"
            if field in seen:
                problems.append(f"Row {i}: {field} is listed twice; the first one is kept.")
                continue
            seen.add(field)
            kind = str(r.get("Kind") or "Column").strip().lower()
            also = [a.strip() for a in re.split(r"[,;\n]", str(r.get("Also called") or "")) if a.strip()]
            values = [v.strip() for v in re.split(r"[,;\n]", str(r.get("Values") or "")) if v.strip()]
            key = re.sub(r"\W+", "_", field.lower()).strip("_")
            if kind.startswith("measure"):
                agg = str(r.get("Default aggregation") or "").strip().lower() or "sum"
                if agg not in cls.AGG_CHOICES:
                    problems.append(f"Row {i}: unknown aggregation '{agg}' for {field}; using sum.")
                    agg = "sum"
                syn = list(dict.fromkeys([*also, *auto_synonyms(col)]))
                unit = "percent" if re.search(r"%|percent|rate", col, re.IGNORECASE) else "count"
                measures.append(Measure(key, col, field, agg, syn, unit=unit))
            else:
                is_time = kind.startswith("date")
                syn = list(dict.fromkeys([*also, *auto_synonyms(col, is_time)]))
                dims.append(Dimension(key, col, field, syn, values=values, is_time=is_time,
                                      is_geo=kind.startswith("location")))
        if not measures:
            problems.append("Add at least one measure (something to count or add up).")
        return cls(measures, dims, name=name), problems

    def fingerprint(self) -> str:
        import hashlib
        import json

        blob = json.dumps([self.name, self.to_rows()], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def prompt_text(self) -> str:
        """The catalog as the AI reader sees it."""
        lines = ["Measures (things to count or add up):"]
        for m in self.measures:
            lines.append(f"- {m.field}: also called {', '.join(m.synonyms[:8])}; default aggregation {m.default_agg}")
        lines.append("Columns (to group by, split by or filter on):")
        for d in self.dimensions:
            tag = " [date]" if d.is_time else " [location]" if d.is_geo else ""
            vals = f"; example values: {', '.join(d.values[:12])}" if d.values else ""
            lines.append(f"- {d.field}{tag}: also called {', '.join(d.synonyms[:8])}{vals}")
        return "\n".join(lines)


def _fuzzy(text: str, index: dict, cutoff: float = 0.82):
    import difflib

    match = difflib.get_close_matches(text, list(index), n=1, cutoff=cutoff)
    return index[match[0]] if match else None


DEFAULT = Catalog(MEASURES, DIMENSIONS, date_field=DATE_FIELD)
DIM_BY_KEY = {d.key: d for d in DIMENSIONS}
MEASURE_BY_KEY = {m.key: m for m in MEASURES}


def resolve_measure(span: str) -> Measure | None:
    return DEFAULT.resolve_measure(span)


def resolve_dimension(span: str) -> Dimension | None:
    return DEFAULT.resolve_dimension(span)


def resolve_filter_value(span: str, context: list[str] | None = None) -> tuple[Dimension | None, str]:
    return DEFAULT.resolve_filter_value(span, context)
