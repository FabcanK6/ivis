"""Request templates, grouped by the chart type they imply.

Template syntax (space-separated tokens):
    {metric}       METRIC span, optionally preceded by an AGG span and a FILTER adjective ("open")
    {metric_bare}  METRIC span with an optional FILTER adjective, no AGG
    {metric2}      a second, different METRIC span (scatter plots)
    {group}        GROUP_BY span (categorical dimension)
    {tgroup}       GROUP_BY span (time grain: month, week, ...)
    {geo}          GROUP_BY span (country / region)
    {series}       SERIES span (legend / columns), different from {group}
    {filter}       a preposition plus a FILTER value ("in Germany", "at Site 104")
    {time}         TIME span ("last 30 days", "in 2025")
    {topn}         TOPN span ("top 10")
    {sort}         SORT span, either direction; {sort_desc} / {sort_asc} force one
    {LABEL=a_b}    literal words "a b" tagged as LABEL (e.g. {AGG=how_many})
    {?x}           optional placeholder x (included with a per-slot probability)
Anything else is a literal word tagged O.

The phrasing is deliberately varied and often indirect, so the chart type has
to be inferred from cues ("trend", "share", "list", "how many", "on a map")
rather than only from an explicit chart name.
"""

TEMPLATES: dict[str, list[str]] = {
    "bar": [
        "can we get a chart of {metric} by {group} {?filter} {?time} ?",
        "show {metric} by {group} {?filter} {?time}",
        "compare {metric} across {group} {?filter} {?time}",
        "which {group} has the {sort} {metric_bare} {?filter} {?time} ?",
        "{topn} {group} with the {sort_desc} {metric_bare} {?filter} {?time}",
        "bar chart of {metric} per {group} {?filter} {?time}",
        "I need {metric} broken out by {group} {?filter} {?time}",
        "rank {group} by {metric} {?filter} {?time}",
        "how do {group} compare on {metric} {?filter} {?time} ?",
        "{metric} for each {group} {?filter} {?time} , sorted {sort}",
        "can you visualize {metric} by {group} {?filter} {?time} for the ops review",
        "give me the {topn} {group} by {metric} {?filter} {?time}",
        "column chart showing {metric} by {group} {?filter} {?time}",
        "where are we seeing the {sort_desc} {metric_bare} by {group} {?filter} {?time} ?",
    ],
    "stacked_bar": [
        "{metric} by {group} broken down by {series} {?filter} {?time}",
        "stacked bar of {metric} by {group} and {series} {?filter} {?time}",
        "show {metric} per {group} split by {series} {?filter} {?time}",
        "can we see {metric} by {group} with {series} as the legend {?filter} {?time} ?",
        "{metric} across {group} , stacked by {series} {?filter} {?time}",
        "composition of {metric} by {series} for each {group} {?filter} {?time}",
        "I want {metric} for every {group} segmented by {series} {?filter} {?time}",
        "break down {metric} by {series} within each {group} {?filter} {?time}",
    ],
    "line": [
        "trend of {metric} over time {?filter} {?time}",
        "how has {metric_bare} changed {time} {?filter} ?",
        "show {metric} by {tgroup} {?filter} {?time}",
        "{metric} per {tgroup} {?filter} {?time}",
        "line chart of {metric} by {tgroup} {?filter} {?time}",
        "trend in {metric} by {tgroup} for each {series} {?filter} {?time}",
        "is {metric_bare} going up or down {?filter} {time} ?",
        "{tgroup} trend of {metric} {?filter} {?time}",
        "plot {metric} over the {time} {?filter}",
        "track {metric} {tgroup} by {series} {?filter} {?time}",
        "can we monitor {metric} week over week {?filter} {?time} ?",
    ],
    "area": [
        "cumulative {metric} over time {?filter} {?time}",
        "area chart of {metric} by {tgroup} {?filter} {?time}",
        "running total of {metric_bare} by {tgroup} {?filter} {?time}",
        "cumulative {metric_bare} by {tgroup} split by {series} {?filter} {?time}",
        "show the build-up of {metric_bare} per {tgroup} {?filter} {?time}",
        "how is cumulative {metric_bare} tracking against plan {?filter} {?time} ?",
    ],
    "pie": [
        "share of {metric_bare} by {group} {?filter} {?time}",
        "what percentage of {metric_bare} comes from each {group} {?filter} {?time} ?",
        "pie chart of {metric} by {group} {?filter} {?time}",
        "proportion of {metric_bare} by {group} {?filter} {?time}",
        "distribution of {metric_bare} across {group} as percentages {?filter} {?time}",
        "how are {metric_bare} split between {group} {?filter} {?time} ?",
        "{group} contribution to total {metric_bare} {?filter} {?time}",
    ],
    "donut": [
        "donut chart of {metric} by {group} {?filter} {?time}",
        "doughnut showing {metric_bare} share by {group} {?filter} {?time}",
        "ring chart of {metric} per {group} {?filter} {?time}",
        "donut of {metric_bare} split by {group} {?filter} {?time}",
    ],
    "table": [
        "list of {group} with {metric} {?filter} {?time}",
        "give me a table of {metric} by {group} {?filter} {?time}",
        "can I get a listing of {group} and their {metric_bare} {?filter} {?time} ?",
        "table showing {group} , {metric} and {metric2} {?filter} {?time}",
        "export-ready list of {topn} {group} by {metric} {?filter} {?time}",
        "detail table of {metric} for each {group} {?filter} {?time}",
        "which {group} have {metric_bare} {?filter} {?time} ? list them",
        "show me all {group} with outstanding {metric_bare} {?filter} {?time} in a table",
    ],
    "matrix": [
        "matrix of {metric} by {group} and {series} {?filter} {?time}",
        "crosstab of {metric} with {group} as rows and {series} as columns {?filter} {?time}",
        "heatmap of {metric} by {group} vs {series} {?filter} {?time}",
        "pivot {metric} by {group} across {series} {?filter} {?time}",
        "grid of {metric} for {group} by {series} {?filter} {?time}",
        "{metric} in a matrix , rows {group} , columns {series} {?filter} {?time}",
    ],
    "card": [
        "{AGG=how_many} {metric_bare} do we have {?filter} {?time} ?",
        "{AGG=total} {metric_bare} {?filter} {?time}",
        "what is the {AGG=total} number of {metric_bare} {?filter} {?time} ?",
        "kpi card for {metric} {?filter} {?time}",
        "single number showing {metric} {?filter} {?time}",
        "what's our {AGG=average} {metric_bare} {?filter} {?time} ?",
        "headline figure for {metric} {?filter} {?time}",
        "{AGG=how_many} {metric_bare} {?filter} {time} ?",
        "current {metric} {filter} {?time} as a card",
    ],
    "scatter": [
        "{metric} vs {metric2} by {group} {?filter} {?time}",
        "scatter plot of {metric} against {metric2} for each {group} {?filter} {?time}",
        "is there a correlation between {metric_bare} and {metric2} across {group} {?filter} {?time} ?",
        "plot {metric} versus {metric2} per {group} {?filter} {?time}",
        "relationship between {metric_bare} and {metric2} by {group} {?filter} {?time}",
    ],
    "map": [
        "map of {metric} by {geo} {?filter} {?time}",
        "show {metric} by {geo} on a map {?filter} {?time}",
        "geographic view of {metric} per {geo} {?filter} {?time}",
        "where in the world are {metric_bare} highest by {geo} {?filter} {?time} ?",
        "{metric} across {geo} as a map {?filter} {?time}",
        "plot {metric_bare} by {geo} geographically {?filter} {?time}",
    ],
}

PREFIXES = [
    "", "", "", "", "hi team ,", "quick ask :", "for the steering committee ,", "hey ,",
    "for tomorrow's study meeting ,", "when you get a chance ,", "the sponsor asked :",
]

SUFFIXES = [
    "", "", "", "", "please", "thanks", "for the dashboard", "for the weekly ops review",
    "asap", "for the CTMS report",
]

TIME_PHRASES = [
    "last 30 days", "past 30 days", "last 7 days", "last 90 days", "past 6 months", "last 12 months",
    "this month", "last month", "this quarter", "last quarter", "this year", "last year",
    "year to date", "YTD", "in 2025", "in 2026", "since January", "since March", "Q1 2026",
    "Q3 2025", "last week", "this week", "past two weeks", "since study start", "since FPI",
]

TOPN_PHRASES = ["top 5", "top 10", "top 3", "top 20", "top five", "top ten", "bottom 5", "bottom 10"]

FILTER_PREPOSITIONS = {
    "site": ["at", "for"],
    "country": ["in", "for"],
    "region": ["in", "for", "across"],
    "study": ["for", "in", "on study", "for protocol"],
    "arm": ["in the", "for the"],
    "cra": ["for CRA", "assigned to CRA", "for monitor"],
    "therapeutic_area": ["in", "for", "within"],
    "visit": ["at", "for"],
    "form": ["on the", "for the"],
    "deviation_category": ["related to", "for"],
}

# Values for dimensions whose catalog list is empty (open-ended)
EXTRA_VALUES = {
    "cra": ["Smith", "Garcia", "Chen", "Okafor", "Novak", "Patel", "Kowalski", "Haddad"],
    "site": ["Site 118", "Site 225", "Site 401", "Site 509", "site 77"],
    "study": ["ONC-512", "CV-088", "HEM-221"],
}

# Adjectival filters that read naturally right before a metric ("open queries")
PRE_FILTERS = {
    "queries": ["query_status"],
    "query_age": ["query_status"],
    "adverse_events": ["severity"],
    "saes": [],
    "deviations": ["deviation_category"],
}
