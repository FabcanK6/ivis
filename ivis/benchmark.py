"""Accuracy check: run any reader on the standard test split and compare it with the labels.

The test split is rebuilt from the generator with the same seed and sizes as the published results (20,000 train,
2,000 validation, 2,000 test, seed 13), so the app and the command line measure the same 2,000 requests. Half use
phrasings the BERT model saw in training; the other half come from 10 held-out templates it never saw.

Besides the word-level scores (chart type, slot F1, exact frame match), every reader gets a **same visual** score:
the spec built from its reading is compared with the spec built from the labels (chart type, measures, axis,
legend, filters, time window, top N). Two readers that mark slightly different words but produce the same Power BI
visual both count as right, which makes the AI reader comparable with BERT.
"""

from __future__ import annotations

import random

from ivis import catalog
from ivis.data.generate import build_splits
from ivis.metrics import evaluate_predictions
from ivis.spec import build_spec
from ivis.text import bio_to_spans

_SPLIT: list[dict] | None = None


def test_split() -> list[dict]:
    global _SPLIT
    if _SPLIT is None:
        _SPLIT = build_splits(20000, 2000, 2000, seed=13, holdout_per_chart=1)[2]
    return _SPLIT


def sample(rows: list[dict], n: int, seed: int = 7) -> list[dict]:
    """A fixed sample with seen and unseen phrasings in equal parts (for slow readers such as the AI)."""
    rng = random.Random(seed)
    seen = [r for r in rows if not r.get("unseen_template")]
    unseen = [r for r in rows if r.get("unseen_template")]
    return rng.sample(seen, min(n // 2, len(seen))) + rng.sample(unseen, min(n - n // 2, len(unseen)))


PARTS = ("chart", "measures", "axis", "legend", "filters", "time window", "top N")


def visual_key(spec: dict) -> tuple:
    t = spec.get("time_filter") or {}
    return (
        spec["chart_type"],
        tuple(sorted(m["field"] or m["name"] for m in spec["measures"])),
        tuple(g["field"] or g["name"] for g in spec["group_by"]),
        tuple(s["field"] or s["name"] for s in spec["series"]),
        tuple(sorted((f["field"] or "?", tuple(sorted(v.lower() for v in f["values"]))) for f in spec["filters"])),
        tuple(sorted((k, str(v)) for k, v in t.items() if k not in ("text", "field"))),
        (spec["top_n"] or {}).get("n"),
    )


def gold_spec(row: dict, cat: catalog.Catalog | None = None) -> dict:
    return build_spec(row["text"], row["tokens"], row["chart_type"], bio_to_spans(row["tokens"], row["tags"]), cat=cat)


def pred_spec(row: dict, pred: dict, cat: catalog.Catalog | None = None) -> dict:
    spans = pred.get("spans") or bio_to_spans(pred["tokens"], pred["tags"])
    return build_spec(row["text"], pred["tokens"], pred["chart_type"], spans, cat=cat)


def _show(value) -> str:
    if value in ((), None, ""):
        return "none"
    if isinstance(value, tuple):
        return ", ".join(_show(v) for v in value)
    return str(value)


def differences(gold: tuple, got: tuple) -> str:
    """Which parts of two visuals differ, e.g. "axis: Site[Site Name] → Site[Country]"."""
    return "; ".join(f"{name}: {_show(a)} → {_show(b)}" for name, a, b in zip(PARTS, gold, got) if a != b)


def score(rows: list[dict], preds: list[dict]) -> dict:
    report = evaluate_predictions(rows, preds)
    keys = [(visual_key(gold_spec(r)), visual_key(pred_spec(r, p))) for r, p in zip(rows, preds)]
    same = [g == k for g, k in keys]
    report["same_visual"] = sum(same) / len(same) if same else 0.0
    for flag, name in ((False, "seen_templates"), (True, "unseen_templates")):
        idx = [i for i, r in enumerate(rows) if bool(r.get("unseen_template")) == flag]
        if idx and name in report:
            report[name]["same_visual"] = sum(same[i] for i in idx) / len(idx)
    report["misses"] = [
        {"request": r["text"], "what differs (expected → got)": differences(*k),
         "new wording": bool(r.get("unseen_template"))}
        for r, k, ok in zip(rows, keys, same) if not ok
    ]
    report["differs_by_part"] = {
        name: sum(1 for (g, k), ok in zip(keys, same) if not ok and g[i] != k[i]) for i, name in enumerate(PARTS)}
    return report
