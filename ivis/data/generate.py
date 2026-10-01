"""Synthetic training data for iVIS.

Produces JSONL files where each line is one stakeholder request:

    {"text": "...", "tokens": [...], "tags": ["O", "B-METRIC", ...],
     "chart_type": "bar", "template_id": "bar/3"}

Usage:
    python -m ivis.data.generate --out data --n-train 20000 --n-val 2000 --n-test 2000

The test split also contains requests built from templates that never appear
in train/val (``--holdout-per-chart``), which gives an honest estimate of how
well the model handles phrasings it has not seen.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ivis.catalog import AGGREGATIONS, DIM_BY_KEY, DIMENSIONS, MEASURES, SORT_WORDS
from ivis.data.templates import (
    EXTRA_VALUES,
    FILTER_PREPOSITIONS,
    PRE_FILTERS,
    PREFIXES,
    SUFFIXES,
    TEMPLATES,
    TIME_PHRASES,
    TOPN_PHRASES,
)
from ivis.schema import CHART_TYPES
from ivis.text import detokenize, words

OPTIONAL_PROB = {"filter": 0.45, "time": 0.4}

CATEGORICAL_DIMS = [d for d in DIMENSIONS if not d.is_time]
GROUPABLE_DIMS = [d for d in CATEGORICAL_DIMS if d.key not in ("query_status",)] + [DIM_BY_KEY["query_status"]]
TIME_DIMS = [d for d in DIMENSIONS if d.is_time]
GEO_DIMS = [d for d in DIMENSIONS if d.is_geo]
FILTERABLE_DIMS = [k for k in FILTER_PREPOSITIONS]


def _tagged(phrase: str, label: str | None) -> list[tuple[str, str]]:
    toks = words(phrase)
    if label is None:
        return [(t, "O") for t in toks]
    return [(t, ("B-" if i == 0 else "I-") + label) for i, t in enumerate(toks)]


class _Example:
    """Mutable state while expanding one template."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.used_dims: set[str] = set()
        self.metric = rng.choice(MEASURES)
        self.metric2 = rng.choice([m for m in MEASURES if m.key != self.metric.key])

    def pick_dim(self, pool):
        choices = [d for d in pool if d.key not in self.used_dims]
        d = self.rng.choice(choices)
        self.used_dims.add(d.key)
        return d

    # --- slot expanders --------------------------------------------------
    def metric_phrase(self, with_agg: bool, which: int = 1) -> list[tuple[str, str]]:
        rng = self.rng
        m = self.metric if which == 1 else self.metric2
        out: list[tuple[str, str]] = []
        if with_agg and rng.random() < 0.25:
            agg = "average" if m.default_agg == "average" else rng.choice(["count", "sum", "average", "median", "count"])
            out += _tagged(rng.choice(AGGREGATIONS[agg]), "AGG")
        if which == 1 and rng.random() < 0.35:
            dims = [k for k in PRE_FILTERS.get(m.key, []) if k not in self.used_dims]
            if dims:
                dim = DIM_BY_KEY[rng.choice(dims)]
                self.used_dims.add(dim.key)
                out += _tagged(rng.choice(dim.values), "FILTER")
        out += _tagged(rng.choice(m.synonyms), "METRIC")
        return out

    def filter_phrase(self) -> list[tuple[str, str]]:
        rng = self.rng
        keys = [k for k in FILTERABLE_DIMS if k not in self.used_dims]
        if not keys:
            return []
        key = rng.choice(keys)
        self.used_dims.add(key)
        dim = DIM_BY_KEY[key]
        values = dim.values + EXTRA_VALUES.get(key, [])
        return _tagged(rng.choice(FILTER_PREPOSITIONS[key]), None) + _tagged(rng.choice(values), "FILTER")

    def expand(self, token: str) -> list[tuple[str, str]]:
        rng = self.rng
        name = token[1:-1]
        if name.startswith("?"):
            name = name[1:]
            if rng.random() >= OPTIONAL_PROB.get(name, 0.4):
                return []
        if "=" in name:
            label, literal = name.split("=", 1)
            return _tagged(literal.replace("_", " "), label)
        if name == "metric":
            return self.metric_phrase(with_agg=True)
        if name == "metric_bare":
            return self.metric_phrase(with_agg=False)
        if name == "metric2":
            return _tagged(rng.choice(self.metric2.synonyms), "METRIC")
        if name == "group":
            return _tagged(rng.choice(self.pick_dim(GROUPABLE_DIMS).synonyms), "GROUP_BY")
        if name == "tgroup":
            return _tagged(rng.choice(self.pick_dim(TIME_DIMS).synonyms), "GROUP_BY")
        if name == "geo":
            return _tagged(rng.choice(self.pick_dim(GEO_DIMS).synonyms), "GROUP_BY")
        if name == "series":
            return _tagged(rng.choice(self.pick_dim(CATEGORICAL_DIMS).synonyms), "SERIES")
        if name == "filter":
            return self.filter_phrase()
        if name == "time":
            return _tagged(rng.choice(TIME_PHRASES), "TIME")
        if name == "topn":
            return _tagged(rng.choice(TOPN_PHRASES), "TOPN")
        if name in ("sort", "sort_desc", "sort_asc"):
            direction = {"sort_desc": "desc", "sort_asc": "asc"}.get(name) or rng.choice(["desc", "asc"])
            return _tagged(rng.choice(SORT_WORDS[direction]), "SORT")
        raise ValueError(f"Unknown placeholder {token!r}")


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 5 or not word.isalpha():
        return word
    i = rng.randrange(1, len(word) - 2)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def make_example(rng: random.Random, chart: str, template_idx: int) -> dict:
    template = TEMPLATES[chart][template_idx]
    ex = _Example(rng)
    pairs: list[tuple[str, str]] = []
    # Expand placeholders that constrain dimensions first ({group}, {series}) so
    # filters never reuse the grouping dimension; then lay out in template order.
    pieces = template.split()
    order = sorted(range(len(pieces)), key=lambda i: 0 if pieces[i] in ("{group}", "{tgroup}", "{geo}", "{series}") else 1)
    expanded: dict[int, list[tuple[str, str]]] = {}
    for i in order:
        p = pieces[i]
        expanded[i] = ex.expand(p) if p.startswith("{") and p.endswith("}") else _tagged(p, None)
    for i in range(len(pieces)):
        pairs += expanded[i]

    suffix = _tagged(rng.choice(SUFFIXES), None)
    if pairs and pairs[-1][0] == "?":
        pairs = pairs[:-1] + suffix + pairs[-1:]
    else:
        pairs = pairs + suffix
    pairs = _tagged(rng.choice(PREFIXES), None) + pairs

    # --- light augmentation ----------------------------------------------
    if rng.random() < 0.15 and pairs and pairs[-1][0] == "?":
        pairs = pairs[:-1]
    if rng.random() < 0.05:
        j = rng.randrange(len(pairs))
        pairs[j] = (_typo(pairs[j][0], rng), pairs[j][1])
    tokens = [t for t, _ in pairs]
    if rng.random() < 0.35:
        tokens = [t.lower() for t in tokens]
    elif rng.random() < 0.6 and tokens:
        tokens[0] = tokens[0][:1].upper() + tokens[0][1:]
    tags = [tag for _, tag in pairs]

    text = detokenize(tokens)
    # Guarantee the text re-tokenizes to the same words the tags refer to.
    assert words(text) == tokens, (text, tokens)
    return {
        "text": text,
        "tokens": tokens,
        "tags": tags,
        "chart_type": chart,
        "template_id": f"{chart}/{template_idx}",
    }


def build_splits(n_train: int, n_val: int, n_test: int, seed: int = 13, holdout_per_chart: int = 1):
    rng = random.Random(seed)
    seen: dict[str, list[int]] = {}
    held: dict[str, list[int]] = {}
    for chart in CHART_TYPES:
        idx = list(range(len(TEMPLATES[chart])))
        rng.shuffle(idx)
        k = holdout_per_chart if len(idx) > 4 else 0
        held[chart], seen[chart] = idx[:k], idx[k:]

    def sample(n: int, pool: dict[str, list[int]]) -> list[dict]:
        out = []
        charts = [c for c in CHART_TYPES if pool[c]]
        for i in range(n):
            chart = charts[i % len(charts)]
            out.append(make_example(rng, chart, rng.choice(pool[chart])))
        rng.shuffle(out)
        return out

    train = sample(n_train, seen)
    val = sample(n_val, seen)
    n_unseen = n_test // 2 if any(held.values()) else 0
    test = sample(n_test - n_unseen, seen) + sample(n_unseen, held)
    for ex in test:
        chart, i = ex["template_id"].split("/")
        ex["unseen_template"] = int(i) in held[chart]
    return train, val, test, held


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--holdout-per-chart", type=int, default=1)
    args = ap.parse_args(argv)

    train, val, test, held = build_splits(args.n_train, args.n_val, args.n_test, args.seed, args.holdout_per_chart)
    out = Path(args.out)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    write_jsonl(out / "test.jsonl", test)
    held_ids = sorted(f"{c}/{i}" for c, ids in held.items() for i in ids)
    (out / "heldout_templates.json").write_text(json.dumps(held_ids, indent=2))
    print(f"wrote {len(train)} train / {len(val)} val / {len(test)} test examples to {out}/")
    print(f"held-out templates (test only): {', '.join(held_ids)}")


if __name__ == "__main__":
    main()
