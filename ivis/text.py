"""Word tokenization and BIO helpers.

Training data and inference both go through :func:`words`, so the word
boundaries the model sees at inference match the ones it was trained on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'/.][A-Za-z0-9]+)*%?|[^\sA-Za-z0-9]")


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def detokenize(tokens: list[str]) -> str:
    out = " ".join(tokens)
    out = re.sub(r"\s+([?.,!;:)])", r"\1", out)
    out = re.sub(r"([(#])\s+", r"\1", out)
    return out


@dataclass
class Span:
    label: str
    start: int  # word index, inclusive
    end: int    # word index, exclusive
    text: str
    field: str | None = None  # a Power BI field chosen by the AI reader (checked against the catalog later)

    def as_tuple(self) -> tuple[str, int, int]:
        return (self.label, self.start, self.end)


def bio_to_spans(tokens: list[str], tags: list[str]) -> list[Span]:
    """Decode BIO tags into spans. A stray ``I-X`` is treated as the start of a span."""
    spans: list[Span] = []
    cur_label, cur_start = None, None
    for i, tag in enumerate([*tags, "O"]):
        prefix, _, label = tag.partition("-")
        continues = prefix == "I" and label == cur_label
        if cur_label is not None and not continues:
            spans.append(Span(cur_label, cur_start, i, " ".join(tokens[cur_start:i])))
            cur_label, cur_start = None, None
        if prefix in ("B", "I") and not continues:
            cur_label, cur_start = label, i
    return spans


def spans_to_bio(n: int, spans: list[tuple[str, int, int]]) -> list[str]:
    tags = ["O"] * n
    for label, s, e in spans:
        tags[s] = f"B-{label}"
        for j in range(s + 1, e):
            tags[j] = f"I-{label}"
    return tags
