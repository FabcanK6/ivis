"""Evaluate a parser on a labelled JSONL split.

Usage:
    python -m ivis.evaluate --data data/test.jsonl --model models/ivis-bert
    python -m ivis.evaluate --data data/test.jsonl --backend rules          # baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ivis.data.generate import read_jsonl
from ivis.metrics import evaluate_predictions, format_report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--model", default="models/ivis-bert")
    ap.add_argument("--backend", choices=["bert", "rules"], default="bert")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--report", default=None, help="write the full report as JSON here")
    ap.add_argument("--errors", type=int, default=10, help="print this many misclassified examples")
    args = ap.parse_args(argv)

    rows = read_jsonl(args.data)
    if args.backend == "rules":
        from ivis.rules import RuleParser

        parser = RuleParser()
        preds = [parser.predict(r["text"]) for r in rows]
    else:
        from ivis.predict import BertParser

        parser = BertParser(args.model)
        preds = []
        for i in range(0, len(rows), args.batch_size):
            preds += parser.predict_batch([r["text"] for r in rows[i:i + args.batch_size]])

    # Predictions are made from raw text; make sure word boundaries line up with the gold rows.
    for r, p in zip(rows, preds):
        if p["tokens"] != r["tokens"]:
            raise ValueError(f"Tokenization mismatch for: {r['text']!r}")

    report = evaluate_predictions(rows, preds)
    print(f"backend: {args.backend}\n{format_report(report)}")

    shown = 0
    for r, p in zip(rows, preds):
        if shown >= args.errors:
            break
        if r["chart_type"] != p["chart_type"] or r["tags"] != p["tags"]:
            shown += 1
            if shown == 1:
                print("\nsample errors:")
            diff = [f"{t}:{g}->{q}" for t, g, q in zip(r["tokens"], r["tags"], p["tags"]) if g != q]
            print(f"- {r['text']}\n    chart {r['chart_type']} -> {p['chart_type']}; tags {', '.join(diff) or 'ok'}")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps({"backend": args.backend, **report}, indent=2))


if __name__ == "__main__":
    main()
