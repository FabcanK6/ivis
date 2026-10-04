"""Evaluate a parser on a labelled JSONL split.

Usage:
    python -m ivis.evaluate --data data/test.jsonl --model models/ivis-bert
    python -m ivis.evaluate --data data/test.jsonl --backend rules          # baseline
    python -m ivis.evaluate --data data/test.jsonl --backend hybrid         # BERT + strong keyword cues
    GEMINI_API_KEY=... python -m ivis.evaluate --data data/test.jsonl --backend ai --sample 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ivis.benchmark import sample, score
from ivis.data.generate import read_jsonl
from ivis.metrics import format_report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--model", default="models/ivis-bert")
    ap.add_argument("--backend", choices=["bert", "hybrid", "rules", "ai"], default="bert")
    ap.add_argument("--sample", type=int, default=0, help="score a fixed sample of this many requests (half unseen)")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between AI requests (free tier limits)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--report", default=None, help="write the full report as JSON here")
    ap.add_argument("--errors", type=int, default=10, help="print this many misclassified examples")
    args = ap.parse_args(argv)

    rows = read_jsonl(args.data)
    if args.sample:
        rows = sample(rows, args.sample)
    if args.backend == "rules":
        from ivis.rules import RuleParser

        parser = RuleParser()
        preds = [parser.predict(r["text"]) for r in rows]
    elif args.backend == "ai":
        import time

        from ivis.ai import AIParser
        from ivis.llm import GeminiClient, get_api_key

        parser = AIParser(GeminiClient(get_api_key() or ""))
        preds = []
        for r in rows:
            preds.append(parser.predict(r["text"]))
            time.sleep(args.sleep)
    else:
        from ivis.predict import BertParser, HybridParser

        parser = BertParser(args.model)
        if args.backend == "hybrid":
            parser = HybridParser(parser)
        preds = []
        for i in range(0, len(rows), args.batch_size):
            preds += parser.predict_batch([r["text"] for r in rows[i:i + args.batch_size]])

    # Predictions are made from raw text; make sure word boundaries line up with the gold rows.
    for r, p in zip(rows, preds):
        if p["tokens"] != r["tokens"]:
            raise ValueError(f"Tokenization mismatch for: {r['text']!r}")

    report = score(rows, preds)
    print(f"backend: {args.backend}\n{format_report(report)}\nsame visual        : {report['same_visual']:.3f}")

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
        report.pop("misses", None)
        Path(args.report).write_text(json.dumps({"backend": args.backend, **report}, indent=2))


if __name__ == "__main__":
    main()
