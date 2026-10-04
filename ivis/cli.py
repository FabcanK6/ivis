"""Command-line interface.

    python -m ivis.cli "top 10 sites by open queries in Germany last 30 days"
    python -m ivis.cli --format devops "trend of SAEs by month for ONC-301"
    echo "how many screen failures at Site 104?" | python -m ivis.cli -
"""

from __future__ import annotations

import argparse
import json
import sys

from ivis.devops import to_json_patch, to_markdown
from ivis.predict import load_parser


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="+", help="request text, or '-' to read lines from stdin")
    ap.add_argument("--model", default="models/ivis-bert", help="checkpoint dir (falls back to rules if missing)")
    ap.add_argument("--backend", choices=["auto", "bert", "rules", "ai"], default="auto",
                    help="auto = BERT + keyword cues (keywords only if there is no model); ai needs GEMINI_API_KEY")
    ap.add_argument("--format", choices=["spec", "devops", "devops-json", "markdown", "visual"], default="spec",
                    help="visual = a Power BI (PBIR) visual.json")
    args = ap.parse_args(argv)

    if args.backend == "ai":
        from ivis.ai import AIParser
        from ivis.llm import GeminiClient, get_api_key

        parser = AIParser(GeminiClient(get_api_key() or ""))
    else:
        parser = load_parser(None if args.backend == "rules" else args.model, hybrid=args.backend == "auto")
    print(f"[iVIS backend: {parser.name}]", file=sys.stderr)
    texts = [line.strip() for line in sys.stdin if line.strip()] if args.text == ["-"] else [" ".join(args.text)]
    for text in texts:
        spec = parser.parse(text)
        if args.format == "spec":
            print(json.dumps(spec, indent=2, ensure_ascii=False))
        elif args.format == "visual":
            from ivis.pbir import to_visual_json

            visual, notes = to_visual_json(spec)
            print(json.dumps(visual, indent=2, ensure_ascii=False))
            for n in notes:
                print(f"note: {n}", file=sys.stderr)
        elif args.format == "devops-json":
            print(json.dumps(to_json_patch(spec), indent=2, ensure_ascii=False))
        else:
            print(to_markdown(spec))


if __name__ == "__main__":
    main()
