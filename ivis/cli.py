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
    ap.add_argument("--backend", choices=["auto", "rules"], default="auto")
    ap.add_argument("--format", choices=["spec", "devops", "devops-json", "markdown"], default="spec")
    args = ap.parse_args(argv)

    parser = load_parser(None if args.backend == "rules" else args.model)
    print(f"[iVIS backend: {parser.name}]", file=sys.stderr)
    texts = [line.strip() for line in sys.stdin if line.strip()] if args.text == ["-"] else [" ".join(args.text)]
    for text in texts:
        spec = parser.parse(text)
        if args.format == "spec":
            print(json.dumps(spec, indent=2, ensure_ascii=False))
        elif args.format == "devops-json":
            print(json.dumps(to_json_patch(spec), indent=2, ensure_ascii=False))
        else:
            print(to_markdown(spec))


if __name__ == "__main__":
    main()
