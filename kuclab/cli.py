from __future__ import annotations

import argparse
import json
import sys

from .infer import SafetyModel

BANNER = "Censer .1 - napis text, Enter. 'exit' pro konec."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kuclab-cneser",
        description="Censer .1: tiny English safety classifier. Prints 1 (unsafe) or 0 (safe).",
    )
    parser.add_argument("text", nargs="*", help="text to classify; omit to read stdin or use --repl")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None, help="cuda / cpu")
    parser.add_argument("--json", action="store_true", help="print JSON with score instead of bare 0/1")
    parser.add_argument("--score", action="store_true", help="print label and score")
    parser.add_argument("--repl", action="store_true", help="interactive mode")
    return parser


def run_repl(model: SafetyModel) -> int:
    print(BANNER)
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line.strip().lower() in {"exit", "quit", ":q"}:
            return 0
        if not line.strip():
            continue
        verdict = model.predict(line)
        print(verdict.label)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    text = " ".join(args.text).strip()
    if not text and not args.repl and not sys.stdin.isatty():
        text = sys.stdin.read().strip()

    model = SafetyModel(checkpoint=args.checkpoint, threshold=args.threshold, device=args.device)

    if args.repl or (not text and sys.stdin.isatty()):
        return run_repl(model)

    if not text:
        parser.error("no input text provided")

    verdict = model.predict(text)
    if args.json:
        print(json.dumps(verdict.as_dict()))
    elif args.score:
        print(f"{verdict.label} {verdict.probability:.4f}")
    else:
        print(verdict.label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
