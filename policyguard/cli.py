"""Command line interface: ``python -m policyguard check|ask``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .checker import check_policy
from .qa import ask


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="policyguard", description="Check and query policy documents.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    check = sub.add_parser("check", help="Check a policy against a framework")
    check.add_argument("policy", type=Path)
    check.add_argument("--framework", default="banking_baseline", help="Framework name or JSON path")
    check.add_argument("--format", choices=["md", "json"], default="md")
    check.add_argument("--fail-under", type=float, default=None, help="Exit 1 if score (0-100) is below this")

    q = sub.add_parser("ask", help="Ask a question about a policy")
    q.add_argument("policy", type=Path)
    q.add_argument("question")

    args = parser.parse_args(argv)
    text = args.policy.read_text(encoding="utf-8")

    if args.cmd == "check":
        report = check_policy(text, args.framework)
        print(json.dumps(report.to_dict(), indent=2) if args.format == "json" else report.to_markdown())
        if args.fail_under is not None and report.score * 100 < args.fail_under:
            return 1
        return 0

    answer = ask(text, args.question)
    print(answer.answer)
    for c in answer.citations:
        print(f"  [source] {c.clause_id} \"{c.heading}\" (line {c.line})")
    print(f"\n{answer.disclaimer}")
    return 0 if answer.confident else 2


if __name__ == "__main__":
    sys.exit(main())
