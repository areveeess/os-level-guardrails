"""
Deliverables CLI — python -m src.deliverables build | blocks | check
====================================================================
  build [--results-root R] [--frozen F] [--reuse-frozen] [--no-docs] [--doc PATH ...]
      Freeze the Phase 4 directories into F (unless --reuse-frozen), regenerate the report,
      charts and numbers.json from the frozen copy, then (unless --no-docs) rewrite the
      generated blocks of the target docs and run the check on them. The last line is
      "BUILD ok" (exit 0) or "FAIL build: reason" (exit 1).
  blocks [--frozen F] [--doc PATH ...]
      Rewrite the generated blocks of the target docs (all present ones without --doc).
  check [--frozen F] [--doc PATH ...]
      The D-03 consistency check; ends with "deliverables-check-ok docs N" (exit 0) or exit 1.

One fact per printed line, ASCII only. --doc must name one of the target docs (README.md,
docs/SDD.md, docs/viva/04-..., docs/viva/05-...), as a path relative to the repository root.
"""

import argparse
import sys

from src.deliverables import FROZEN_DIR, RESULTS_ROOT
from src.deliverables.blocks import DOC_TARGETS
from src.deliverables.build import REPO_ROOT, DeliverablesError, apply_docs, build
from src.deliverables.consistency import load_numbers, run_check, select_docs


def _say(line: str):
    print(line, flush=True)


def _fail(command: str, e: Exception) -> int:
    reason = str(e) if isinstance(e, (DeliverablesError, ValueError)) else f"{type(e).__name__}: {e}"
    _say(f"FAIL {command}: {reason}")
    return 1


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.deliverables",
                                     description="Freeze the Phase 4 results and regenerate the Phase 5 deliverables.")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="freeze (or reuse) the results, regenerate everything, refresh and check docs")
    b.add_argument("--results-root", default=RESULTS_ROOT, help="directory holding the phase4-* directories")
    b.add_argument("--reuse-frozen", action="store_true", help="keep the existing frozen copy; only re-render")
    b.add_argument("--no-docs", action="store_true", help="do not touch or check the target docs")
    others = [sub.add_parser("blocks", help="rewrite the generated blocks of the target docs"),
              sub.add_parser("check", help="check every target doc against numbers.json")]
    for p in [b] + others:
        p.add_argument("--frozen", default=FROZEN_DIR, help="frozen directory (default results/phase5-final)")
        p.add_argument("--doc", action="append", default=None,
                       help=f"a target doc to process (repeatable; default all of {', '.join(DOC_TARGETS)})")
    args = parser.parse_args(argv)

    if args.command == "build":
        docs = None if args.no_docs else (args.doc or list(DOC_TARGETS))
        try:
            build(args.results_root, args.frozen, reuse=args.reuse_frozen, say=_say, docs=docs)
        except (DeliverablesError, OSError, ValueError, KeyError) as e:
            return _fail("build", e)
        return 0
    try:
        docs = select_docs(args.doc)
        numbers = load_numbers(args.frozen)
        if args.command == "blocks":
            for line in apply_docs(REPO_ROOT, args.frozen, docs, numbers):
                _say(line)
            return 0
        lines, code = run_check(REPO_ROOT, args.frozen, docs, numbers=numbers)
    except (DeliverablesError, OSError, ValueError, KeyError) as e:
        return _fail(args.command, e)
    for line in lines:
        _say(line)
    return code


if __name__ == "__main__":
    sys.exit(main())
