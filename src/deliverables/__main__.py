"""
Deliverables CLI — python -m src.deliverables build | blocks | check | deck
===========================================================================
  build [--results-root R] [--frozen F] [--reuse-frozen] [--no-docs] [--doc PATH ...] [--no-deck] [--deck-out P]
        [--allow-blank-template]
      Freeze the Phase 4 directories into F (unless --reuse-frozen), regenerate the report,
      charts and numbers.json from the frozen copy, then (unless --no-docs) rewrite the
      generated blocks of the target docs and run the check on them, then (unless --no-deck)
      render and check the Review-3 deck (printing TEMPLATE, how the template was used; a deck drawn on the
      blank fallback fails unless --allow-blank-template). The last line is "BUILD ok" (exit 0), "FAIL build:
      reason" (exit 1), or "FAIL deck: python-pptx is not importable by ..." (exit 3: everything
      else was written, the deck was not rebuilt).
  blocks [--frozen F] [--doc PATH ...]
      Rewrite the generated blocks of the target docs (all present ones without --doc).
  check [--frozen F] [--doc PATH ...]
      The D-03 consistency check; ends with "deliverables-check-ok docs N" (exit 0) or exit 1.
  deck [--frozen F] [--out P] [--template T] [--check] [--allow-blank-template]
      Render the Review-3 deck from F/numbers.json and the frozen charts onto the template (unless --check),
      then check it; ends with "DECK P slides S notes N charts C ok" (exit 0), FAIL deck lines (exit 1), or
      "FAIL deck: python-pptx is not importable by ..." (exit 3: this interpreter cannot draw the deck).

--frozen for build must name a phase5-* directory directly under the results root that is empty, absent or
an earlier frozen directory (it is replaced); any other path, a source directory included, is refused.

One fact per printed line, ASCII only. --doc must name one of the target docs (README.md,
docs/SDD.md, docs/viva/04-..., docs/viva/05-...), as a path relative to the repository root.
"""

import argparse
import os
import sys

from src.deliverables import FROZEN_DIR, RESULTS_ROOT, slides
from src.deliverables.blocks import DOC_TARGETS
from src.deliverables.build import REPO_ROOT, DeckUnavailable, DeliverablesError, apply_docs, build
from src.deliverables.consistency import load_numbers, run_check, select_docs


def _say(line: str):
    print(line, flush=True)


def _fail(command: str, e: Exception) -> int:
    reason = str(e) if isinstance(e, (DeliverablesError, ValueError)) else f"{type(e).__name__}: {e}"
    _say(f"FAIL {command}: {reason}")
    return 1


def _deck(args) -> int:
    try:
        numbers = load_numbers(args.frozen)
        lines, code = slides.run_deck(numbers, args.frozen, args.out, args.template, render=not args.check,
                                      allow_blank_template=args.allow_blank_template)
    except slides.PptxUnavailable:
        _say(slides.missing_line(sys.executable))
        return 3
    except (OSError, ValueError, KeyError) as e:
        return _fail("deck", e)
    for line in lines:
        _say(line)
    return code


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.deliverables",
                                     description="Freeze the Phase 4 results and regenerate the Phase 5 deliverables.")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="freeze (or reuse) the results, regenerate everything, refresh and check docs")
    b.add_argument("--results-root", default=RESULTS_ROOT, help="directory holding the phase4-* directories")
    b.add_argument("--reuse-frozen", action="store_true", help="keep the existing frozen copy; only re-render")
    b.add_argument("--no-docs", action="store_true", help="do not touch or check the target docs")
    b.add_argument("--no-deck", action="store_true", help="do not render or check the Review-3 deck")
    b.add_argument("--deck-out", default=None,
                   help="the deck to write (default docs/OS_Level_Guardrails_Review_3.pptx in the repository)")
    b.add_argument("--allow-blank-template", action="store_true",
                   help="accept a deck drawn on a blank 16:9 deck when the template cannot be opened")
    others = [sub.add_parser("blocks", help="rewrite the generated blocks of the target docs"),
              sub.add_parser("check", help="check every target doc against numbers.json")]
    d = sub.add_parser("deck", help="render and check the Review-3 slide deck (python-pptx)")
    d.add_argument("--out", default=slides.DEFAULT_OUT,
                   help="the deck to write (default docs/OS_Level_Guardrails_Review_3.pptx)")
    d.add_argument("--template", default=slides.DEFAULT_TEMPLATE, help="the template deck (default docs/template.pptx)")
    d.add_argument("--check", action="store_true", help="only check the existing deck")
    d.add_argument("--allow-blank-template", action="store_true",
                   help="accept a deck drawn on a blank 16:9 deck when the template cannot be opened")
    d.add_argument("--frozen", default=FROZEN_DIR, help="frozen directory (default results/phase5-final)")
    for p in [b] + others:
        p.add_argument("--frozen", default=FROZEN_DIR, help="frozen directory (default results/phase5-final)")
        p.add_argument("--doc", action="append", default=None,
                       help=f"a target doc to process (repeatable; default all of {', '.join(DOC_TARGETS)})")
    args = parser.parse_args(argv)

    if args.command == "build":
        docs = None if args.no_docs else (args.doc or list(DOC_TARGETS))
        deck_out = args.deck_out or os.path.join(REPO_ROOT, slides.DEFAULT_OUT)
        try:
            build(args.results_root, args.frozen, reuse=args.reuse_frozen, say=_say, docs=docs, deck_out=deck_out,
                  no_deck=args.no_deck, allow_blank_template=args.allow_blank_template)
        except DeckUnavailable as e:
            _say(str(e))
            return 3
        except (DeliverablesError, OSError, ValueError, KeyError) as e:
            return _fail("build", e)
        return 0
    if args.command == "deck":
        return _deck(args)
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
