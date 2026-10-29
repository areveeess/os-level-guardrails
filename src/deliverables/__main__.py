"""
Deliverables CLI — python -m src.deliverables build
===================================================
  build [--results-root R] [--frozen F] [--reuse-frozen]
      Freeze the Phase 4 directories into F (unless --reuse-frozen), then regenerate the
      report and numbers.json from the frozen copy. One fact per printed line, ASCII only;
      the last line is "BUILD ok" (exit 0) or "FAIL build: reason" (exit 1).
"""

import argparse
import sys

from src.deliverables import FROZEN_DIR, RESULTS_ROOT
from src.deliverables.build import DeliverablesError, build


def _say(line: str):
    print(line, flush=True)


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.deliverables",
                                     description="Freeze the Phase 4 results and regenerate the Phase 5 deliverables.")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="freeze (or reuse) the results and regenerate report and numbers.json")
    b.add_argument("--results-root", default=RESULTS_ROOT, help="directory holding the phase4-* directories")
    b.add_argument("--frozen", default=FROZEN_DIR, help="frozen output directory (default results/phase5-final)")
    b.add_argument("--reuse-frozen", action="store_true", help="keep the existing frozen copy; only re-render")
    args = parser.parse_args(argv)

    if args.command == "build":
        try:
            build(args.results_root, args.frozen, reuse=args.reuse_frozen, say=_say)
        except (DeliverablesError, OSError, ValueError, KeyError) as e:
            _say(f"FAIL build: {e}" if isinstance(e, DeliverablesError) else f"FAIL build: {type(e).__name__}: {e}")
            return 1
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
