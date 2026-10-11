#!/usr/bin/env python3
"""
Multi-LLM prompt-injection benchmark
=====================================
Runs InjecAgent indirect-injection cases (+ benign tasks) against each model,
once WITHOUT guardrails (emulated, nothing executes) and once WITH the full
stack (tool broker → policy engine → namespaces → seccomp), then writes
results/<timestamp>/{raw.jsonl, summary.csv, summary.md, charts/}.

Usage (in WSL):
  python run_benchmark.py --models all
  python run_benchmark.py --models gemini-flash-lite,ollama-llama3.2-3b --runs 1 --dh 3 --ds 3
  python run_benchmark.py --resume results/20261011-1500      # continue an interrupted run
  python run_benchmark.py --report-only results/20261011-1500
  python run_benchmark.py --report-only results/final --merge results/final/*/   # one run per model
"""

import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.benchmark.injecagent import attack_cases, benign_cases, load_tool_specs
from src.benchmark.report import generate_report
from src.benchmark.runner import ARMS, run_benchmark
from src.llm import list_models


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="all", help="Comma-separated model names, or 'all' (config/models.yaml)")
    parser.add_argument("--runs", type=int, default=3, help="Repetitions per case (default 3)")
    parser.add_argument("--dh", type=int, default=20, help="Direct-harm attack cases (default 20)")
    parser.add_argument("--ds", type=int, default=20, help="Data-stealing attack cases (default 20)")
    parser.add_argument("--benign", type=int, default=17, help="Benign cases (max 17)")
    parser.add_argument("--seed", type=int, default=0, help="Case sampling seed")
    parser.add_argument("--arms", default=",".join(ARMS), help="unprotected,protected")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds between episodes (free-tier rate limits)")
    parser.add_argument("--resume", help="Existing results directory to continue")
    parser.add_argument("--report-only", help="Only regenerate the report for a results directory")
    parser.add_argument("--merge", nargs="+", metavar="DIR",
                        help="With --report-only: build one report from several result directories")
    args = parser.parse_args()

    if args.report_only:
        generate_report(args.report_only, sources=args.merge)
        print(f"Report written to {args.report_only}")
        return

    models = list_models(benchmark_only=True) if args.models == "all" else args.models.split(",")
    specs = load_tool_specs()
    cases = benign_cases(specs)[:args.benign] + attack_cases(args.dh, args.ds, args.seed, specs)
    out_dir = args.resume or os.path.join("results", datetime.datetime.now().strftime("%Y%m%d-%H%M"))

    print(f"Models: {', '.join(models)}")
    print(f"Cases:  {len(cases)} ({args.benign} benign, {args.dh} direct-harm, {args.ds} data-stealing) × {args.runs} runs")
    print(f"Output: {out_dir}\n")
    run_benchmark(models, cases, out_dir, runs=args.runs, arms=tuple(args.arms.split(",")),
                  max_steps=args.max_steps, sleep_s=args.sleep)
    summary = generate_report(out_dir)
    print(f"\nReport: {os.path.join(out_dir, 'summary.md')}")
    for s in summary:
        print(f"  {s['model']:22} attempted {s['attempted_pct']}%  "
              f"success {s['success_unprotected_pct']}% → {s['success_protected_pct']}% with guardrails")


if __name__ == "__main__":
    main()
