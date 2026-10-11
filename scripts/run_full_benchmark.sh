#!/bin/bash
# Full multi-LLM benchmark: one background process per model (they don't share
# rate limits), each writing results/<run>/<model>/. Survives closing the terminal.
#
#   bash scripts/run_full_benchmark.sh [run-name] [model ...]
#   tail -f results/<run>/*/progress.log          # watch progress
#   bash scripts/run_full_benchmark.sh <run-name>   # re-run = resume unfinished episodes
#
# When every model has finished:
#   venv/bin/python run_benchmark.py --report-only results/<run> --merge results/<run>/*/
set -euo pipefail
cd "$(dirname "$0")/.."

RUN="${1:-final-$(date +%Y%m%d)}"
shift || true
MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  mapfile -t MODELS < <(venv/bin/python -c "from src.llm import list_models; print('\n'.join(list_models(benchmark_only=True)))")
fi

for model in "${MODELS[@]}"; do
  dir="results/$RUN/$model"
  mkdir -p "$dir"
  nohup venv/bin/python -u run_benchmark.py --models "$model" --resume "$dir" >> "$dir/progress.log" 2>&1 &
  echo "started $model (pid $!) -> $dir/progress.log"
done
