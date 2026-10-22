#!/bin/bash
# GPU job helper for the Qwen LoRA run: preflight, launch, poll.
#
#   bash src/qwen_lora/run_train.sh preflight OUT_DIR
#   bash src/qwen_lora/run_train.sh launch    OUT_DIR [train args...]
#   bash src/qwen_lora/run_train.sh poll      OUT_DIR [MAX_WAIT_SECONDS]
#
# preflight  Reads the GPU (nvidia-smi), the available RAM (/proc/meminfo) and `ollama ps`, appends the reading
#            to OUT_DIR/gpu_preflight.log and exits 0 only when the card has MIN_FREE_MIB free (default 3584) and
#            MIN_AVAIL_RAM_MIB of RAM is available (default 3000). It only reads: this machine is shared with
#            another project, so the script never stops or signals any process and never touches the model
#            server. If memory is short it says so and exits 1; waiting or freeing memory is a human decision.
# launch     Runs the preflight (re-checking PREFLIGHT_RECHECKS more times, PREFLIGHT_WAIT_S seconds apart), then
#            runs `python -m src.qwen_lora.train --out-dir OUT_DIR ...` in the FOREGROUND with stdout and stderr in
#            OUT_DIR/train_progress.log, and writes the trainer's exit code to OUT_DIR/train.done. Start it from a
#            wsl.exe call that stays attached for the whole run (the attached client keeps the WSL VM alive;
#            Phase 2 lost runs to VM restarts). With --resume among the arguments the log is appended to after a
#            separator line, otherwise it starts empty.
# poll       Waits at most MAX_WAIT_SECONDS (default 480) for the run to finish, then prints the last progress
#            lines, how many "step" lines there are, whether a FAILED line exists, and one STATUS line:
#            running | done-ok | done-failed <code> | dead. It sends no signal to anything.
#
# Done-marker codes in OUT_DIR/train.done: 0 ok, 3 a loss stopped being finite, 4 GPU memory (the 95 % guard or a
# CUDA out-of-memory error), 5 the preflight never passed, anything else a crash.
#
# Environment overrides: PYTHON (default venv/bin/python), MIN_FREE_MIB, MIN_AVAIL_RAM_MIB, PREFLIGHT_RECHECKS,
# PREFLIGHT_WAIT_S.
set -u
cd "$(dirname "$0")/../.." || exit 2

PYTHON="${PYTHON:-venv/bin/python}"
MIN_FREE_MIB="${MIN_FREE_MIB:-3584}"
MIN_AVAIL_RAM_MIB="${MIN_AVAIL_RAM_MIB:-3000}"
PREFLIGHT_RECHECKS="${PREFLIGHT_RECHECKS:-6}"
PREFLIGHT_WAIT_S="${PREFLIGHT_WAIT_S:-300}"

usage() {
  echo "usage: bash src/qwen_lora/run_train.sh preflight OUT_DIR"
  echo "       bash src/qwen_lora/run_train.sh launch OUT_DIR [train args...]"
  echo "       bash src/qwen_lora/run_train.sh poll OUT_DIR [MAX_WAIT_SECONDS]"
}

is_number() {
  case "$1" in
    ''|*[!0-9]*) return 1 ;;
    *) return 0 ;;
  esac
}

preflight() {
  local out="$1"
  mkdir -p "$out"
  local reading free avail_kib avail_mib
  reading=$(nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1)
  free=$(printf '%s\n' "$reading" | awk -F',' 'NF >= 4 { gsub(/[ \t\r]/, "", $3); print $3 }')
  avail_kib=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo 2>/dev/null)
  if is_number "${avail_kib:-}"; then avail_mib=$((avail_kib / 1024)); else avail_mib=0; fi
  {
    echo "=== preflight $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    echo "nvidia-smi: ${reading:-<no reading>}"
    echo "free_mib ${free:-unknown}"
    echo "avail_ram_mib ${avail_mib}"
    if command -v ollama > /dev/null 2>&1; then
      echo "ollama ps:"
      timeout 10 ollama ps 2>&1 | head -n 10
    fi
  } >> "$out/gpu_preflight.log"
  if ! is_number "${free:-}"; then
    echo "GPU preflight failed: no usable nvidia-smi reading ('${reading:-}'), ${MIN_FREE_MIB} MiB required."
    echo "This script only reads; it never stops other processes."
    return 1
  fi
  if [ "$free" -lt "$MIN_FREE_MIB" ]; then
    echo "GPU preflight failed: ${free} MiB free, ${MIN_FREE_MIB} MiB required."
    echo "This script never stops other processes; wait for the other job or free the memory yourself."
    return 1
  fi
  if [ "$avail_mib" -lt "$MIN_AVAIL_RAM_MIB" ]; then
    echo "RAM preflight failed: ${avail_mib} MiB available, ${MIN_AVAIL_RAM_MIB} MiB required."
    echo "This script never stops other processes; wait for the other job or free the memory yourself."
    return 1
  fi
  echo "preflight ok: ${free} MiB GPU free, ${avail_mib} MiB RAM available"
  return 0
}

launch() {
  local out="$1"
  shift
  mkdir -p "$out"
  # The previous run's done-marker goes first, before the preflight can wait for up to 30 minutes: otherwise
  # `poll` reports that run's done-ok (or done-failed) for a launch that has not started (review WR-02).
  rm -f "$out/train.done"
  local attempts=$((PREFLIGHT_RECHECKS + 1)) tried=0 passed=0
  while [ "$tried" -lt "$attempts" ]; do
    if preflight "$out"; then
      passed=1
      break
    fi
    tried=$((tried + 1))
    if [ "$tried" -lt "$attempts" ]; then
      echo "preflight attempt ${tried}/${attempts} failed; checking again in ${PREFLIGHT_WAIT_S} s"
      sleep "$PREFLIGHT_WAIT_S"
    fi
  done
  if [ "$passed" -ne 1 ]; then
    echo "preflight never passed after ${attempts} attempt(s); not starting the training"
    echo 5 > "$out/train.done"
    exit 5
  fi
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  local resuming=0 arg
  for arg in "$@"; do
    if [ "$arg" = "--resume" ]; then resuming=1; fi
  done
  if [ "$resuming" -eq 1 ]; then
    echo "=== relaunch with --resume $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$out/train_progress.log"
    "$PYTHON" -u -m src.qwen_lora.train --out-dir "$out" "$@" >> "$out/train_progress.log" 2>&1
  else
    "$PYTHON" -u -m src.qwen_lora.train --out-dir "$out" "$@" > "$out/train_progress.log" 2>&1
  fi
  local code=$?
  echo "$code" > "$out/train.done"
  exit "$code"
}

alive() {
  # the training process, or the launcher that is still waiting for the preflight to pass
  local out="$1"
  pgrep -f -- "src[.]qwen_lora[.]train --out-dir ${out}( |\$)" > /dev/null 2>&1 && return 0
  pgrep -f -- "run_train[.]sh launch ${out}( |\$)" > /dev/null 2>&1 && return 0
  return 1
}

poll() {
  local out="$1" max_wait="${2:-480}" waited=0 nap
  while true; do
    [ -f "$out/train.done" ] && break
    alive "$out" || break
    [ "$waited" -ge "$max_wait" ] && break
    nap=$((max_wait - waited))
    [ "$nap" -gt 15 ] && nap=15
    sleep "$nap"
    waited=$((waited + nap))
  done
  if [ -f "$out/train_progress.log" ]; then
    tail -n 3 "$out/train_progress.log"
    echo "step lines: $(grep -c '^step ' "$out/train_progress.log")"
    if grep -q '^FAILED' "$out/train_progress.log"; then echo "FAILED line: yes"; else echo "FAILED line: no"; fi
  else
    echo "no progress log yet"
  fi
  if [ -f "$out/train.done" ]; then
    local code
    code=$(tr -d '[:space:]' < "$out/train.done")
    if [ "$code" = "0" ]; then echo "STATUS done-ok"; else echo "STATUS done-failed ${code}"; fi
  elif alive "$out"; then
    echo "STATUS running"
  else
    echo "STATUS dead"
  fi
}

command_name="${1:-}"
case "$command_name" in
  preflight)
    [ $# -ge 2 ] || { usage; exit 2; }
    preflight "$2"
    exit $?
    ;;
  launch)
    [ $# -ge 2 ] || { usage; exit 2; }
    out_dir="$2"
    shift 2
    launch "$out_dir" "$@"
    ;;
  poll)
    [ $# -ge 2 ] || { usage; exit 2; }
    poll "$2" "${3:-480}"
    ;;
  *)
    usage
    exit 2
    ;;
esac
