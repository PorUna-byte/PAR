#!/bin/bash
#SBATCH --job-name=par-rating
#SBATCH --partition=batch
## Cluster-specific: uncomment and edit for your own node names.
##SBATCH --nodelist=node-[0-7]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=30-00:00:00
#SBATCH --output=logs/par-rating-%j.log
#SBATCH --error=logs/par-rating-%j.error

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch src/sbatch/05_rating.sh
  sbatch src/sbatch/05_rating.sh --run RUN_NAME
  sbatch src/sbatch/05_rating.sh --run /absolute/path/to/run_dir

Default discovery rates sampled online RL runs and sampled analysis_* runs.

Environment overrides:
  RATING_PARALLEL_WORKERS       Default: SLURM_CPUS_PER_TASK or 4
  RATING_STEP_PROGRESS_INTERVAL Default: 10
  RATING_FORCE                  Default: 0
EOF
}

resolve_common_sh() {
  if [[ -n "${PROJECT_ROOT:-}" && -f "${PROJECT_ROOT}/src/sbatch/_common.sh" ]]; then
    printf '%s\n' "${PROJECT_ROOT}/src/sbatch/_common.sh"
    return 0
  fi
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/src/sbatch/_common.sh" ]]; then
    printf '%s\n' "${SLURM_SUBMIT_DIR}/src/sbatch/_common.sh"
    return 0
  fi
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -f "${script_dir}/_common.sh" ]]; then
    printf '%s\n' "${script_dir}/_common.sh"
    return 0
  fi
  return 1
}

COMMON_SH="$(resolve_common_sh)"
if [[ -z "${COMMON_SH:-}" ]]; then
  echo "[ERROR] Could not locate src/sbatch/_common.sh" >&2
  exit 1
fi
source "$COMMON_SH"

export PIPELINE_REQUIRES_GPU=0
export AUTO_BATCH_SIZE=0
export PYTHONUNBUFFERED=1
export PIPELINE_NAME=05_rating

RATING_PARALLEL_WORKERS="${RATING_PARALLEL_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
RATING_STEP_PROGRESS_INTERVAL="${RATING_STEP_PROGRESS_INTERVAL:-10}"
RATING_FORCE="${RATING_FORCE:-0}"

RUN_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      [[ $# -ge 2 ]] || { echo "[ERROR] --run requires a value" >&2; exit 1; }
      RUN_ARGS+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

init_pipeline "05_rating"

COMPLETED_LOG="$PIPELINE_STATE_DIR/completed.log"
mkdir -p "$PIPELINE_STATE_DIR"
echo "Rating pipeline state: $PIPELINE_STATE_DIR"
echo "Rating completed log: $COMPLETED_LOG"

EXP_RUNS_DIR="${EXP_RUNS_DIR:-$PROJECT_ROOT/exp_runs}"
mkdir -p "$EXP_RUNS_DIR"

rate_run_dir() {
  local run_dir="$1"
  local run_name
  run_name="$(basename "$run_dir")"

  if [[ ! -d "$run_dir" ]]; then
    echo "[SKIP] rating:${run_name} directory does not exist"
    return 0
  fi

  local step_count=0
  if [[ -d "$run_dir/sample_on_test" ]]; then
    step_count="$(find "$run_dir/sample_on_test" -maxdepth 1 -type d -name 'step_*' | wc -l | tr -d ' ')"
  fi

  echo "[RUN ] rating:${run_name} step_count=${step_count} workers=${RATING_PARALLEL_WORKERS}"

  local cmd=(
    python "$SRC_ROOT/llm_rating/main.py"
    --run_dir "$run_dir"
    --parallel_workers "$RATING_PARALLEL_WORKERS"
    --step_progress_interval "$RATING_STEP_PROGRESS_INTERVAL"
  )

  if [[ "$RATING_FORCE" == "1" ]]; then
    cmd+=(--force)
  fi

  "${cmd[@]}"
}

should_rate_discovered_run() {
  local run_dir="$1"
  local run_name
  run_name="$(basename "$run_dir")"

  case "$run_name" in
    sft*|reward*|analysis_reward*)
      return 1
      ;;
    ppo_*|grpo_*|a2c_*|dpo_*)
      return 0
      ;;
    analysis_*)
      [[ -d "$run_dir/sample_on_test" ]]
      return $?
      ;;
    *)
      return 1
      ;;
  esac
}

declare -a RUN_DIRS=()
if [[ "${#RUN_ARGS[@]}" -gt 0 ]]; then
  for run_arg in "${RUN_ARGS[@]}"; do
    if [[ -d "$run_arg" ]]; then
      RUN_DIRS+=("$(cd "$run_arg" && pwd)")
    else
      RUN_DIRS+=("${EXP_RUNS_DIR}/${run_arg}")
    fi
  done
else
  while IFS= read -r run_dir; do
    should_rate_discovered_run "$run_dir" || continue
    RUN_DIRS+=("$run_dir")
  done < <(find "$EXP_RUNS_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
fi

if [[ "${#RUN_DIRS[@]}" -eq 0 ]]; then
  echo "[SKIP] rating no matching run directories found under ${EXP_RUNS_DIR}"
  exit 0
fi

for run_dir in "${RUN_DIRS[@]}"; do
  run_name="$(basename "$run_dir")"
  if [[ "${#RUN_ARGS[@]}" -eq 0 ]] && ! should_rate_discovered_run "$run_dir"; then
    echo "[SKIP] rating:${run_name} excluded by default discovery"
    continue
  fi
  rate_run_dir "$run_dir"
done
