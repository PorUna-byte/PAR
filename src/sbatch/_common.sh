#!/usr/bin/env bash

_step_completed() {
  local label="$1"

  [[ -f "$PIPELINE_LOG" ]] || return 1

  awk -F '\t' -v label="$label" '
    $2 == "success" && $3 == label { found = 1; exit }
    END { exit(found ? 0 : 1) }
  ' "$PIPELINE_LOG"
}

_mark_step_done() {
  local label="$1"
  local reason="$2"

  if _step_completed "$label"; then
    return 0
  fi

  printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$reason" "$label" >> "$PIPELINE_LOG"
  echo "[MARK] $reason -> $label"
}

_extract_first_int() {
  local value="$1"
  if [[ "$value" =~ ([0-9]+) ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

_count_visible_gpus() {
  local value="$1"
  [[ -z "$value" || "$value" == "NoDevFiles" ]] && return 1
  awk -F',' '{print NF}' <<< "$value"
}

_first_gpu_total_memory_gb() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local mib
    mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')
    if [[ -n "$mib" ]]; then
      awk -v mib="$mib" 'BEGIN { printf "%d", int((mib / 1024.0) + 0.5) }'
      return 0
    fi
  fi
  return 1
}

_detect_gpu_count() {
  local detected

  detected=$(_extract_first_int "${SLURM_GPUS_ON_NODE:-}") && { printf '%s' "$detected"; return 0; }
  detected=$(_count_visible_gpus "${CUDA_VISIBLE_DEVICES:-}") && { printf '%s' "$detected"; return 0; }
  detected=$(_extract_first_int "${GPUS_PER_NODE:-}") && { printf '%s' "$detected"; return 0; }
  detected=$(_extract_first_int "${SLURM_GPUS_PER_NODE:-}") && { printf '%s' "$detected"; return 0; }
  detected=$(_count_visible_gpus "${SLURM_JOB_GPUS:-}") && { printf '%s' "$detected"; return 0; }

  if command -v nvidia-smi >/dev/null 2>&1; then
    detected=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    [[ -n "$detected" && "$detected" -gt 0 ]] && { printf '%s' "$detected"; return 0; }
  fi

  printf '4'
}

configure_runtime_tmpdirs() {
  local inherited_tmpdir="${TMPDIR:-}"
  local tmp_root
  if [[ -n "${PAR_TMP_ROOT:-}" ]]; then
    tmp_root="$PAR_TMP_ROOT"
  elif [[ -n "$inherited_tmpdir" && "$inherited_tmpdir" != "/tmp" ]]; then
    tmp_root="$inherited_tmpdir"
  else
    tmp_root="$HOME/tmp"
  fi
  local torch_ext_root="${TORCH_EXTENSIONS_DIR:-$HOME/torch_extensions}"
  local ray_tmp_root="${RAY_TMPDIR:-$tmp_root/ray}"
  local ray_spill_root="${RAY_OBJECT_SPILLING_DIR:-$ray_tmp_root/spill}"
  local ray_air_root="${RAY_AIR_LOCAL_CACHE_DIR:-$tmp_root/ray_air}"
  local triton_root="${TRITON_CACHE_DIR:-$tmp_root/triton}"

  mkdir -p "$tmp_root" "$torch_ext_root" "$ray_tmp_root" "$ray_spill_root" "$ray_air_root" "$triton_root"
  chmod 700 "$tmp_root" 2>/dev/null || true

  export TMPDIR="$tmp_root"
  export TEMP="$tmp_root"
  export TMP="$tmp_root"
  export TORCH_EXTENSIONS_DIR="$torch_ext_root"
  export RAY_TMPDIR="$ray_tmp_root"
  export RAY_OBJECT_SPILLING_DIR="$ray_spill_root"
  export RAY_AIR_LOCAL_CACHE_DIR="$ray_air_root"
  export TRITON_CACHE_DIR="$triton_root"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  export NCCL_NET="${NCCL_NET:-socket}"
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
}

configure_pythonpath() {
  local current="${PYTHONPATH:-}"
  current=":$current:"
  current="${current//:$SRC_ROOT:/:}"
  while [[ "$current" == *"::"* ]]; do
    current="${current//::/:}"
  done
  current="${current#:}"
  current="${current%:}"
  export PYTHONPATH="$SRC_ROOT${current:+:$current}"
}

_normalize_batch_key() {
  local value="$1"
  value="${value//[^a-zA-Z0-9]/_}"
  printf '%s' "$value" | tr '[:upper:]' '[:lower:]'
}

shell_join() {
  local arg
  for arg in "$@"; do
    printf '%q ' "$arg"
  done
}

DEFAULT_GEMMA2_2B_GRPO_CHUNK_SIZE=${DEFAULT_GEMMA2_2B_GRPO_CHUNK_SIZE:-9}
DEFAULT_GEMMA2_9B_GRPO_CHUNK_SIZE=${DEFAULT_GEMMA2_9B_GRPO_CHUNK_SIZE:-2}
DEFAULT_QWEN3_4B_BASE_GRPO_CHUNK_SIZE=${DEFAULT_QWEN3_4B_BASE_GRPO_CHUNK_SIZE:-4}
DEFAULT_LLAMA3_1_8B_GRPO_CHUNK_SIZE=${DEFAULT_LLAMA3_1_8B_GRPO_CHUNK_SIZE:-2}

attention_impl_for_model() {
  local model_key
  model_key=$(_normalize_batch_key "$1")

  case "$model_key" in
    gemma2_2b|gemma2_9b) echo eager ;;
    qwen3_4b_base|llama_3_1_8b) echo sdpa ;;
    *) echo sdpa ;;
  esac
}

step_pending() {
  local label="$1"
  ! _step_completed "$label"
}

_default_batch_size_per_gpu() {
  local phase="$1"
  local objective="$2"
  local model="$3"

  case "${phase}:${objective}:${model}" in
    train:sft:gemma2_2b) echo 12 ;;
    eval:sft:gemma2_2b) echo 12 ;;
    train:sft:gemma2_9b) echo 3 ;;
    eval:sft:gemma2_9b) echo 3 ;;
    train:sft:qwen3_4b_base) echo 6 ;;
    eval:sft:qwen3_4b_base) echo 6 ;;
    train:sft:llama_3_1_8b) echo 3 ;;
    eval:sft:llama_3_1_8b) echo 3 ;;

    infer:reference:gemma2_2b) echo 12 ;;
    infer:rollout:gemma2_2b) echo 12 ;;
    infer:reference:gemma2_9b) echo 4 ;;
    infer:rollout:gemma2_9b) echo 4 ;;
    infer:reference:qwen3_4b_base) echo 8 ;;
    infer:rollout:qwen3_4b_base) echo 8 ;;
    infer:reference:llama_3_1_8b) echo 5 ;;
    infer:rollout:llama_3_1_8b) echo 5 ;;

    train:reward:gemma2_2b) echo 8 ;;
    eval:reward:gemma2_2b) echo 8 ;;
    train:reward:gemma2_9b) echo 2 ;;
    eval:reward:gemma2_9b) echo 2 ;;
    train:reward:qwen3_4b_base) echo 4 ;;
    eval:reward:qwen3_4b_base) echo 4 ;;
    train:reward:llama_3_1_8b) echo 2 ;;
    eval:reward:llama_3_1_8b) echo 2 ;;

    train:ppo:gemma2_2b) echo 9 ;;
    train:a2c:gemma2_2b) echo 9 ;;
    train:grpo:gemma2_2b) echo 9 ;;
    eval:ppo:gemma2_2b) echo 9 ;;
    eval:a2c:gemma2_2b) echo 9 ;;
    eval:grpo:gemma2_2b) echo 9 ;;
    train:ppo:gemma2_9b) echo 1 ;;
    train:a2c:gemma2_9b) echo 1 ;;
    train:grpo:gemma2_9b) echo 1 ;;
    eval:ppo:gemma2_9b) echo 2 ;;
    eval:a2c:gemma2_9b) echo 2 ;;
    eval:grpo:gemma2_9b) echo 2 ;;
    train:ppo:qwen3_4b_base) echo 4 ;;
    train:a2c:qwen3_4b_base) echo 4 ;;
    train:grpo:qwen3_4b_base) echo 4 ;;
    eval:ppo:qwen3_4b_base) echo 6 ;;
    eval:a2c:qwen3_4b_base) echo 6 ;;
    eval:grpo:qwen3_4b_base) echo 6 ;;
    train:ppo:llama_3_1_8b) echo 2 ;;
    train:a2c:llama_3_1_8b) echo 2 ;;
    train:grpo:llama_3_1_8b) echo 2 ;;
    eval:ppo:llama_3_1_8b) echo 3 ;;
    eval:a2c:llama_3_1_8b) echo 3 ;;
    eval:grpo:llama_3_1_8b) echo 3 ;;

    train:dpo:gemma2_2b) echo 16 ;;
    eval:dpo:gemma2_2b) echo 16 ;;
    train:dpo:gemma2_9b) echo 4 ;;
    eval:dpo:gemma2_9b) echo 4 ;;
    train:dpo:qwen3_4b_base) echo 8 ;;
    eval:dpo:qwen3_4b_base) echo 8 ;;
    train:dpo:llama_3_1_8b) echo 4 ;;
    eval:dpo:llama_3_1_8b) echo 4 ;;

    train:analysis_ppo:gemma2_2b) echo 9 ;;
    eval:analysis_ppo:gemma2_2b) echo 9 ;;
    train:analysis_ppo:gemma2_9b) echo 2 ;;
    eval:analysis_ppo:gemma2_9b) echo 2 ;;
    train:analysis_ppo:qwen3_4b_base) echo 4 ;;
    eval:analysis_ppo:qwen3_4b_base) echo 4 ;;
    train:analysis_ppo:llama_3_1_8b) echo 2 ;;
    eval:analysis_ppo:llama_3_1_8b) echo 2 ;;

    *)
      case "$model" in
        gemma2_2b) echo 9 ;;
        gemma2_9b) echo 2 ;;
        qwen3_4b_base) echo 4 ;;
        llama_3_1_8b) echo 2 ;;
        *) echo 1 ;;
      esac
      ;;
  esac
}

_clamp_positive_int() {
  local value="$1"
  local minimum="${2:-1}"
  if [[ -z "$value" || "$value" -lt "$minimum" ]]; then
    printf '%s' "$minimum"
  else
    printf '%s' "$value"
  fi
}

_auto_batch_strategy() {
  printf '%s' "${AUTO_BATCH_STRATEGY:-probe}"
}

_auto_batch_probe_enabled() {
  [[ "${AUTO_BATCH_SIZE:-1}" != "0" && "$(_auto_batch_strategy)" == "probe" ]]
}

_auto_batch_probe_multiplier_for_phase() {
  local phase="$1"
  local phase_upper
  phase_upper=$(printf '%s' "$phase" | tr '[:lower:]' '[:upper:]')
  local env_name="AUTO_BATCH_PROBE_MAX_MULTIPLIER_${phase_upper}"
  printf '%s' "${!env_name:-${AUTO_BATCH_PROBE_MAX_MULTIPLIER:-3.0}}"
}

_auto_batch_probe_max_batch_size() {
  local base_batch_size="$1"
  local phase="$2"
  local multiplier
  multiplier=$(_auto_batch_probe_multiplier_for_phase "$phase")
  python - "$base_batch_size" "$multiplier" <<'PY'
import math
import sys
base = max(int(sys.argv[1]), 1)
multiplier = max(float(sys.argv[2]), 1.0)
print(max(base, int(math.ceil(base * multiplier))))
PY
}

num_attention_heads_for_model() {
  local model_key
  model_key=$(_normalize_batch_key "$1")

  case "$model_key" in
    gemma2_2b) echo 8 ;;
    gemma2_9b) echo 16 ;;
    qwen3_4b_base) echo 32 ;;
    llama_3_1_8b) echo 32 ;;
    *) echo 1 ;;
  esac
}

vllm_tensor_parallel_size_for_model() {
  local model="$1"
  local gpu_count="$2"
  local leave_free_gpus="${3:-1}"
  local heads max_tp candidate

  heads=$(num_attention_heads_for_model "$model")
  if (( gpu_count <= leave_free_gpus )); then
    echo 1
    return 0
  fi

  max_tp=$((gpu_count - leave_free_gpus))
  for ((candidate=max_tp; candidate>=1; candidate--)); do
    if (( heads % candidate == 0 )); then
      echo "$candidate"
      return 0
    fi
  done

  echo 1
}

autotune_batch_size_from_command() {
  local label="$1"
  local phase="$2"
  local base_batch_size="$3"
  local shell_command="$4"
  local max_batch_size="${5:-}"

  if ! _auto_batch_probe_enabled; then
    printf '%s\n' "$base_batch_size"
    return 0
  fi

  if [[ -z "$max_batch_size" ]]; then
    max_batch_size=$(_auto_batch_probe_max_batch_size "$base_batch_size" "$phase")
  fi

  local extra_args=()
  if [[ "${AUTO_BATCH_FORCE_RETUNE:-0}" == "1" ]]; then
    extra_args+=(--force)
  fi
  extra_args+=(--search-mode "${AUTO_BATCH_PROBE_SEARCH_MODE:-fast}")
  if [[ -n "${AUTO_BATCH_PROBE_GROWTH_FACTOR:-}" ]]; then
    extra_args+=(--growth-factor "${AUTO_BATCH_PROBE_GROWTH_FACTOR}")
  fi
  if [[ -n "${AUTO_BATCH_PROBE_FAST_REFINE_BIAS:-}" ]]; then
    extra_args+=(--fast-refine-bias "${AUTO_BATCH_PROBE_FAST_REFINE_BIAS}")
  fi
  if [[ -n "${AUTO_BATCH_PROBE_MIN_MEMORY_UTILIZATION:-}" ]]; then
    extra_args+=(--min-memory-utilization "${AUTO_BATCH_PROBE_MIN_MEMORY_UTILIZATION}")
  fi
  if [[ -n "${AUTO_BATCH_PROBE_TARGET_MEMORY_UTILIZATION:-}" ]]; then
    extra_args+=(--target-memory-utilization "${AUTO_BATCH_PROBE_TARGET_MEMORY_UTILIZATION}")
  fi
  if [[ -n "${AUTO_BATCH_PROBE_MAX_MEMORY_UTILIZATION:-}" ]]; then
    extra_args+=(--max-memory-utilization "${AUTO_BATCH_PROBE_MAX_MEMORY_UTILIZATION}")
  fi
  if [[ -n "${AUTO_BATCH_PROBE_MEMORY_TOLERANCE:-}" ]]; then
    extra_args+=(--memory-tolerance "${AUTO_BATCH_PROBE_MEMORY_TOLERANCE}")
  fi
  if [[ -n "${AUTO_BATCH_PROBE_TARGET_GROWTH_CAP:-}" ]]; then
    extra_args+=(--target-growth-cap "${AUTO_BATCH_PROBE_TARGET_GROWTH_CAP}")
  fi
  if [[ -n "${AUTO_BATCH_PROBE_TARGET_SAFETY_FACTOR:-}" ]]; then
    extra_args+=(--target-safety-factor "${AUTO_BATCH_PROBE_TARGET_SAFETY_FACTOR}")
  fi
  if [[ -n "${AUTO_BATCH_PROBE_TARGET_REFINE_BIAS:-}" ]]; then
    extra_args+=(--target-refine-bias "${AUTO_BATCH_PROBE_TARGET_REFINE_BIAS}")
  fi
  if [[ -n "${AUTO_BATCH_PROBE_MAX_MEMORY_GUIDED_TRIALS:-}" ]]; then
    extra_args+=(--max-memory-guided-trials "${AUTO_BATCH_PROBE_MAX_MEMORY_GUIDED_TRIALS}")
  fi

  python "$SRC_ROOT/scripts/autotune_batch_size.py" \
    --label "$label" \
    --shell-command "$shell_command" \
    --cache-root "$PIPELINE_STATE_ROOT/autobatch" \
    --base-batch-size "$base_batch_size" \
    --min-batch-size 1 \
    --max-batch-size "$max_batch_size" \
    --timeout-seconds "${AUTO_BATCH_PROBE_TIMEOUT_SECONDS:-1800}" \
    "${extra_args[@]}"
}

rollout_max_batch_prompts_for() {
  local train_batch_size_per_gpu="$1"
  local eval_batch_size_per_gpu="$2"
  local train_gpus="${TRAIN_GPUS:-$GPUS}"
  local per_gpu_max="$train_batch_size_per_gpu"
  if (( eval_batch_size_per_gpu > per_gpu_max )); then
    per_gpu_max="$eval_batch_size_per_gpu"
  fi
  printf '%s\n' "$((per_gpu_max * train_gpus))"
}

rollout_max_batch_tokens_for() {
  local prompt_budget="$1"
  local model_len="${2:-1212}"
  printf '%s\n' "$((prompt_budget * model_len))"
}

_objective_uses_fixed_batch_size() {
  local objective="$1"
  local objective_key
  objective_key=$(_normalize_batch_key "$objective")

  case "$objective_key" in
    ppo|grpo|dpo|a2c|analysis_ppo)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

batch_size_per_gpu_for() {
  local phase="$1"
  local objective="$2"
  local model="$3"

  local objective_key model_key
  objective_key=$(_normalize_batch_key "$objective")
  model_key=$(_normalize_batch_key "$model")

  local batch_size
  batch_size=$(_default_batch_size_per_gpu "$phase" "$objective_key" "$model_key")

  local phase_upper
  phase_upper=$(printf '%s' "$phase" | tr '[:lower:]' '[:upper:]')
  local env_specific_max="MAX_${phase_upper}_BATCH_SIZE_PER_GPU"
  local env_global_max="${MAX_BATCH_SIZE_PER_GPU:-}"
  local specific_max="${!env_specific_max:-}"
  if [[ -n "$specific_max" && "$batch_size" -gt "$specific_max" ]]; then
    batch_size="$specific_max"
  fi
  if [[ -n "$env_global_max" && "$batch_size" -gt "$env_global_max" ]]; then
    batch_size="$env_global_max"
  fi

  _clamp_positive_int "$batch_size"
}

default_batch_size_per_gpu_for() {
  local phase="$1"
  local objective="$2"
  local model="$3"

  local objective_key model_key
  objective_key=$(_normalize_batch_key "$objective")
  model_key=$(_normalize_batch_key "$model")

  _clamp_positive_int "$(_default_batch_size_per_gpu "$phase" "$objective_key" "$model_key")"
}

train_batch_size_per_gpu_for() {
  local model="$1"
  local objective="$2"
  batch_size_per_gpu_for train "$objective" "$model"
}

eval_batch_size_per_gpu_for() {
  local model="$1"
  local objective="$2"
  batch_size_per_gpu_for eval "$objective" "$model"
}

grpo_sequence_chunk_size_for() {
  local model_key
  model_key=$(_normalize_batch_key "$1")

  case "$model_key" in
    gemma2_2b) _clamp_positive_int "$DEFAULT_GEMMA2_2B_GRPO_CHUNK_SIZE" ;;
    gemma2_9b) _clamp_positive_int "$DEFAULT_GEMMA2_9B_GRPO_CHUNK_SIZE" ;;
    qwen3_4b_base) _clamp_positive_int "$DEFAULT_QWEN3_4B_BASE_GRPO_CHUNK_SIZE" ;;
    llama_3_1_8b) _clamp_positive_int "$DEFAULT_LLAMA3_1_8B_GRPO_CHUNK_SIZE" ;;
    *) _clamp_positive_int 1 ;;
  esac
}

inference_batch_size_per_gpu_for() {
  local model="$1"
  local objective="${2:-reference}"
  batch_size_per_gpu_for infer "$objective" "$model"
}

print_batch_size_per_gpu_plan() {
  local model="$1"
  local objective="$2"
  local train_batch_size_per_gpu eval_batch_size_per_gpu
  train_batch_size_per_gpu=$(train_batch_size_per_gpu_for "$model" "$objective")
  eval_batch_size_per_gpu=$(eval_batch_size_per_gpu_for "$model" "$objective")
  echo "[BATCH] objective=${objective} model=${model} train_batch_size_per_gpu=${train_batch_size_per_gpu} eval_batch_size_per_gpu=${eval_batch_size_per_gpu}"
}

init_pipeline() {
  if [[ -z "${SRC_ROOT:-}" ]]; then
    local common_dir
    common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SRC_ROOT="$(cd "$common_dir/.." && pwd)"
  fi

  if [[ -z "${PROJECT_ROOT:-}" ]]; then
    PROJECT_ROOT="$(cd "$SRC_ROOT/.." && pwd)"
  fi

  configure_pythonpath
  configure_runtime_tmpdirs
  local pipeline_requires_gpu="${PIPELINE_REQUIRES_GPU:-1}"
  if [[ "$pipeline_requires_gpu" == "0" ]]; then
    GPUS=${GPUS:-0}
  else
    GPUS=${GPUS:-$(_detect_gpu_count)}
  fi
  EXP_RUNS_DIR=${EXP_RUNS_DIR:-$PROJECT_ROOT/exp_runs}
  PIPELINE_STATE_ROOT=${PIPELINE_STATE_ROOT:-$PROJECT_ROOT/pipeline_state}
  PIPELINE_NAME=${PIPELINE_NAME:-$(basename "$0" .sh)}
  PIPELINE_STATE_DIR="$PIPELINE_STATE_ROOT/$PIPELINE_NAME"
  PIPELINE_LOG="$PIPELINE_STATE_DIR/completed.log"
  export PIPELINE_STATE_ROOT

  if [[ "${RESET_PIPELINE_STATE:-0}" == "1" ]]; then
    rm -rf "$PIPELINE_STATE_DIR"
  fi

  cd "$PROJECT_ROOT"
  mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/results" "$EXP_RUNS_DIR" "$PIPELINE_STATE_DIR"
  touch "$PIPELINE_LOG"

  echo "PWD=$(pwd)"
  echo "Start job on $(hostname)"
  if [[ "$pipeline_requires_gpu" == "0" ]]; then
    echo "Resolved GPUS=$GPUS (CPU-only pipeline)"
  else
    echo "Resolved GPUS=$GPUS"
  fi
  echo "AUTO_BATCH_SIZE=${AUTO_BATCH_SIZE:-1}"
  echo "AUTO_BATCH_STRATEGY=$(_auto_batch_strategy)"
  echo "AUTO_BATCH_PROBE_MAX_MULTIPLIER=${AUTO_BATCH_PROBE_MAX_MULTIPLIER:-3.0}"
  echo "AUTO_BATCH_PROBE_SEARCH_MODE=${AUTO_BATCH_PROBE_SEARCH_MODE:-fast}"
  echo "AUTO_BATCH_PROBE_GROWTH_FACTOR=${AUTO_BATCH_PROBE_GROWTH_FACTOR:-<default>}"
  echo "AUTO_BATCH_PROBE_FAST_REFINE_BIAS=${AUTO_BATCH_PROBE_FAST_REFINE_BIAS:-<default>}"
  echo "AUTO_BATCH_PROBE_MIN_MEMORY_UTILIZATION=${AUTO_BATCH_PROBE_MIN_MEMORY_UTILIZATION:-0.70}"
  echo "AUTO_BATCH_PROBE_TARGET_MEMORY_UTILIZATION=${AUTO_BATCH_PROBE_TARGET_MEMORY_UTILIZATION:-0.80}"
  echo "AUTO_BATCH_PROBE_MAX_MEMORY_UTILIZATION=${AUTO_BATCH_PROBE_MAX_MEMORY_UTILIZATION:-0.90}"
  echo "AUTO_BATCH_PROBE_MEMORY_TOLERANCE=${AUTO_BATCH_PROBE_MEMORY_TOLERANCE:-0.02}"
  echo "AUTO_BATCH_PROBE_TARGET_GROWTH_CAP=${AUTO_BATCH_PROBE_TARGET_GROWTH_CAP:-3.0}"
  echo "AUTO_BATCH_PROBE_TARGET_SAFETY_FACTOR=${AUTO_BATCH_PROBE_TARGET_SAFETY_FACTOR:-0.98}"
  echo "AUTO_BATCH_PROBE_TARGET_REFINE_BIAS=${AUTO_BATCH_PROBE_TARGET_REFINE_BIAS:-0.75}"
  echo "AUTO_BATCH_PROBE_MAX_MEMORY_GUIDED_TRIALS=${AUTO_BATCH_PROBE_MAX_MEMORY_GUIDED_TRIALS:-3}"
  echo "DEFAULT_GEMMA2_2B_GRPO_CHUNK_SIZE=$DEFAULT_GEMMA2_2B_GRPO_CHUNK_SIZE"
  echo "DEFAULT_GEMMA2_9B_GRPO_CHUNK_SIZE=$DEFAULT_GEMMA2_9B_GRPO_CHUNK_SIZE"
  echo "DEFAULT_QWEN3_4B_BASE_GRPO_CHUNK_SIZE=$DEFAULT_QWEN3_4B_BASE_GRPO_CHUNK_SIZE"
  echo "DEFAULT_LLAMA3_1_8B_GRPO_CHUNK_SIZE=$DEFAULT_LLAMA3_1_8B_GRPO_CHUNK_SIZE"
  echo "EXP_RUNS_DIR=$EXP_RUNS_DIR"
  echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<empty>}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<empty>}"
  echo "SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-<empty>}"
  echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<empty>}"
  echo "GPUS_PER_NODE=${GPUS_PER_NODE:-<empty>}"
  echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-<empty>}"
  echo "NCCL_NET=${NCCL_NET:-<empty>}"
  echo "NCCL_DEBUG=${NCCL_DEBUG:-<empty>}"
  echo "TMPDIR=${TMPDIR:-<empty>}"
  echo "TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-<empty>}"
  echo "RAY_TMPDIR=${RAY_TMPDIR:-<empty>}"
  echo "RAY_OBJECT_SPILLING_DIR=${RAY_OBJECT_SPILLING_DIR:-<empty>}"
  echo "RAY_AIR_LOCAL_CACHE_DIR=${RAY_AIR_LOCAL_CACHE_DIR:-<empty>}"
  echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-<empty>}"
  echo "Pipeline state: $PIPELINE_STATE_DIR"
  echo "Completed log: $PIPELINE_LOG"
  python --version
  if [[ "$pipeline_requires_gpu" == "0" ]]; then
    echo "Skipping nvidia-smi for CPU-only pipeline."
  else
    nvidia-smi
  fi
}

_visible_device_list() {
  local value="${CUDA_VISIBLE_DEVICES:-}"
  if [[ -n "$value" && "$value" != "NoDevFiles" ]]; then
    printf '%s\n' "$value"
    return 0
  fi

  local devices=()
  local idx
  for ((idx = 0; idx < GPUS; idx++)); do
    devices+=("$idx")
  done
  local joined
  joined=$(IFS=,; echo "${devices[*]}")
  printf '%s\n' "$joined"
}

_take_last_visible_devices() {
  local csv="$1"
  local keep="$2"
  IFS=',' read -r -a devices <<< "$csv"
  local total="${#devices[@]}"
  local start=$((total - keep))
  local selected=()
  local idx
  for ((idx = start; idx < total; idx++)); do
    selected+=("${devices[idx]}")
  done
  local joined
  joined=$(IFS=,; echo "${selected[*]}")
  printf '%s\n' "$joined"
}

_take_first_visible_devices() {
  local csv="$1"
  local keep="$2"
  IFS=',' read -r -a devices <<< "$csv"
  local total="${#devices[@]}"
  if (( keep >= total )); then
    printf '%s\n' "$csv"
    return 0
  fi
  local selected=()
  local idx
  for ((idx = 0; idx < keep; idx++)); do
    selected+=("${devices[idx]}")
  done
  local joined
  joined=$(IFS=,; echo "${selected[*]}")
  printf '%s\n' "$joined"
}

_auto_rollout_gpus_for_objective() {
  local objective="$1"
  local total_gpus="$2"
  local objective_key
  objective_key=$(_normalize_batch_key "$objective")

  if (( total_gpus <= 1 )); then
    printf '0\n'
    return 0
  fi

  case "$objective_key" in
    sft|reward|reward_odin|dpo)
      printf '0\n'
      ;;
    grpo)
      printf '0\n'
      ;;
    ppo|a2c|analysis_ppo)
      printf '0\n'
      ;;
    *)
      printf '1\n'
      ;;
  esac
}

_vllm_ray_rollout_preflight() {
  if [[ "${VLLM_RAY_PREFLIGHT:-1}" == "0" ]]; then
    return 0
  fi

  if [[ -n "${VLLM_RAY_PREFLIGHT_RESULT:-}" ]]; then
    [[ "$VLLM_RAY_PREFLIGHT_RESULT" == "ok" ]]
    return $?
  fi

  local output
  if output=$(python - <<'PY' 2>&1
import os
import sys
import traceback

os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
ray_tmpdir = os.environ.get("RAY_TMPDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "ray"))
ray_spill_dir = os.environ.get("RAY_OBJECT_SPILLING_DIR", os.path.join(ray_tmpdir, "spill"))
os.makedirs(ray_tmpdir, exist_ok=True)
os.makedirs(ray_spill_dir, exist_ok=True)

try:
    import ray
    import ray._private.ray_constants as ray_constants
    import ray._private.node as ray_node
    import psutil

    ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False
    if ray.is_initialized():
        ray.shutdown()

    original = ray_node.Node._get_system_processes_for_resource_isolation

    def patched(self):
        try:
            return original(self)
        except (psutil.AccessDenied, PermissionError):
            system_process_pids = []
            for process_infos in getattr(self, "all_processes", {}).values():
                for process_info in process_infos:
                    process = getattr(process_info, "process", None)
                    pid = getattr(process, "pid", None)
                    if pid is not None:
                        system_process_pids.append(str(pid))
            return ",".join(system_process_pids)

    ray_node.Node._get_system_processes_for_resource_isolation = patched

    init_kwargs = dict(
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
        num_cpus=1,
        enable_resource_isolation=False,
        _temp_dir=ray_tmpdir,
        object_spilling_directory=ray_spill_dir,
        _skip_env_hook=True,
    )
    try:
        ray.init(**init_kwargs)
    except TypeError:
        init_kwargs.pop("enable_resource_isolation", None)
        ray.init(**init_kwargs)
    ray.shutdown()
except Exception as exc:
    traceback.print_exc()
    sys.exit(1)
PY
  ); then
    VLLM_RAY_PREFLIGHT_RESULT="ok"
    return 0
  fi

  VLLM_RAY_PREFLIGHT_RESULT="failed"
  echo "[WARN] vllm_ray preflight failed: ${output}" >&2
  return 1
}

configure_rollout_layout() {
  local rollout_objective="${1:-${ROLLOUT_OBJECTIVE:-}}"
  local objective_key
  objective_key=$(_normalize_batch_key "$rollout_objective")
  ROLLOUT_BACKEND=${ROLLOUT_BACKEND:-deepspeed}

  if [[ "$ROLLOUT_BACKEND" == "deepspeed" ]]; then
    TRAIN_GPUS=${TRAIN_GPUS:-$GPUS}
    ROLLOUT_GPUS=${ROLLOUT_GPUS:-0}
    ROLLOUT_VLLM_TP=${ROLLOUT_VLLM_TP:-1}
    ROLLOUT_VISIBLE_DEVICES=${ROLLOUT_VISIBLE_DEVICES:-}
  else
    case "$objective_key" in
      sft|reward|reward_odin|dpo)
        echo "[INFO] objective=${rollout_objective:-<unset>} is offline; disabling dedicated rollout GPUs." >&2
        ROLLOUT_BACKEND=deepspeed
        TRAIN_GPUS=${TRAIN_GPUS:-$GPUS}
        ROLLOUT_GPUS=0
        ROLLOUT_VLLM_TP=1
        ROLLOUT_VISIBLE_DEVICES=
        echo "TRAIN_GPUS=${TRAIN_GPUS}"
        echo "ROLLOUT_BACKEND=${ROLLOUT_BACKEND}"
        echo "ROLLOUT_GPUS=${ROLLOUT_GPUS}"
        echo "ROLLOUT_VLLM_TP=${ROLLOUT_VLLM_TP}"
        echo "ROLLOUT_OBJECTIVE=${rollout_objective:-<unset>}"
        echo "ROLLOUT_VISIBLE_DEVICES=${ROLLOUT_VISIBLE_DEVICES:-<empty>}"
        return 0
        ;;
    esac

    if (( GPUS < 2 )); then
      echo "[WARN] Need at least 2 GPUs to dedicate rollout GPUs to vLLM; falling back to DeepSpeed generation." >&2
      ROLLOUT_BACKEND=deepspeed
      TRAIN_GPUS=${TRAIN_GPUS:-$GPUS}
      ROLLOUT_GPUS=0
      ROLLOUT_VLLM_TP=1
      ROLLOUT_VISIBLE_DEVICES=
      echo "TRAIN_GPUS=${TRAIN_GPUS}"
      echo "ROLLOUT_BACKEND=${ROLLOUT_BACKEND}"
      echo "ROLLOUT_GPUS=${ROLLOUT_GPUS}"
      echo "ROLLOUT_VLLM_TP=${ROLLOUT_VLLM_TP}"
      echo "ROLLOUT_VISIBLE_DEVICES=${ROLLOUT_VISIBLE_DEVICES:-<empty>}"
      return 0
    fi

    if [[ -z "${ROLLOUT_GPUS+x}" && -z "${TRAIN_GPUS+x}" ]]; then
      ROLLOUT_GPUS=$(_auto_rollout_gpus_for_objective "$rollout_objective" "$GPUS")
      ROLLOUT_GPUS=$(_clamp_positive_int "$ROLLOUT_GPUS" 0)
      TRAIN_GPUS=$((GPUS - ROLLOUT_GPUS))
    elif [[ -z "${ROLLOUT_GPUS+x}" ]]; then
      TRAIN_GPUS=${TRAIN_GPUS:-$((GPUS - 1))}
      ROLLOUT_GPUS=$((GPUS - TRAIN_GPUS))
    elif [[ -z "${TRAIN_GPUS+x}" ]]; then
      ROLLOUT_GPUS=${ROLLOUT_GPUS:-1}
      TRAIN_GPUS=$((GPUS - ROLLOUT_GPUS))
    fi

    if (( TRAIN_GPUS + ROLLOUT_GPUS < GPUS )); then
      local leftover=$((GPUS - TRAIN_GPUS - ROLLOUT_GPUS))
      echo "[WARN] ${leftover} GPU(s) would be idle; assigning them to rollout." >&2
      ROLLOUT_GPUS=$((ROLLOUT_GPUS + leftover))
    fi

    if (( ROLLOUT_GPUS == 0 )); then
      echo "[INFO] objective=${rollout_objective:-<unset>} uses no dedicated rollout GPUs; falling back to DeepSpeed generation." >&2
      ROLLOUT_BACKEND=deepspeed
      TRAIN_GPUS=${TRAIN_GPUS:-$GPUS}
      ROLLOUT_VLLM_TP=1
      ROLLOUT_VISIBLE_DEVICES=
      echo "TRAIN_GPUS=${TRAIN_GPUS}"
      echo "ROLLOUT_BACKEND=${ROLLOUT_BACKEND}"
      echo "ROLLOUT_GPUS=${ROLLOUT_GPUS}"
      echo "ROLLOUT_VLLM_TP=${ROLLOUT_VLLM_TP}"
      echo "ROLLOUT_OBJECTIVE=${rollout_objective:-<unset>}"
      echo "ROLLOUT_VISIBLE_DEVICES=${ROLLOUT_VISIBLE_DEVICES:-<empty>}"
      return 0
    fi

    if (( TRAIN_GPUS < 1 )); then
      echo "TRAIN_GPUS must be >= 1 when rollout backend is enabled." >&2
      exit 1
    fi
    if (( ROLLOUT_GPUS < 1 )); then
      echo "ROLLOUT_GPUS must be >= 1 when rollout backend is enabled." >&2
      exit 1
    fi
    if (( TRAIN_GPUS + ROLLOUT_GPUS > GPUS )); then
      echo "TRAIN_GPUS + ROLLOUT_GPUS cannot exceed total GPUS." >&2
      exit 1
    fi
    if [[ -z "${ROLLOUT_VLLM_TP+x}" ]]; then
      ROLLOUT_VLLM_TP=$ROLLOUT_GPUS
    fi
    if (( ROLLOUT_VLLM_TP > ROLLOUT_GPUS )); then
      echo "[WARN] ROLLOUT_VLLM_TP=${ROLLOUT_VLLM_TP} exceeds ROLLOUT_GPUS=${ROLLOUT_GPUS}; clamping TP to rollout GPU count." >&2
      ROLLOUT_VLLM_TP=$ROLLOUT_GPUS
    fi
    if (( ROLLOUT_VLLM_TP < ROLLOUT_GPUS )); then
      echo "[WARN] ROLLOUT_VLLM_TP=${ROLLOUT_VLLM_TP} is smaller than ROLLOUT_GPUS=${ROLLOUT_GPUS}; some rollout GPUs will stay idle." >&2
    fi
    if [[ -z "${ROLLOUT_VISIBLE_DEVICES:-}" ]]; then
      ROLLOUT_VISIBLE_DEVICES=$(_take_last_visible_devices "$(_visible_device_list)" "$ROLLOUT_GPUS")
    fi
  fi

  if [[ "$ROLLOUT_BACKEND" == "vllm_ray" ]]; then
    export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
    if ! _vllm_ray_rollout_preflight; then
      echo "[ERROR] vllm_ray rollout preflight failed; refusing to silently fall back to DeepSpeed." >&2
      return 1
    fi
  fi

  echo "TRAIN_GPUS=${TRAIN_GPUS}"
  echo "ROLLOUT_BACKEND=${ROLLOUT_BACKEND}"
  echo "ROLLOUT_GPUS=${ROLLOUT_GPUS}"
  echo "ROLLOUT_VLLM_TP=${ROLLOUT_VLLM_TP}"
  echo "ROLLOUT_OBJECTIVE=${rollout_objective:-<unset>}"
  echo "ROLLOUT_VISIBLE_DEVICES=${ROLLOUT_VISIBLE_DEVICES:-<empty>}"
  echo "RAY_ENABLE_UV_RUN_RUNTIME_ENV=${RAY_ENABLE_UV_RUN_RUNTIME_ENV:-<empty>}"
}

run_step() {
  local label="$1"
  shift

  if _step_completed "$label"; then
    echo "[SKIP] $label"
    return 0
  fi

  echo "[RUN ] $label"
  "$@"
  _mark_step_done "$label" success
  echo "[DONE] $label"
}

run_train_step() {
  local label="$1"
  shift
  run_step "$label" torchrun --standalone --nnodes=1 --nproc-per-node="${TRAIN_GPUS:-$GPUS}" "$SRC_ROOT/train.py" "$@"
}

run_python_step() {
  local label="$1"
  shift
  run_step "$label" python "$@"
}

run_distributed_python_step() {
  local label="$1"
  shift
  run_step "$label" torchrun --standalone --nnodes=1 --nproc-per-node="$GPUS" "$@"
}

merge_run_samples_step() {
  local label="$1"
  local run_dir="$2"
  local dataset="$3"
  local model_name="$4"

  run_python_step "$label" \
    "$SRC_ROOT/scripts/merge_sample_outputs.py" \
    --run_dir "$run_dir" \
    --dataset "$dataset" \
    --model_name "$model_name"
}
