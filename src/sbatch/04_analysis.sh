#!/bin/bash
#SBATCH --job-name=par-analysis
#SBATCH --partition=batch
#SBATCH --gres=gpu:h200:4
## Cluster-specific: uncomment and edit for your own node names.
##SBATCH --nodelist=node-[0-7]
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=30-00:00:00
#SBATCH --output=logs/par-analysis-%j.log
#SBATCH --error=logs/par-analysis-%j.error

set -eo pipefail
set +u
source ~/.bashrc
conda activate "${CONDA_ENV:-par}"
set -u
if [[ -n "${PROJECT_ROOT:-}" && -f "${PROJECT_ROOT}/src/sbatch/_common.sh" ]]; then
  SRC_ROOT="${PROJECT_ROOT}/src"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/src/sbatch/_common.sh" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
  SRC_ROOT="${PROJECT_ROOT}/src"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -f "${SCRIPT_DIR}/_common.sh" ]]; then
    SRC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SRC_ROOT/.." && pwd)}"
  elif git_root=$(git rev-parse --show-toplevel 2>/dev/null) && [[ -f "${git_root}/src/sbatch/_common.sh" ]]; then
    PROJECT_ROOT="$git_root"
    SRC_ROOT="${PROJECT_ROOT}/src"
  else
    echo "Cannot locate src/sbatch/_common.sh. Set PROJECT_ROOT=/path/to/par before sbatch." >&2
    exit 1
  fi
fi
source "$SRC_ROOT/sbatch/_common.sh"

BASE_MODEL=gemma2-2b
BASE_DATASET=hh_rlhf
ANALYSIS_OBJECTIVE=analysis_ppo
PIPELINE_NAME=04_analysis

AUTO_BATCH_SIZE=0
init_pipeline

# Match src/sbatch/02_rl_matrix_part1.sh for ppo_gemma2-2b_hh_rlhf.
ROLLOUT_BACKEND=deepspeed
ROLLOUT_SYNC_INTERVAL_STEPS=${ROLLOUT_SYNC_INTERVAL_STEPS:-1}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.9}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-true}
TRAIN_GPUS=${TRAIN_GPUS:-$GPUS}
ROLLOUT_GPUS=0
ROLLOUT_VLLM_TP=1
ROLLOUT_VISIBLE_DEVICES=
configure_rollout_layout "$ANALYSIS_OBJECTIVE"

policy_run_dir="$EXP_RUNS_DIR/sft_${BASE_MODEL}_${BASE_DATASET}"
vanilla_reward_run_dir="$EXP_RUNS_DIR/reward_${BASE_MODEL}_${BASE_DATASET}"
warm_reward_run_dir="${vanilla_reward_run_dir}_warm"
analysis_odin_reward_run_dir="$EXP_RUNS_DIR/analysis_reward_odin"
analysis_reg_reward_run_dir="$EXP_RUNS_DIR/analysis_reward_reg"

policy_dir="${policy_run_dir}/final_hf"
reference_dir="$policy_dir"
vanilla_reward_dir="${vanilla_reward_run_dir}/final_hf"
warm_reward_dir="${warm_reward_run_dir}/final_hf"
analysis_odin_reward_dir="${analysis_odin_reward_run_dir}/final_hf"
analysis_reg_reward_dir="${analysis_reg_reward_run_dir}/final_hf"

train_batch_size_per_gpu=$(train_batch_size_per_gpu_for "$BASE_MODEL" "$ANALYSIS_OBJECTIVE")
eval_batch_size_per_gpu=$(eval_batch_size_per_gpu_for "$BASE_MODEL" "$ANALYSIS_OBJECTIVE")
reward_train_batch_size_per_gpu=$(train_batch_size_per_gpu_for "$BASE_MODEL" reward)
reward_eval_batch_size_per_gpu=$(eval_batch_size_per_gpu_for "$BASE_MODEL" reward)
rollout_max_batch_prompts=${ROLLOUT_MAX_BATCH_PROMPTS:-$(rollout_max_batch_prompts_for "$train_batch_size_per_gpu" "$eval_batch_size_per_gpu")}
rollout_max_batch_tokens=${ROLLOUT_MAX_BATCH_TOKENS:-$(rollout_max_batch_tokens_for "$rollout_max_batch_prompts" 1212)}

POLICY_LR=${POLICY_LR:-6e-7}
CRITIC_LR=${CRITIC_LR:-6e-6}
REWARD_LR=${REWARD_LR:-5e-6}
BASE_KL_COEF=${BASE_KL_COEF:-0.005}
BASE_RL_EPOCHS=${BASE_RL_EPOCHS:-0.8}
RL_BUFFER_SIZE=${RL_BUFFER_SIZE:-8}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}

echo "[PLAN] analysis=${ANALYSIS_OBJECTIVE} model=${BASE_MODEL} dataset=${BASE_DATASET} train_batch_size_per_gpu=${train_batch_size_per_gpu} eval_batch_size_per_gpu=${eval_batch_size_per_gpu} policy_lr=${POLICY_LR} critic_lr=${CRITIC_LR} KL_coef=${BASE_KL_COEF} buffer_size=${RL_BUFFER_SIZE} rollout_max_batch_prompts=${rollout_max_batch_prompts} rollout_max_batch_tokens=${rollout_max_batch_tokens}"

analysis_reward_train() {
  local label="$1"
  local exp_name="$2"
  shift 2

  if step_pending "$label"; then
    run_train_step "$label" \
      --model_name "$BASE_MODEL" \
      --dataset "$BASE_DATASET" \
      --exp_name "$exp_name" \
      --train_batch_size_per_gpu "$reward_train_batch_size_per_gpu" \
      --eval_batch_size_per_gpu "$reward_eval_batch_size_per_gpu" \
      --eval_splits_per_epoch 5 \
      --n_epochs 1 \
      --learning_rate "$REWARD_LR" \
      --save_every_eval false \
      --save_final_checkpoint true \
      "$@"
  else
    echo "[SKIP] $label"
  fi
}

analysis_reward_train \
  "analysis_reward:ODIN" \
  "analysis_reward_odin" \
  --loss_name reward_odin \
  --reward_odin true \
  --reward_odin_L 1.0 \
  --reward_odin_O 1.0

analysis_reward_train \
  "analysis_reward:Reg" \
  "analysis_reward_reg" \
  --loss_name reward \
  --reward_reg true \
  --reward_reg_val "${REWARD_REG_VAL:-0.005}"

declare -a CALIBRATION_RUNS=()
declare -a CALIBRATION_LABELS=()

register_calibration_run() {
  CALIBRATION_RUNS+=("$1")
  CALIBRATION_LABELS+=("$2")
}

analysis_train() {
  local exp_name="$1"
  shift

  local reward_dir="$vanilla_reward_dir"
  local critic_dir="$vanilla_reward_dir"
  local reward_odin="false"
  local calibration_label=""
  local extra_args=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --analysis_reward_dir)
        reward_dir="$2"
        shift 2
        ;;
      --analysis_critic_dir)
        critic_dir="$2"
        shift 2
        ;;
      --analysis_reward_odin)
        reward_odin="$2"
        shift 2
        ;;
      --analysis_calibration_label)
        calibration_label="$2"
        shift 2
        ;;
      *)
        extra_args+=("$1")
        shift
        ;;
    esac
  done

  if step_pending "analysis:${exp_name}"; then
    run_train_step "analysis:${exp_name}" \
      --loss_name ppo \
      --model_name "$BASE_MODEL" \
      --dataset "$BASE_DATASET" \
      --exp_name "$exp_name" \
      --policy_path "$policy_dir" \
      --reference_path "$reference_dir" \
      --reward_path "$reward_dir" \
      --critic_path "$critic_dir" \
      --reward_odin "$reward_odin" \
      --sample_ontest \
      --eval_splits_per_epoch 10 \
      --n_epochs "$BASE_RL_EPOCHS" \
      --KL_coef "$BASE_KL_COEF" \
      --train_batch_size_per_gpu "$train_batch_size_per_gpu" \
      --eval_batch_size_per_gpu "$eval_batch_size_per_gpu" \
      --learning_rate "$POLICY_LR" \
      --critic_lr "$CRITIC_LR" \
      --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
      --buffer_size "$RL_BUFFER_SIZE" \
      --rollout_backend "$ROLLOUT_BACKEND" \
      --rollout_visible_devices "${ROLLOUT_VISIBLE_DEVICES:-}" \
      --rollout_vllm_tensor_parallel_size "$ROLLOUT_VLLM_TP" \
      --rollout_sync_interval_steps "$ROLLOUT_SYNC_INTERVAL_STEPS" \
      --rollout_gpu_memory_utilization "$ROLLOUT_GPU_MEMORY_UTILIZATION" \
      --rollout_enforce_eager "$ROLLOUT_ENFORCE_EAGER" \
      --rollout_max_batch_prompts "$rollout_max_batch_prompts" \
      --rollout_max_batch_tokens "$rollout_max_batch_tokens" \
      --disable_checkpoint_saving true \
      --save_final_checkpoint false \
      "${extra_args[@]}"
  else
    echo "[SKIP] analysis:${exp_name}"
  fi

  merge_run_samples_step "merge:${exp_name}" "$EXP_RUNS_DIR/$exp_name" "$BASE_DATASET" "$BASE_MODEL"

  if [[ -n "$calibration_label" ]]; then
    register_calibration_run "$EXP_RUNS_DIR/$exp_name" "$calibration_label"
  fi
}

analysis_train "analysis_suite_Vanilla" \
  --reward_shaping vanilla \
  --analysis_calibration_label "Vanilla"

analysis_train "analysis_suite_WARM" \
  --reward_shaping vanilla \
  --analysis_reward_dir "$warm_reward_dir" \
  --analysis_critic_dir "$warm_reward_dir" \
  --analysis_calibration_label "WARM"

analysis_train "analysis_suite_ODIN" \
  --reward_shaping vanilla \
  --analysis_reward_dir "$analysis_odin_reward_dir" \
  --analysis_critic_dir "$analysis_odin_reward_dir" \
  --analysis_reward_odin true \
  --analysis_calibration_label "ODIN"

analysis_train "analysis_suite_Reg" \
  --reward_shaping vanilla \
  --analysis_reward_dir "$analysis_reg_reward_dir" \
  --analysis_critic_dir "$analysis_reg_reward_dir" \
  --analysis_calibration_label "Reg"

analysis_train "analysis_suite_Meanstd" \
  --reward_shaping meanstd \
  --analysis_calibration_label "Meanstd"

analysis_train "analysis_suite_Clip" \
  --reward_shaping clip \
  --analysis_calibration_label "Clip"

analysis_train "analysis_suite_Minmax" \
  --reward_shaping minmax \
  --analysis_calibration_label "Minmax"

analysis_train "analysis_suite_LSC" \
  --reward_shaping lsc \
  --analysis_calibration_label "LSC"

analysis_train "analysis_suite_PAR" \
  --reward_shaping par \
  --analysis_calibration_label "PAR"


for kl in 0.01 0.05 0.1; do
  analysis_train "analysis_principle1_kl${kl}" --KL_coef "$kl" --reward_shaping vanilla
done

for ceil in 5 4 3; do
  analysis_train "analysis_principle1_ceil${ceil}" --reward_shaping vanilla --reward_ceil "$ceil"
done

for shape in tanh fittedpoly sigmoid sigmoidk2 sigmoidk3; do
  analysis_train "analysis_principle2_${shape}_centered" --reward_shaping "$shape" --reward_centered true
  analysis_train "analysis_principle2_${shape}_uncentered" --reward_shaping "$shape" --reward_centered false
done

for num_refs in 1 3 5; do
  analysis_train "analysis_dataeffi_num${num_refs}" \
    --reward_shaping par \
    --reward_maxref "$num_refs" \
    --num_refs "$num_refs"
done

analysis_train "analysis_robust_PAR" \
  --reward_shaping par \
  --n_epochs 2

analysis_train "analysis_robust_LSC" \
  --reward_shaping lsc \
  --n_epochs 2

analysis_train "analysis_robust_MINMAX" \
  --reward_shaping minmax \
  --n_epochs 2

if [[ "${RUN_ANALYSIS_CALIBRATION:-0}" == "1" ]]; then
  calibration_args=(
    --output "$PROJECT_ROOT/results/analysis_preference_calibration.png"
    --summary_output "$PROJECT_ROOT/results/analysis_preference_calibration.json"
  )
  for idx in "${!CALIBRATION_RUNS[@]}"; do
    calibration_args+=(--run "${CALIBRATION_LABELS[$idx]}=${CALIBRATION_RUNS[$idx]}")
  done
  run_python_step "analysis:calibration" "$SRC_ROOT/scripts/plot_preference_calibration.py" "${calibration_args[@]}"
else
  echo "[INFO] Set RUN_ANALYSIS_CALIBRATION=1 after LLM ratings exist to draw the preference calibration plot."
fi
