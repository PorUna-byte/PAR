#!/bin/bash
#SBATCH --job-name=par-rl2
#SBATCH --partition=batch
#SBATCH --gres=gpu:h200:4
## Cluster-specific: uncomment and edit for your own node names.
##SBATCH --nodelist=node-[0-7]
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=30-00:00:00
#SBATCH --output=logs/par-rl2-%j.log
#SBATCH --error=logs/par-rl2-%j.error

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

MODELS=(gemma2-2b gemma2-9b qwen3-4b-base llama-3.1-8b)
DATASETS=(hh_rlhf ultrafb_bin)
ALGOS=(grpo dpo)
# SHAPINGS=(vanilla warm meanstd par)
SHAPINGS=(vanilla par)
PIPELINE_NAME=03_rl_matrix_part2
critic_learning_rate=1e-5
AUTO_BATCH_SIZE=0
init_pipeline
ROLLOUT_BACKEND=deepspeed
ROLLOUT_SYNC_INTERVAL_STEPS=${ROLLOUT_SYNC_INTERVAL_STEPS:-1}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.8}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-true}
USER_TRAIN_GPUS=${TRAIN_GPUS-__unset__}

# ─── Hyperparameter tables: key="${model}:${dataset}" ──────────────────────
# To add dataset-specific values, replace the right-hand side for that key.
declare -A _HP_POLICY_LR=(
  ["gemma2-2b:hh_rlhf"]="${GEMMA2_2B_RL_LR:-2e-6}"              ["gemma2-2b:ultrafb_bin"]="${GEMMA2_2B_RL_LR:-2e-6}"
  ["gemma2-9b:hh_rlhf"]="${GEMMA2_9B_RL_LR:-3e-7}"              ["gemma2-9b:ultrafb_bin"]="${GEMMA2_9B_RL_LR:-5e-7}"
  ["qwen3-4b-base:hh_rlhf"]="${QWEN3_4B_BASE_RL_LR:-4e-6}"      ["qwen3-4b-base:ultrafb_bin"]="${QWEN3_4B_BASE_RL_LR:-4e-6}"
  ["llama-3.1-8b:hh_rlhf"]="${LLAMA3_1_8B_RL_LR:-2e-6}"         ["llama-3.1-8b:ultrafb_bin"]="${LLAMA3_1_8B_RL_LR:-4e-6}"
)
declare -A _HP_EPOCHS=(
  ["gemma2-2b:hh_rlhf"]="0.2"                                    ["gemma2-2b:ultrafb_bin"]="0.2"
  ["gemma2-9b:hh_rlhf"]="${GEMMA2_9B_RL_EPOCHS:-0.1}"           ["gemma2-9b:ultrafb_bin"]="${GEMMA2_9B_RL_EPOCHS:-0.1}"
  ["qwen3-4b-base:hh_rlhf"]="${QWEN3_4B_BASE_RL_EPOCHS:-0.1}"   ["qwen3-4b-base:ultrafb_bin"]="${QWEN3_4B_BASE_RL_EPOCHS:-0.1}"
  ["llama-3.1-8b:hh_rlhf"]="${LLAMA3_1_8B_RL_EPOCHS:-0.1}"      ["llama-3.1-8b:ultrafb_bin"]="${LLAMA3_1_8B_RL_EPOCHS:-0.1}"
)
declare -A _HP_KL_COEF=(
  ["gemma2-2b:hh_rlhf"]="0.005"                                  ["gemma2-2b:ultrafb_bin"]="0.005"
  ["gemma2-9b:hh_rlhf"]="${GEMMA2_9B_RL_KL_COEF:-0.005}"        ["gemma2-9b:ultrafb_bin"]="${GEMMA2_9B_RL_KL_COEF:-0.005}"
  ["qwen3-4b-base:hh_rlhf"]="${QWEN3_4B_BASE_RL_KL_COEF:-0.005}" ["qwen3-4b-base:ultrafb_bin"]="${QWEN3_4B_BASE_RL_KL_COEF:-0.005}"
  ["llama-3.1-8b:hh_rlhf"]="${LLAMA3_1_8B_RL_KL_COEF:-0.005}"   ["llama-3.1-8b:ultrafb_bin"]="${LLAMA3_1_8B_RL_KL_COEF:-0.005}"
)
declare -A _HP_GRAD_ACCUM=(
  ["gemma2-2b:hh_rlhf"]="${GEMMA_RL_GRAD_ACCUM:-1}"              ["gemma2-2b:ultrafb_bin"]="${GEMMA_RL_GRAD_ACCUM:-1}"
  ["gemma2-9b:hh_rlhf"]="${GEMMA2_9B_RL_GRAD_ACCUM:-1}"         ["gemma2-9b:ultrafb_bin"]="${GEMMA2_9B_RL_GRAD_ACCUM:-1}"
  ["qwen3-4b-base:hh_rlhf"]="${QWEN3_4B_BASE_RL_GRAD_ACCUM:-1}" ["qwen3-4b-base:ultrafb_bin"]="${QWEN3_4B_BASE_RL_GRAD_ACCUM:-1}"
  ["llama-3.1-8b:hh_rlhf"]="${LLAMA3_1_8B_RL_GRAD_ACCUM:-1}"    ["llama-3.1-8b:ultrafb_bin"]="${LLAMA3_1_8B_RL_GRAD_ACCUM:-1}"
)

for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for algo in "${ALGOS[@]}"; do
      if [[ "$USER_TRAIN_GPUS" == "__unset__" ]]; then unset TRAIN_GPUS; else TRAIN_GPUS="$USER_TRAIN_GPUS"; fi
      ROLLOUT_BACKEND=deepspeed
      ROLLOUT_GPUS=0
      ROLLOUT_VLLM_TP=1
      ROLLOUT_VISIBLE_DEVICES=
      unset VLLM_RAY_PREFLIGHT
      configure_rollout_layout "$algo"
      base_train_batch_size_per_gpu=$(train_batch_size_per_gpu_for "$model" "$algo")
      base_eval_batch_size_per_gpu=$(eval_batch_size_per_gpu_for "$model" "$algo")
      _key="${model}:${dataset}"
      policy_learning_rate="${_HP_POLICY_LR[$_key]:-}"
      train_epochs="${_HP_EPOCHS[$_key]:-}"
      kl_coef="${_HP_KL_COEF[$_key]:-}"
      gradient_accumulation_steps="${_HP_GRAD_ACCUM[$_key]:-}"
      if [[ -z "$policy_learning_rate" || -z "$train_epochs" ]]; then
        echo "No hyperparameters defined for model=$model dataset=$dataset" >&2
        exit 1
      fi
      skip_initial_eval_ontest="${SKIP_INITIAL_EVAL_ONTEST:-false}"
      policy_run_dir="$EXP_RUNS_DIR/sft_${model}_${dataset}"
      reward_run_dir="$EXP_RUNS_DIR/reward_${model}_${dataset}"
      policy_dir="${policy_run_dir}/final_hf"
      train_batch_size_per_gpu="$base_train_batch_size_per_gpu"
      eval_batch_size_per_gpu="$base_eval_batch_size_per_gpu"
      grpo_sequence_chunk_size=""
      if [[ "$algo" == "grpo" ]]; then
        grpo_sequence_chunk_size=$(grpo_sequence_chunk_size_for "$model")
      fi
      rollout_max_batch_prompts=${ROLLOUT_MAX_BATCH_PROMPTS:-$(rollout_max_batch_prompts_for "$train_batch_size_per_gpu" "$eval_batch_size_per_gpu")}
      rollout_max_batch_tokens=${ROLLOUT_MAX_BATCH_TOKENS:-$(rollout_max_batch_tokens_for "$rollout_max_batch_prompts" 1212)}
      echo "[PLAN] objective=${algo} model=${model} dataset=${dataset} train_batch_size_per_gpu=${train_batch_size_per_gpu} eval_batch_size_per_gpu=${eval_batch_size_per_gpu} grpo_sequence_chunk_size=${grpo_sequence_chunk_size:-n/a} skip_initial_eval_ontest=${skip_initial_eval_ontest} rollout_max_batch_prompts=${rollout_max_batch_prompts} rollout_max_batch_tokens=${rollout_max_batch_tokens}"

      if [[ "$algo" == "dpo" ]]; then
        algo_shapings=(vanilla)
      else
        algo_shapings=("${SHAPINGS[@]}")
      fi

      for shaping in "${algo_shapings[@]}"; do
        current_reward_run_dir="$reward_run_dir"
        [[ "$shaping" == "warm" ]] && current_reward_run_dir="${current_reward_run_dir}_warm"
        reward_dir="${current_reward_run_dir}/final_hf"
        run_dir="$EXP_RUNS_DIR/${algo}_${model}_${dataset}_${shaping}"
        extra_rl_args=(--gradient_accumulation_steps "$gradient_accumulation_steps")
        if [[ "$algo" == "grpo" ]]; then
          extra_rl_args+=(
            --grpo_sequence_chunk_size "$grpo_sequence_chunk_size"
          )
        fi

        if step_pending "rl:${algo}:${model}:${dataset}:${shaping}"; then
          run_train_step "rl:${algo}:${model}:${dataset}:${shaping}" \
            --loss_name "$algo" \
            --model_name "$model" \
            --dataset "$dataset" \
            --reward_shaping "$shaping" \
            --exp_name "${algo}_${model}_${dataset}_${shaping}" \
            --sample_ontest \
            --train_batch_size_per_gpu "$train_batch_size_per_gpu" \
            --eval_batch_size_per_gpu "$eval_batch_size_per_gpu" \
            --eval_splits_per_epoch 10 \
            --n_epochs "$train_epochs" \
            --KL_coef "$kl_coef" \
            --learning_rate "$policy_learning_rate" \
            --critic_lr "$critic_learning_rate" \
            "${extra_rl_args[@]}" \
            --skip_initial_eval_ontest "$skip_initial_eval_ontest" \
            --rollout_backend "$ROLLOUT_BACKEND" \
            --rollout_visible_devices "${ROLLOUT_VISIBLE_DEVICES:-}" \
            --rollout_vllm_tensor_parallel_size "$ROLLOUT_VLLM_TP" \
            --rollout_sync_interval_steps "$ROLLOUT_SYNC_INTERVAL_STEPS" \
            --rollout_gpu_memory_utilization "$ROLLOUT_GPU_MEMORY_UTILIZATION" \
            --rollout_enforce_eager "$ROLLOUT_ENFORCE_EAGER" \
            --rollout_max_batch_prompts "$rollout_max_batch_prompts" \
            --rollout_max_batch_tokens "$rollout_max_batch_tokens" \
            --policy_path "$policy_dir" \
            --reference_path "$policy_dir" \
            --reward_path "$reward_dir" \
            --critic_path "$reward_dir" \
            --disable_checkpoint_saving true \
            --save_final_checkpoint false
        else
          echo "[SKIP] rl:${algo}:${model}:${dataset}:${shaping}"
        fi

        merge_run_samples_step "merge:${algo}:${model}:${dataset}:${shaping}" "$run_dir" "$dataset" "$model"
      done
    done
  done
done
