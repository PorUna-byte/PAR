#!/bin/bash
#SBATCH --job-name=par-supervised
#SBATCH --partition=batch
#SBATCH --gres=gpu:h200:4
## Cluster-specific: uncomment and edit for your own node names.
##SBATCH --nodelist=node-[0-7]
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=30-00:00:00
#SBATCH --output=logs/par-supervised-%j.log
#SBATCH --error=logs/par-supervised-%j.error

set -eo pipefail
set +u
source ~/.bashrc
conda activate "${CONDA_ENV:-par}"
set -u

mkdir -p $HOME/tmp
mkdir -p $HOME/torch_extensions
chmod 700 $HOME/tmp

export TMPDIR=$HOME/tmp
export TEMP=$HOME/tmp
export TMP=$HOME/tmp
export TORCH_EXTENSIONS_DIR=$HOME/torch_extensions

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

PIPELINE_NAME=01_supervised_pipeline

init_pipeline
TRAIN_GPUS=$GPUS
ROLLOUT_GPUS=0
ROLLOUT_BACKEND=deepspeed
ROLLOUT_VLLM_TP=1
ROLLOUT_VISIBLE_DEVICES=
REFERENCE_GENERATION_BACKEND=${REFERENCE_GENERATION_BACKEND:-vllm}
REFERENCE_VLLM_GPU_MEMORY_UTILIZATION=${REFERENCE_VLLM_GPU_MEMORY_UTILIZATION:-0.8}
REFERENCE_VLLM_ENFORCE_EAGER=${REFERENCE_VLLM_ENFORCE_EAGER:-true}
REFERENCE_VLLM_DTYPE=${REFERENCE_VLLM_DTYPE:-bfloat16}
REFERENCE_VLLM_SHARE_REWARD_GPU=${REFERENCE_VLLM_SHARE_REWARD_GPU:-true}
WARM_CHECKPOINT_COUNT=${WARM_CHECKPOINT_COUNT:-5}
WARM_REWARD_EVAL_SPLITS_PER_EPOCH=${WARM_REWARD_EVAL_SPLITS_PER_EPOCH:-$((WARM_CHECKPOINT_COUNT))}
echo "WARM_CHECKPOINT_COUNT=$WARM_CHECKPOINT_COUNT"
echo "WARM_REWARD_EVAL_SPLITS_PER_EPOCH=$WARM_REWARD_EVAL_SPLITS_PER_EPOCH"

for model in "${MODELS[@]}"; do
  attn_impl=$(attention_impl_for_model "$model")

  base_sft_train_batch_size_per_gpu=$(train_batch_size_per_gpu_for "$model" sft)
  base_sft_eval_batch_size_per_gpu=$(eval_batch_size_per_gpu_for "$model" sft)
  base_reward_train_batch_size_per_gpu=$(default_batch_size_per_gpu_for train reward "$model")
  base_reward_eval_batch_size_per_gpu=$(default_batch_size_per_gpu_for eval reward "$model")
  base_reference_batch_size_per_gpu=$(inference_batch_size_per_gpu_for "$model" reference)
  reference_batch_size_per_gpu=${REFERENCE_BATCH_SIZE_PER_GPU:-$base_reference_batch_size_per_gpu}
  reference_refs_per_generation_call=${REFERENCE_REFS_PER_GENERATION_CALL:-5}
  reference_share_reward_gpu=false
  if [[ "$REFERENCE_VLLM_SHARE_REWARD_GPU" == "1" || "$REFERENCE_VLLM_SHARE_REWARD_GPU" == "true" || "$REFERENCE_VLLM_SHARE_REWARD_GPU" == "yes" ]]; then
    reference_share_reward_gpu=true
  fi
  reference_leave_free_gpus=1
  if [[ "$reference_share_reward_gpu" == "true" ]]; then
    reference_leave_free_gpus=0
  fi
  if [[ "$REFERENCE_GENERATION_BACKEND" == "vllm" ]]; then
    if (( GPUS > 1 )); then
      reference_vllm_tp_auto=$(vllm_tensor_parallel_size_for_model "$model" "$GPUS" "$reference_leave_free_gpus")
      reference_vllm_tp_default="${REFERENCE_VLLM_TP_DEFAULT:-$reference_vllm_tp_auto}"
      if (( reference_vllm_tp_default >= GPUS )); then
        reference_reward_device_index_default=$((GPUS - 1))
      else
        reference_reward_device_index_default="$reference_vllm_tp_default"
      fi
    else
      reference_vllm_tp_default=1
      reference_reward_device_index_default=0
    fi
  else
    reference_vllm_tp_default=1
    reference_reward_device_index_default=0
  fi
  reference_vllm_tp=${REFERENCE_VLLM_TP:-$reference_vllm_tp_default}
  if [[ "$REFERENCE_GENERATION_BACKEND" == "vllm" && "$reference_share_reward_gpu" != "true" && "$GPUS" -gt 1 && "$reference_vllm_tp" -ge "$GPUS" ]]; then
    echo "[WARN] reference_vllm_tp=${reference_vllm_tp} leaves no dedicated GPU for reward scoring; clamping to $((GPUS - 1))."
    reference_vllm_tp=$((GPUS - 1))
  fi
  if [[ "$REFERENCE_GENERATION_BACKEND" == "vllm" ]]; then
    reference_attention_heads=$(num_attention_heads_for_model "$model")
    if (( reference_attention_heads > 0 && reference_attention_heads % reference_vllm_tp != 0 )); then
      valid_reference_vllm_tp=$(vllm_tensor_parallel_size_for_model "$model" "$GPUS" "$reference_leave_free_gpus")
      echo "[WARN] reference_vllm_tp=${reference_vllm_tp} is invalid for ${model} (${reference_attention_heads} attention heads); using ${valid_reference_vllm_tp}."
      reference_vllm_tp="$valid_reference_vllm_tp"
    fi
    if (( reference_reward_device_index_default >= GPUS || reference_reward_device_index_default < reference_vllm_tp )); then
      if (( reference_vllm_tp >= GPUS )); then
        reference_reward_device_index_default=$((GPUS - 1))
      else
        reference_reward_device_index_default="$reference_vllm_tp"
      fi
    fi
  fi
  reference_reward_device_index=${REFERENCE_REWARD_DEVICE_INDEX:-$reference_reward_device_index_default}
  reference_vllm_extra_args=()
  if [[ "$REFERENCE_VLLM_ENFORCE_EAGER" == "1" || "$REFERENCE_VLLM_ENFORCE_EAGER" == "true" || "$REFERENCE_VLLM_ENFORCE_EAGER" == "yes" ]]; then
    reference_vllm_extra_args+=(--vllm_enforce_eager)
  fi

  for dataset in "${DATASETS[@]}"; do
    sft_dir="$EXP_RUNS_DIR/sft_${model}_${dataset}"
    reward_dir="$EXP_RUNS_DIR/reward_${model}_${dataset}"
    warm_dir="${reward_dir}_warm"
    sft_train_batch_size_per_gpu="$base_sft_train_batch_size_per_gpu"
    sft_eval_batch_size_per_gpu="$base_sft_eval_batch_size_per_gpu"
    reward_train_batch_size_per_gpu="$base_reward_train_batch_size_per_gpu"
    reward_eval_batch_size_per_gpu="$base_reward_eval_batch_size_per_gpu"

    if step_pending "sft:${model}:${dataset}"; then
      run_train_step "sft:${model}:${dataset}" \
        --loss_name sft \
        --model_name "$model" \
        --dataset "$dataset" \
        --exp_name "sft_${model}_${dataset}" \
        --train_batch_size_per_gpu "$sft_train_batch_size_per_gpu" \
        --eval_batch_size_per_gpu "$sft_eval_batch_size_per_gpu" \
        --n_epochs 2 \
        --learning_rate 5e-6
    else
      echo "[SKIP] sft:${model}:${dataset}"
    fi

    if step_pending "reward:${model}:${dataset}"; then
      run_train_step "reward:${model}:${dataset}" \
        --loss_name reward \
        --model_name "$model" \
        --dataset "$dataset" \
        --exp_name "reward_${model}_${dataset}" \
        --save_every_eval true \
        --train_batch_size_per_gpu "$reward_train_batch_size_per_gpu" \
        --eval_batch_size_per_gpu "$reward_eval_batch_size_per_gpu" \
        --eval_splits_per_epoch "$WARM_REWARD_EVAL_SPLITS_PER_EPOCH" \
        --n_epochs 1 \
        --learning_rate 5e-6
    else
      echo "[SKIP] reward:${model}:${dataset}"
    fi

    if step_pending "warm_merge:${model}:${dataset}"; then
      run_python_step "warm_merge:${model}:${dataset}" \
        "$SRC_ROOT/merge/merge_rm_checkpoints.py" \
        --reward_run_dir "$reward_dir" \
        --output_dir "$warm_dir" \
        --count "$WARM_CHECKPOINT_COUNT" \
        --attn_impl "$attn_impl"
    else
      echo "[SKIP] warm_merge:${model}:${dataset}"
    fi

    if step_pending "reference:${model}:${dataset}"; then
      reference_reward_batch_size_per_gpu=${REFERENCE_REWARD_BATCH_SIZE_PER_GPU:-$((reference_batch_size_per_gpu * reference_refs_per_generation_call))}
      reference_vllm_max_batch_prompts=${REFERENCE_VLLM_MAX_BATCH_PROMPTS:-$((reference_batch_size_per_gpu * reference_vllm_tp))}
      reference_vllm_max_batch_tokens=${REFERENCE_VLLM_MAX_BATCH_TOKENS:-$((reference_vllm_max_batch_prompts * 1212))}

      echo "[PLAN] model=${model} dataset=${dataset} sft(train_batch_size_per_gpu=${sft_train_batch_size_per_gpu},eval_batch_size_per_gpu=${sft_eval_batch_size_per_gpu}) reward(train_batch_size_per_gpu=${reward_train_batch_size_per_gpu},eval_batch_size_per_gpu=${reward_eval_batch_size_per_gpu}) reference(backend=${REFERENCE_GENERATION_BACKEND},batch_size_per_gpu=${reference_batch_size_per_gpu},refs_per_generation_call=${reference_refs_per_generation_call},reward_batch_size_per_gpu=${reference_reward_batch_size_per_gpu},vllm_tp=${reference_vllm_tp},reward_device_index=${reference_reward_device_index},share_reward_gpu=${reference_share_reward_gpu},vllm_gpu_memory_utilization=${REFERENCE_VLLM_GPU_MEMORY_UTILIZATION},vllm_max_batch_prompts=${reference_vllm_max_batch_prompts},vllm_max_batch_tokens=${reference_vllm_max_batch_tokens})"

      if [[ "$REFERENCE_GENERATION_BACKEND" == "vllm" ]]; then
        run_python_step "reference:${model}:${dataset}" \
          "$SRC_ROOT/scripts/generate_reference_responses.py" \
          --dataset "$dataset" \
          --model_name "$model" \
          --model_path "$sft_dir/final_hf" \
          --reward_model_path "$reward_dir/final_hf" \
          --batch_size_per_gpu "$reference_batch_size_per_gpu" \
          --refs_per_generation_call "$reference_refs_per_generation_call" \
          --reward_batch_size_per_gpu "$reference_reward_batch_size_per_gpu" \
          --num_refs 5 \
          --generation_backend vllm \
          --vllm_tensor_parallel_size "$reference_vllm_tp" \
          --vllm_dtype "$REFERENCE_VLLM_DTYPE" \
          --vllm_gpu_memory_utilization "$REFERENCE_VLLM_GPU_MEMORY_UTILIZATION" \
          --vllm_max_batch_prompts "$reference_vllm_max_batch_prompts" \
          --vllm_max_batch_tokens "$reference_vllm_max_batch_tokens" \
          "${reference_vllm_extra_args[@]}" \
          --reward_device_index "$reference_reward_device_index"
      else
        run_distributed_python_step "reference:${model}:${dataset}" \
          "$SRC_ROOT/scripts/generate_reference_responses.py" \
          --dataset "$dataset" \
          --model_name "$model" \
          --model_path "$sft_dir/final_hf" \
          --reward_model_path "$reward_dir/final_hf" \
          --batch_size_per_gpu "$reference_batch_size_per_gpu" \
          --refs_per_generation_call "$reference_refs_per_generation_call" \
          --reward_batch_size_per_gpu "$reference_reward_batch_size_per_gpu" \
          --num_refs 5 \
          --generation_backend hf
      fi
    else
      echo "[SKIP] reference:${model}:${dataset}"
    fi

  done
done
