
from __future__ import annotations

from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
SBATCH_DIR = SRC_ROOT / "sbatch"


def main() -> None:
    script = """#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SRC_ROOT/.." && pwd)}"
source "$SRC_ROOT/sbatch/_common.sh"
configure_pythonpath

for MODEL in gemma2-2b gemma2-9b; do
  for DATASET in ultrafb_bin hh_rlhf; do
    TRAIN_BATCH_SIZE_PER_GPU=$(train_batch_size_per_gpu_for "$MODEL" sft)
    EVAL_BATCH_SIZE_PER_GPU=$(eval_batch_size_per_gpu_for "$MODEL" sft)
    torchrun --standalone --nnodes=1 --nproc-per-node=4 "$SRC_ROOT/train.py" \
      --loss_name sft --model_name "$MODEL" --dataset "$DATASET" \
      --exp_name "sft_${MODEL}_${DATASET}" \
      --train_batch_size_per_gpu "$TRAIN_BATCH_SIZE_PER_GPU" \
      --eval_batch_size_per_gpu "$EVAL_BATCH_SIZE_PER_GPU" \
      --sample_ontest --n_epochs 2 --learning_rate 5e-6

    REWARD_TRAIN_BATCH_SIZE_PER_GPU=$(train_batch_size_per_gpu_for "$MODEL" reward)
    REWARD_EVAL_BATCH_SIZE_PER_GPU=$(eval_batch_size_per_gpu_for "$MODEL" reward)
    torchrun --standalone --nnodes=1 --nproc-per-node=4 "$SRC_ROOT/train.py" \
      --loss_name reward --model_name "$MODEL" --dataset "$DATASET" \
      --exp_name "reward_${MODEL}_${DATASET}" \
      --train_batch_size_per_gpu "$REWARD_TRAIN_BATCH_SIZE_PER_GPU" \
      --eval_batch_size_per_gpu "$REWARD_EVAL_BATCH_SIZE_PER_GPU" \
      --save_every_eval true --eval_splits_per_epoch 5 \
      --n_epochs 1 --learning_rate 5e-6

	    BATCH_SIZE_PER_GPU=$(inference_batch_size_per_gpu_for "$MODEL" reference)
	    REFERENCE_VLLM_TP="${REFERENCE_VLLM_TP:-$(vllm_tensor_parallel_size_for_model "$MODEL" 4 0)}"
	    if (( REFERENCE_VLLM_TP >= 4 )); then
	      REFERENCE_REWARD_DEVICE_INDEX="${REFERENCE_REWARD_DEVICE_INDEX:-3}"
	    else
	      REFERENCE_REWARD_DEVICE_INDEX="${REFERENCE_REWARD_DEVICE_INDEX:-$REFERENCE_VLLM_TP}"
	    fi
	    REFERENCE_VLLM_MAX_BATCH_PROMPTS=$((BATCH_SIZE_PER_GPU * REFERENCE_VLLM_TP))
	    REFERENCE_VLLM_MAX_BATCH_TOKENS=$((REFERENCE_VLLM_MAX_BATCH_PROMPTS * 1212))
    python "$SRC_ROOT/scripts/generate_reference_responses.py" \
      --dataset "$DATASET" --model_name "$MODEL" \
      --model_path "$PROJECT_ROOT/exp_runs/sft_${MODEL}_${DATASET}/final_hf" \
      --reward_model_path "$PROJECT_ROOT/exp_runs/reward_${MODEL}_${DATASET}/final_hf" \
      --batch_size_per_gpu "$BATCH_SIZE_PER_GPU" \
      --generation_backend vllm \
	      --vllm_tensor_parallel_size "$REFERENCE_VLLM_TP" \
	      --vllm_dtype "${REFERENCE_VLLM_DTYPE:-bfloat16}" \
	      --vllm_gpu_memory_utilization "${REFERENCE_VLLM_GPU_MEMORY_UTILIZATION:-0.8}" \
	      --vllm_enforce_eager \
	      --vllm_max_batch_prompts "$REFERENCE_VLLM_MAX_BATCH_PROMPTS" \
	      --vllm_max_batch_tokens "$REFERENCE_VLLM_MAX_BATCH_TOKENS" \
	      --reward_device_index "$REFERENCE_REWARD_DEVICE_INDEX"
  done
done
"""
    output_path = SBATCH_DIR / "generated_commands.sh"
    output_path.write_text(script)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
