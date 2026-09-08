#!/bin/bash
#SBATCH --job-name=par-download-models
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/par-download-models-%j.log
#SBATCH --error=logs/par-download-models-%j.error

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


cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/models_ck"

# Gemma and Llama are gated on the Hugging Face Hub: export HF_TOKEN first.
# Override with e.g. MODELS="gemma2_2b qwen3_4b_base" to fetch a subset.
MODELS="${MODELS:-gemma2_2b gemma2_9b qwen3_4b_base llama_3_1_8b}"
for model in $MODELS; do
  echo "[RUN ] download:${model}"
  python "$SRC_ROOT/scripts/download_${model}.py"
done
