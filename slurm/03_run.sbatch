#!/bin/bash
#SBATCH --job-name=ids_run
#SBATCH --output=logs/run_%A_%a.out
#SBATCH --error=logs/run_%A_%a.err
#SBATCH --partition=shortq        # adjust to your cluster
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# activate your environment; adjust this line to your setup
source "${IDS_VENV:-$HOME/envs/idsrerun}/bin/activate"

# ---------------------------------------------------------------------------
# CPU-ONLY, DELIBERATELY.
#
# The manuscript reports CPU timings and compares ML against DL compute cost
# (sections 4.3-4.4). Letting TensorFlow use the H100 MIG slice would make the
# DL timings incomparable to both the ML timings and the originally submitted
# results, turning a like-for-like correction into a hardware change. Remove
# this line only if you also rewrite the timing discussion.
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTRAOP_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTEROP_THREADS=2
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1

ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" manifest.txt)
if [[ -z "$ARGS" ]]; then echo "no task at line $SLURM_ARRAY_TASK_ID"; exit 0; fi

echo "task $SLURM_ARRAY_TASK_ID on $(hostname): $ARGS"
python run_group.py $ARGS
