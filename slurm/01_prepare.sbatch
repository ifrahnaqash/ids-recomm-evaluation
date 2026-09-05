#!/bin/bash
#SBATCH --job-name=ids_clean
#SBATCH --output=logs/clean_%A_%a.out
#SBATCH --error=logs/clean_%A_%a.err
#SBATCH --array=1-5
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=shortq        # adjust to your cluster
#SBATCH --time=12:00:00

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# activate your environment; adjust this line to your setup
source "${IDS_VENV:-$HOME/envs/idsrerun}/bin/activate"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

DATASETS=("NSL-KDD" "UNSW-NB15" "CIC-IDS2017" "HIKARI-2021" "HIKARI-2021-noID")
DS="${DATASETS[$((SLURM_ARRAY_TASK_ID-1))]}"

echo "cleaning $DS on $(hostname)"
python prepare.py --dataset "$DS" --stage clean
