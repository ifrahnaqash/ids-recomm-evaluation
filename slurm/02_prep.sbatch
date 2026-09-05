#!/bin/bash
#SBATCH --job-name=ids_prep
#SBATCH --output=logs/prep_%A_%a.out
#SBATCH --error=logs/prep_%A_%a.err
#SBATCH --array=1-25
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=shortq        # adjust to your cluster
#SBATCH --time=24:00:00

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# activate your environment; adjust this line to your setup
source "${IDS_VENV:-$HOME/envs/idsrerun}/bin/activate"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

DATASETS=("NSL-KDD" "UNSW-NB15" "CIC-IDS2017" "HIKARI-2021" "HIKARI-2021-noID")
SEEDS=(42 43 44 45 46)

i=$((SLURM_ARRAY_TASK_ID-1))
DS="${DATASETS[$((i / 5))]}"
SEED="${SEEDS[$((i % 5))]}"

echo "prep $DS seed=$SEED on $(hostname)"
python prepare.py --dataset "$DS" --stage prep --seed "$SEED"
