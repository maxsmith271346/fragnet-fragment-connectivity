#!/bin/bash
#SBATCH --job-name=fragnet-z250k-all
#SBATCH --time=3-00:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --array=0-47%4
#SBATCH --output=%x.%A_%a.out
#SBATCH --error=%x.%A_%a.err

set -euo pipefail

module load GCCcore/12.2.0 Python/3.10.8 Anaconda3

PROJECT_DIR="${FRAGNET_PROJECT_ROOT:-${SCRATCH}/fragnet-fragment-connectivity}"
cd "${PROJECT_DIR}"
source activate fragNet

python3 run_hprc_experiment.py \
    --config experiment/ZINC_full_all.yaml \
    --index "${SLURM_ARRAY_TASK_ID}"
