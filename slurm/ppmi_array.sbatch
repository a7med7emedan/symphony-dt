#!/bin/bash
#SBATCH --job-name=symphony_dt_shard
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-19

set -euo pipefail

CONFIG_PATH=${1:-configs/config.demo.yaml}
NUM_SHARDS=${2:-20}

mkdir -p logs

symphony-dt run \
  --config "${CONFIG_PATH}" \
  --mode shard \
  --shard-id "${SLURM_ARRAY_TASK_ID}" \
  --num-shards "${NUM_SHARDS}"
