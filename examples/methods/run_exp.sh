#!/usr/bin/env bash
#SBATCH --job-name=flashrag_run_exp
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hanzhanz@andrew.cmu.edu

NCCL_P2P_DISABLE=1 python run_exp.py \
  --method_name "recomp" \
  --split "test" \
  --dataset_name "nq" \
  --gpu_id "0" \
  --config_path "my_config.yaml"
