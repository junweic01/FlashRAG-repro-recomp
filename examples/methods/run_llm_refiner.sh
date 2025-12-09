#!/usr/bin/env bash
#SBATCH --job-name=flashrag_llm_refiner_tqa
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --gres=gpu:L40S:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hanzhanz@andrew.cmu.edu

python run_exp.py \
  --method_name "llm-refiner" \
  --config_path "my_config_llm_refiner.yaml" \
  --split "test" \
  --dataset_name "triviaqa" \
  --gpu_id "0,1,2,3" \
  --refiner_batch_size 16 \
  --refiner_max_input_length 2048 \
  --refiner_max_output_length 1024 \
  --refiner_model_path "/data/user_data/hanzhanz/models/llama3-8b-instruct" \
