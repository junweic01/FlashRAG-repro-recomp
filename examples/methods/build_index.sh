#!/usr/bin/env bash
#SBATCH --job-name=flashrag_build_index
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --gres=gpu:L40S:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hanzhanz@andrew.cmu.edu


NCCL_P2P_DISABLE=1 python -m flashrag.retriever.index_builder \
  --retrieval_method e5 \
  --model_path /data/user_data/hanzhanz//models/e5-base-v2/ \
  --corpus_path /data/user_data/hanzhanz/flashrag_datasets/retrieval-corpus/wiki18_100w.jsonl \
  --save_dir /data/user_data/hanzhanz/flashrag_datasets/retrieval-corpus/indexes/ \
  --use_fp16 \
  --max_length 512 \
  --batch_size 256 \
  --pooling_method mean \
  --faiss_type Flat
