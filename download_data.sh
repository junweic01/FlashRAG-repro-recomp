#!/bin/bash

export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda activate flashrag

cd /home/ec2-user/flashrag
mkdir -p ./models/
mkdir -p ./recomp/

# Check if HF_TOKEN is set in environment
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set."
    echo "Please set it before running this script:"
    echo "  export HF_TOKEN='your_token_here'"
    exit 1
fi
hf download meta-llama/Meta-Llama-3-8B-Instruct --local-dir ./models/llama3-8b-instruct --token "$HF_TOKEN"
hf download intfloat/e5-base-v2 --local-dir ./models/e5-base-v2
hf download RUC-NLPIR/FlashRAG_datasets   --repo-type dataset   --local-dir ./datasets
hf download fangyuan/nq_abstractive_compressor --local-dir ./recomp/nq
hf download fangyuan/tqa_abstractive_compressor --local-dir ./recomp/tqa
hf download fangyuan/hotpotqa_abstractive --local-dir ./recomp/hotpotqa

cd /home/ec2-user/flashrag/datasets/retrieval-corpus/
unzip wiki18_100w.zip

# total 21015324 lines
head -n 525383 wiki18_100w.jsonl > wiki18_100w_1_40.jsonl