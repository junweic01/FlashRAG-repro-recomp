#!/bin/bash
export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# Set your CMU AI Gateway API key here
# You can also set this in your shell environment before running this script
if [ -z "$AI_GATEWAY_API_KEY" ]; then
    echo "ERROR: AI_GATEWAY_API_KEY environment variable not set!"
    echo "Please set it with: export AI_GATEWAY_API_KEY='your-key-here'"
    exit 1
fi

export OPENAI_API_KEY="$AI_GATEWAY_API_KEY"
export OPENAI_BASE_URL="https://ai-gateway.andrew.cmu.edu/v1"

conda activate flashrag

echo "Starting Claude Refiner Experiment..."
echo "Using model: claude-sonnet-4-20250514-v1:0"
echo "Dataset: nq (Natural Questions)"
echo "GPU(s): 0"

# Index should already exist at: /home/ec2-user/flashrag/FlashRAG-repro-recomp/indexes/e5_Flat.index
if [ ! -f "indexes/e5_Flat.index" ]; then
    echo "ERROR: Index not found at indexes/e5_Flat.index"
    echo "Please build the index first using your RECOMP setup."
    exit 1
fi

# 2. Run Experiment
echo "Running experiment..."
cd examples/methods
python run_exp.py --method_name 'claude' --split 'dev' --dataset_name 'hotpotqa' --gpu_id '0'

echo "Experiment finished."
echo "Results saved to: /home/ec2-user/flashrag/FlashRAG-repro-recomp/output/"
