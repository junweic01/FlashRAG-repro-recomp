#!/bin/bash

# Check for API Key
if [ -z "$AI_GATEWAY_API_KEY" ]; then
    echo "Error: AI_GATEWAY_API_KEY environment variable is not set."
    echo "Please set it using: export AI_GATEWAY_API_KEY=your_key"
    exit 1
fi

echo "Starting Claude Refiner Experiment..."

# 1. Build Index (if not exists)
if [ ! -f "indexes/e5_Flat.index" ]; then
    echo "Building index..."
    mkdir -p indexes
    python -m flashrag.retriever.index_builder \
        --retrieval_method e5 \
        --model_path intfloat/e5-base-v2 \
        --corpus_path examples/quick_start/indexes/general_knowledge.jsonl \
        --save_dir indexes/ \
        --use_fp16 \
        --max_length 512 \
        --batch_size 256 \
        --pooling_method mean \
        --faiss_type Flat
else
    echo "Index already exists. Skipping build."
fi

# 2. Run Experiment
echo "Running experiment..."
cd examples/methods
python run_exp.py --method_name 'claude' --split 'test' --dataset_name 'nq' --gpu_id '0'

echo "Experiment finished."
