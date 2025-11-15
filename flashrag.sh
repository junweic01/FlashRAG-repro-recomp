#!/bin/bash
sudo yum install -y git
sudo dnf install -y htop

# Setup GPU drivers if GPU is present
./setup_gpu.sh

# Install Miniconda only if not already installed
if [ ! -d "$HOME/miniconda3" ]; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b
else
    echo "Miniconda already installed. Skipping."
fi


export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

if ! conda info --envs | grep -q "flashrag"; then
    conda create -n flashrag python=3.10 -y
fi
conda activate flashrag

if [ ! -d "FlashRAG" ]; then
    git clone https://github.com/RUC-NLPIR/FlashRAG.git
fi
cd FlashRAG
pip install -e . 

# Fix dependency compatibility issues
pip install "accelerate<1.0.0" "scipy<1.15" --upgrade

cd ../
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

cd ./datasets/retrieval-corpus/
unzip wiki18_100w.zip
cd ../../

pip install termcolor
conda install -c pytorch faiss-gpu -y

cd ./FlashRAG/examples/quick_start
python simple_pipeline.py --model_path ../../../models/llama3-8b-instruct/ --retriever_path ../../../models/e5-base-v2/

# python -m flashrag.retriever.index_builder \
#     --retrieval_method e5 \
#     --model_path ../models/e5-base-v2/ \
#     --corpus_path ../datasets/retrieval-corpus/wiki18_100w.jsonl \
#     --save_dir ./indexes/ \
#     --use_fp16 \
#     --max_length 512 \
#     --batch_size 256 \
#     --pooling_method mean \
#     --faiss_type Flat 