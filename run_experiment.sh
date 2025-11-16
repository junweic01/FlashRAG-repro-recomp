#!/bin/bash
export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda activate flashrag

# cd /home/ec2-user/flashrag/FlashRAG-repro-recomp/examples/quick_start
# python simple_pipeline.py --model_path ../../../models/llama3-8b-instruct/ --retriever_path ../../../models/e5-base-v2/

# cd /home/ec2-user/flashrag/FlashRAG-repro-recomp/
# python -m flashrag.retriever.index_builder \
#     --retrieval_method e5 \
#     --model_path ../models/e5-base-v2/ \
#     --corpus_path ../datasets/retrieval-corpus/wiki18_100w_1_20.jsonl \
#     --save_dir ./indexes/ \
#     --use_fp16 \
#     --max_length 512 \
#     --batch_size 256 \
#     --pooling_method mean \
#     --faiss_type Flat 

# cd /home/ec2-user/flashrag/FlashRAG-repro-recomp/examples/methods
# python run_exp.py --method_name 'recomp' --split 'test' --dataset_name 'nq' --gpu_id '0' > run_exp.log 2>&1 &