#!/bin/bash
export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda activate flashrag

# cd /home/ubuntu/FlashRAG-repro-recomp/examples/quick_start
# python simple_pipeline.py --model_path ../../models/llama3-8b-instruct/ --retriever_path ../../models/e5-base-v2/

cd /home/ubuntu/FlashRAG-repro-recomp/
python -m flashrag.retriever.index_builder \
    --retrieval_method e5 \
    --model_path models/e5-base-v2/ \
    --corpus_path datasets/retrieval-corpus/wiki18_100w_1_10.jsonl \
    --save_dir ./wiki18_100w_1_10_indexes/ \
    --use_fp16 \
    --max_length 512 \
    --batch_size 8 \
    --pooling_method mean \
    --faiss_type Flat

# cd /home/ubuntu/FlashRAG-repro-recomp/examples/methods
# python run_exp.py --method_name 'gpt' --split 'dev' --dataset_name 'hotpotqa' --gpu_id '0' > 1_10_data_gpt_hotpotqa_topk_5.log 2>&1 &