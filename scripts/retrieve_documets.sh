python /home/user4/Speculative_RAG/src/retrieve.py \
    --jsonl_path /home/user4/Speculative_RAG/data/inference/csqa/csqa-dev.jsonl \
    --output_path /home/user4/Speculative_RAG/data/inference/csqa/csqa-dev_retrieved.jsonl \
    --load_index_path ./data/inference/raco.index \
    --load_metadata_path ./data/inference/raco_meta.pkl
