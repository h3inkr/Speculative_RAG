python /home/user4/Speculative_RAG/src/retrieve.py \
    --jsonl_path ./data/inference/csqa2/CSQA2_dev.json \
    --output_path ./data/inference/csqa2/CSQA2_dev_retrieved.json \
    --load_index_path ./data/inference/raco.faiss \
    --load_metadata_path ./data/inference/raco.json
