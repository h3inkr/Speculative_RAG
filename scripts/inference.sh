CUDA_VISIBLE_DEVICES="0,3" python src/inference.py \
    --index_path /home/user4/Speculative_RAG/data/inference/arc-c/test.index \
    --meta_path /home/user4/Speculative_RAG/data/inference/arc-c/test_meta.pkl \
    --drafter_path ./draft-model-sft_Llama8B \
    --verifier meta-llama/Llama-3.1-8B-Instruct
