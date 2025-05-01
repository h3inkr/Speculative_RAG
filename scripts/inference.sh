CUDA_VISIBLE_DEVICES=1 python ./src/inference.py \
    --index_path /home/user4/Speculative_RAG/data/inference/popqa_test.index \
    --meta_path /home/user4/Speculative_RAG/data/inference/popqa_test_meta.pkl \
    --drafter_path ./draft-model-sft_Llama8B