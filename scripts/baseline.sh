CUDA_VISIBLE_DEVICES=1 python ./src/baseline.py \
    -ip /home/user4/Speculative_RAG/data/inference/arc-c/test.index \
    -mp /home/user4/Speculative_RAG/data/inference/arc-c/test_meta.pkl \
    -m meta-llama/Llama-3.1-8B-Instruct
