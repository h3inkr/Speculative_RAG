CUDA_VISIBLE_DEVICES=1 python ./src/finetune.py \
    --train_file ./data/train/knowledge_intensive/sft/sft_data_retrieved.jsonl \
    --model mistralai/Mistral-7B-v0.1 \
    --output_dir ./test_57