import torch
torch.cuda.empty_cache()
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    default_data_collator
)
from peft import (
    prepare_model_for_kbit_training,
    get_peft_model, LoraConfig, TaskType
)
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map={"": 3}
)

# LoRA 설정
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# LoRA 적용
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# load dataset
dataset = load_dataset(
    "json", 
    data_files={"train": "/home/user4/Speculative_RAG/data/train/knowledge_intensive/sft/sft_data_retrieved.jsonl"} # 훈련 데이터셋
)["train"]


# 포맷팅 및 토크나이징 함수
def format_and_tokenize(example):
    question = example["question"]
    answer = example["answer"]
    document = example["text"]
    rationale = example["rationale"]

    prompt = f"[QUESTION] {question}\n[DOCUMENT] {document}"
    response = f"[ANSWER] {answer}\n[RATIONALE] {rationale}"
    full = prompt + "\n\n" + response

    tokenized = tokenizer(
        full,
        truncation=True,
        max_length=512,
        padding="max_length"
    )

    input_len = len(tokenized["input_ids"])
    labels = tokenized["input_ids"].copy()

    # input 전체에 대해 label 설정 (LoRA SFT 방식)
    tokenized["labels"] = labels

    return tokenized

# 데이터 전처리
tokenized_dataset = dataset.map(format_and_tokenize, remove_columns=dataset.column_names)

args = TrainingArguments(
    output_dir="./draft-model-sft_mistral7B",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    logging_dir="./logs",
    save_steps=500,
    save_total_limit=2,
    fp16=True,
    learning_rate=2e-5,
    warmup_steps=100,
    weight_decay=0.01,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    data_collator=default_data_collator
)

trainer.train()

model.save_pretrained("./draft-model-sft_mistral7B")
tokenizer.save_pretrained("./draft-model-sft_mistral7B")
