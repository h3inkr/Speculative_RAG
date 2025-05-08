import os
import torch
import argparse
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_prompt(question: str, document: str) -> str:
    return f"[QUESTION] {question}\n[DOCUMENT] {document}"

def build_response(answer: str, rationale: str) -> str:
    return f"[ANSWER] {answer}\n[RATIONALE] {rationale}"


def format_and_tokenize(example, tokenizer):
    question = example["question"]
    answer = example["answer"]
    document = example["text"]
    rationale = example["rationale"]

    prompt = build_prompt(question, document)
    response = build_response(answer, rationale)
    full = prompt + "\n\n" + response

    tokenized = tokenizer(
        full,
        truncation=True,
        max_length=512,
        padding="max_length"
    )

    labels = tokenized["input_ids"].copy()
    tokenized["labels"] = labels
    return tokenized

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", "-t", type=str, required=True) # "./data/train/knowledge_intensive/sft/sft_data_retrieved.jsonl"
    parser.add_argument("--model", "-m", type=str, default="mistralai/Mistral-7B-v0.1") 
    parser.add_argument("--output_dir", "-o", type=str, default="./draft-model-sft_mistral7B")
   
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = parse_argument()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    special_tokens = {
    "additional_special_tokens": ["[QUESTION]", "[DOCUMENT]", "[ANSWER]", "[RATIONALE]"]
    }

    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # LoRA Configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    dataset = load_dataset("json", data_files={"train": args.train_file})["train"]
    tokenized_dataset = dataset.map(
        lambda ex: format_and_tokenize(ex, tokenizer),
        remove_columns=dataset.column_names
    )

    # Train
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        #per_device_train_batch_size=1,
        per_device_train_batch_size=2,
        #gradient_accumulation_steps=8,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        logging_dir="./logs",
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        #learning_rate=2e-5,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",     # 안정적인 학습 스케줄
        logging_steps=50,
        save_strategy="epoch",
        warmup_steps=100,
        weight_decay=0.01,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=default_data_collator
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n❗ Complete fine-tuning!")