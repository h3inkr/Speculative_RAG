from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import faiss
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse
import sys

sys.path.append('./src/utils')
from embedding import DPR, BasicRAG
from metrics import batched_select_best_choice

def generate_prompt(question, ctxs):
    #context_texts = "\n\n".join([f"[{doc['title']}]\n{doc['text']}" for doc in ctxs])
    prompt = (
        f"You are a helpful assistant.\n\n"
        f"Context:\n{ctxs}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    return prompt

def answer_question_batch(questions, contexts, max_tokens=256):
    prompts = [generate_prompt(q, ctx) for q, ctx in zip(questions, contexts)]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    tokenizer.pad_token = tokenizer.eos_token
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return [tokenizer.decode(o, skip_special_tokens=True).split("Answer:")[-1].strip() for o in outputs]

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", "-ip", type=str, required=True, help="faiss index")
    parser.add_argument("--meta_path", "-mp", type=str, required=True)
    parser.add_argument("--model", "-m", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--model_name_or_path", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--retrieval_model_name_or_path", default="/mnt2/user4/coconut_documents/question_encoder")
    parser.add_argument("--embedding_path", default="/mnt2/user4/coconut_documents/")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_argument()
    rag = BasicRAG(args)

    # Load FAISS index & metadata
    index = faiss.read_index(args.index_path)
    with open(args.meta_path, "rb") as f:
        metadata = pickle.load(f)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Parallel retrieval
    def retrieve_ctx(item):
        docs = rag.retrieve(query=item["question"], topk=10)
        #print(f"docs: {docs}")
        #texts = [{"title": doc.get("doc_title", ""), "text": doc.get("text", "")} for doc in docs]
        return {
            "question": item.get("question"),
            "answer": item.get("answer"),
            "choices": item.get("choices"),
            "texts": docs
        }

    with ThreadPoolExecutor(max_workers=8) as executor:
        qa_data = list(tqdm(executor.map(retrieve_ctx, metadata), total=len(metadata), desc="Retrieving"))

    # QA Evaluation in batch
    outputs = []
    batch_size = 8
    for i in tqdm(range(0, len(qa_data), batch_size), desc="QA Evaluation"):
        batch = qa_data[i:i + batch_size]
        questions = [item["question"] for item in batch if item["question"] and item["answer"]]
        contexts = [item["texts"] for item in batch if item["question"] and item["answer"]]
        answers = [item["answer"] for item in batch if item["question"] and item["answer"]]
        choices_list = [item.get("choices") for item in batch if item["question"] and item["answer"]]

        if not questions:
            continue

        generated_answers = answer_question_batch(questions, contexts)

        for gt, choices, gen in zip(answers, choices_list, generated_answers):
            outputs.append({
                "ground_truth": gt,
                "choices": choices,
                "generated_answer": generated_answers,
            })

    
    batched_select_best_choice(outputs) # Closed-set evaluation
    # batched_select_best_choice_open(outputs) # Open-set evaluation (alternative)
