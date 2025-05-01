from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import faiss
import pickle
from tqdm import tqdm
from itertools import islice
import sys
sys.path.append('./src/utils')
from metrics import batched_select_best_choice, batched_select_best_choice_open
import argparse

def generate_non_instruction_prompt(question, ctxs):
    context_texts = "\n\n".join(
        [f"[{i+1}] {doc.get('title', 'No Title')}\n{doc.get('text', '')}" for i, doc in enumerate(ctxs)]
    )
    prompt = (
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"Evidence:\n{context_texts}\n\n"
        f"Instruction: {question}\n"
        f"Response:"
    )
    return prompt

def generate_instruction_prompt(question, ctxs):
    context_texts = "\n\n".join(
        [f"[{i+1}] {doc.get('title', 'No Title')}\n{doc.get('text', '')}" for i, doc in enumerate(ctxs)]
    )
    prompt = (
        f"[INST] Below is an instruction that describes a task. Write a response for it and state your explanation supporting your response.\n\n"
        f"Instruction: {question}\n"
        f"Evidence:\n{context_texts}\n"
        f"[/INST] The Response is:"
    )
    return prompt

def generate_prompt(question, ctxs):
    context_texts = "\n\n".join([f"[{doc['title']}]\n{doc['text']}" for doc in ctxs])
    prompt = (
        f"You are a helpful assistant.\n\n"
        f"Context:\n{context_texts}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    return prompt


def answer_question(question, ctxs, max_tokens=256):
    prompt = generate_prompt(question, ctxs)
    #prompt = generate_non_instruction_prompt(question, ctxs)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            #temperature=0.5,
            #top_p=0.9,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #print(f"answer: {answer}")
    #return answer.split("Response:")[-1].strip()
    #return answer.split("[/INST] The Response is:")[-1].strip()
    return answer.split("Answer:")[-1].strip()


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", "-ip", type=str, required=True, help="faiss index")
    parser.add_argument("--meta_path", "-mp", type=str, required=True)
    parser.add_argument("--model", "-m", type=str, default="mistralai/Mistral-7B-v0.1")
    args = parser.parse_args()

    return args 

if __name__ == "__main__":
    args = parse_argument()

    # Load embedding
    index = faiss.read_index(args.index_path)
    with open(args.meta_path, "rb") as f:
        metadata = pickle.load(f)

    # Extract instruction-following QA pairs
    qa_pairs = {}
    for item in metadata:
        n = 0
        texts = [] 

        q = item.get("question")
        a = item.get("answer")
        choices = item.get("choices")
        docs=item.get("retrieved_docs")

        n += 1
        # item의 여러 문서 정보들을 texts에 추가
        for n, doc in enumerate(item.get('retrieved_docs', []), start=1):
            texts.append({
                "title": doc.get('doc_title', ''),
                "text": doc.get('text', '')
            })

        if q and a:
            qa_pairs[q] = {
                "answer": a,
                "choices": choices,
                "texts": texts,
            }

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    outputs = []
    #for question, answithchoi in tqdm(islice(qa_pairs.items(), 138), desc="QA Evaluation", unit="pair"): 
    for question, answithchoi in tqdm(qa_pairs.items(), desc="QA Evaluation", unit="pair"): 

        answer = answithchoi["answer"]
        #print(f"answer: {answer}")
        choices = answithchoi["choices"]
        ctxs = answithchoi["texts"]

        generated_answer = answer_question(question, ctxs)
        #print(f"generated_answer: {generated_answer}")

        d = {
            "ground_truth": answer,
            "choices": choices,
            "generated_answer": generated_answer,
        }
        outputs.append(d)

    #batched_select_best_choice(outputs) # Closed-set
    batched_select_best_choice_open(outputs) # Open-set