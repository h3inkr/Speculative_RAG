import argparse
import json
import pickle
import faiss
from tqdm import tqdm
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import time
import openai
from dotenv import load_dotenv
import os

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_file", "-if", type=str, required=True)
    parser.add_argument("--metadata_file", "-mf", type=str, required=True)
    parser.add_argument("--save_path", "-sp", type=str, required=True) # "./data/train/sft_data_asqa.jsonl"
    return parser.parse_args()

def generate_rationale_item(client, item, model_name="gpt-4", max_retries=3, retry_delay=60):
    try:
        question = re.search(r"## Input:\s*(.*)", item['question'], re.DOTALL).group(1).strip()
    except:
        print(f"[!] Malformed question: {item['question'][:30]}...")
        return None

    prompt = f"""Memorize this piece of evidence in mind and use it as if you already know it.

# Evidence: {item['text']}

# Instruction: {question}

# Response: {item['answer']}

# Rationale: (complete the following sentence with details from the evidence; you can only use the information from the evidence)
"""

    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            rationale = response.choices[0].message.content.strip()
            return {
                "question": item["question"],
                "answer": item["answer"],
                "text": item["text"],
                "rationale": rationale
            }
        except Exception as e:
            if "429" in str(e):
                print(f"[!] 429 error: Quota exceeded. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retries += 1
            else:
                print(f"[!] Error generating rationale for: {item['question'][:30]}... -> {e}")
                return None
    print(f"[!] Max retries reached for: {item['question'][:30]}...")
    return None

if __name__ == "__main__":
    args = parse_argument()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)
    model_name = "gpt-4" 

    index = faiss.read_index(args.index_file)
    with open(args.metadata_file, "rb") as f:
        metadata = pickle.load(f)

    qa_pairs = [
        {
            "question": item["question"],
            "answer": item["answer"],
            "text": item["texts"]
        }
        for item in metadata
        if item.get("question") and item.get("answer") and item.get("texts")
    ]

    with open(args.save_path, "a", encoding="utf-8") as fout:
        for item in tqdm(qa_pairs, desc="Generating rationales", total=len(qa_pairs)):
            result = generate_rationale_item(client, item, model_name=model_name)
            if result:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()