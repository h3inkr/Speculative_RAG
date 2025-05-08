import argparse
import json
import pickle
import faiss
from tqdm import tqdm
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import time
import re
from threading import Lock
from dotenv import load_dotenv

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_file", "-if", type=str, required=True)
    parser.add_argument("--metadata_file", "-mf", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--rationale_generator", "-dr", type=str, default="Gemini-Ultra")
    return parser.parse_args()

last_called = {}
lock = Lock()

def rate_limited_generate(model_idx, model, prompt, min_interval=7.5):
    global last_called
    with lock:
        now = time.time()
        last_time = last_called.get(model_idx, 0)
        wait_time = last_time + min_interval - now
        if wait_time > 0:
            time.sleep(wait_time)
        last_called[model_idx] = time.time()
    return model.generate_content(prompt)

def generate_rationale_item(model_idx, model, item, max_retries=3, retry_delay=30):
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
            response = rate_limited_generate(model_idx, model, prompt)
            return {
                "question": item["question"],
                "answer": item["answer"],
                "text": item["text"],
                "rationale": response.text.strip()
            }
        except Exception as e:
            if "429" in str(e):
            
                #print(f"[!] 429 error: Quota exceeded. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retries += 1
            else:
                print(f"[!] Error: {e}")
                return None
    return None

if __name__ == "__main__":
    args = parse_argument()

    load_dotenv()
    api_keys = os.getenv("GEMINI_API_KEYS", "").split(",")

    models = []
    for key in api_keys:
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        models.append(model)

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

    with open(args.save_path, "a", encoding="utf-8") as fout, ThreadPoolExecutor(max_workers=5) as executor:
        future_to_item = {
            executor.submit(generate_rationale_item, i % len(models), models[i % len(models)], item): item
            for i, item in enumerate(qa_pairs)
        }

        for future in tqdm(as_completed(future_to_item), total=len(future_to_item), desc="Generating rationales"):
            result = future.result()
            if result:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
