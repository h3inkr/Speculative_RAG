import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
import faiss
import json
import argparse
from tqdm import tqdm
import numpy as np
import os
import pickle

def safe_load_tsv(path):
    data = {"id": [], "text": [], "title": []}
    with open(path, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                doc_id, text, title = parts
            elif len(parts) == 2:
                doc_id, text = parts
                title = ""
            else:
                continue
            data["id"].append(doc_id)
            data["text"].append(text)
            data["title"].append(title)
    return pd.DataFrame(data)

@torch.no_grad()
def embed_texts(texts, batch_size=64):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding docs"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].float()
        normed = normalize(embeddings.cpu().numpy(), axis=1)
        all_embeddings.append(normed)
    return np.vstack(all_embeddings)

def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

@torch.no_grad()
def retrieve_for_questions(questions, top_k=10):
    results = []
    for q in tqdm(questions, desc="Retrieving"):
        if "choices" in q:  # e.g., {"question": ..., "choices": ..., ...}
            q_text = q.get("question", q.get("question_stem", ""))
            choices = q["choices"]
            answerKey = q.get("answerKey", "")
        elif "question" in q and "choices" in q["question"] and "stem" in q["question"]:
            q_text = q["question"]["stem"]
            choices = q["question"]["choices"]
            answerKey = q.get("answerKey", "")
        elif "goal" in q:
            q_text = q["goal"]
            choices = {"text": [q["sol1"], q["sol2"]], "label": ["A", "B"]}
            answerKey = "A" if q["sol1"] in q_text else "B"
        elif "sentence" in q:
            q_text = q["sentence"]
            choices = {"text": [q["option1"], q["option2"]], "label": ["A", "B"]}
            answerKey = q["answer"]
        else:
            print(f"⚠️ Unknown format: {q}")
            continue

        
        inputs = tokenizer(q_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(**inputs)
        q_emb = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()
        q_emb = normalize(q_emb, axis=1)

        D, I = index.search(q_emb, top_k)
        retrieved = []
        for idx in I[0]:
            retrieved.append({
                "id": doc_ids[idx],
                "title": doc_titles[idx],
                "text": doc_texts[idx]
            })

        results.append({
            #"id": q["id"],
            "question": q_text,
            "choices": choices,
            "answerKey": answerKey,
            "retrieved_docs": retrieved
        })

    return results

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv_path", type=str, help="Required for saving FAISS index and metadata")
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="facebook/contriever-msmarco")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--embed_batch_size", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=10)

    parser.add_argument("--save_index_path", type=str, help="Path to save FAISS index")
    parser.add_argument("--save_metadata_path", type=str, help="Path to save metadata (pkl)")
    parser.add_argument("--load_index_path", type=str, help="Path to import FAISS index")
    parser.add_argument("--load_metadata_path", type=str, help="Path to import metadata (pkl)")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_argument()

    DEVICE = args.device if torch.cuda.is_available() else "cpu"
    MODEL_NAME = args.model_name
    TOP_K = args.top_k
    EMBED_BATCH_SIZE = args.embed_batch_size

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    
    if args.load_index_path and args.load_metadata_path:
        print("📥 Import saved index and metadata...")
        index = faiss.read_index(args.load_index_path)

        with open(args.load_metadata_path, "rb") as f:
            metadata = pickle.load(f)

        doc_ids = metadata["id"]
        doc_texts = metadata["text"]
        doc_titles = metadata["title"]
    else:
        if not args.tsv_path:
            raise ValueError("Need TSV file path (--tsv_path)")
        print("📄 Load TSV document and execute embedding...")
        doc_df = safe_load_tsv(args.tsv_path)
        doc_texts = doc_df["text"].tolist()
        doc_ids = doc_df["id"].tolist()
        doc_titles = doc_df["title"].tolist()

        doc_embeddings = embed_texts(doc_texts, batch_size=EMBED_BATCH_SIZE)
        dim = doc_embeddings.shape[1]

        # 압축 인덱스 사용 (IVFPQ)
        print("📦 Creating compressed FAISS index with IVFPQ...")
        nlist = 100  # 클러스터 개수
        m = 16       # 서브 벡터 개수
        nbits = 8    # 서브 벡터 당 비트 수

        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)

        print("🧠 Training FAISS index...")
        index.train(doc_embeddings)
        index.add(doc_embeddings)

        if args.save_index_path:
            faiss.write_index(index, args.save_index_path)
            print(f"✅ Save compressed FAISS index: {args.save_index_path}")

        if args.save_metadata_path:
            metadata = {
                "id": doc_ids,
                "text": doc_texts,
                "title": doc_titles
            }
            with open(args.save_metadata_path, "wb") as f:
                pickle.dump(metadata, f)
            print(f"✅ Save metadata as pickle: {args.save_metadata_path}")

    print("📂 Loading questions from JSONL...")
    questions = load_questions(args.jsonl_path)

    print("🔍 Retrieving top-k documents per question...")
    retrieved_results = retrieve_for_questions(questions, top_k=TOP_K)

    print(f"💾 Saving retrieval results to {args.output_path}...")
    with open(args.output_path, "w", encoding="utf-8") as f:
        for item in retrieved_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("✅ Retrieval complete.")
