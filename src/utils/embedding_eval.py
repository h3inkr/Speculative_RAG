import json
import torch
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import argparse
import numpy as np
import os
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def InbedderEmbedding(input_path, index_output_path, metadata_output_path):
    embedding_model = "KomeijiForce/inbedder-roberta-large"

    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    model = AutoModel.from_pretrained(embedding_model).to(device).eval()

    # Load corpus (json or jsonl)
    corpus = []
    if input_path.endswith(".json"):
        with open(input_path, "r") as f:
            corpus = json.load(f)
    elif input_path.endswith(".jsonl"):
        with open(input_path, "r") as f:
            for line in f:
                corpus.append(json.loads(line))
    else:
        raise ValueError("Unsupported file format. Please provide a .json or .jsonl file.")

    doc_texts = []
    metadata = []

    #print(f"corpus: {corpus}")
    for item in corpus:
        if not item.get("retrieved_docs"):
            continue  # skip if no retrieved_docs

        # 모든 retrieved_docs의 text를 수집
        docs = item["retrieved_docs"]
        for doc in docs:
            doc_texts.append(doc["text"])
            #print(f"doc: {doc}")

        #print(f"item: {item}")
        # metadata 하나에 retrieved_docs 전체를 저장
        metadata.append({
            "question": item.get("question"),
            "answer": item.get("answerKey"),
            "choices": item.get("choices"),
            "retrieved_docs": [
                {
                    "doc_id": doc.get("id", None),
                    "doc_title": doc.get("title", None),
                    "text": doc.get("text", None)
                }
                for doc in docs
            ]
        })
        #print(f"metadata: {metadata}")

    # Embed corpus
    def embed_texts(texts, tokenizer, model, batch_size=32):
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding corpus"):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs).last_hidden_state.mean(dim=1)
            embeddings.append(outputs.cpu())
        return torch.cat(embeddings, dim=0)

    doc_embeddings_np = embed_texts(doc_texts, tokenizer, model).numpy().astype("float32")

    # Build FAISS index
    index = faiss.IndexFlatL2(doc_embeddings_np.shape[1])
    index.add(doc_embeddings_np)

    # Save FAISS index
    faiss.write_index(index, index_output_path)

    # Save metadata
    with open(metadata_output_path, "wb") as f:
        pickle.dump(metadata, f)

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-ip", type=str, help="Path to input .json or .jsonl file") # ./data/inference/arc-c/test_retrieved.jsonl
    parser.add_argument("--index_path", "-inp", type=str, help="Path to save FAISS index") # ./data/inference/arc-c/test_psg.index
    parser.add_argument("--meta_path", "-mp", type=str, help="Path to save doc metadata") # ./data/inference/arc-c/test_psg_meta.pkl
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_argument()
    InbedderEmbedding(args.input_path, args.index_path, args.meta_path)
