import json
import torch
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import argparse
import numpy as np
import os
import pickle

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def InbedderEmbedding(input_path, index_output_path, metadata_output_path):
    embedding_model = "KomeijiForce/inbedder-roberta-large"

    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    model = AutoModel.from_pretrained(embedding_model).to(device).eval()

    # Load corpus
    with open(input_path, "r") as f:
        corpus = json.load(f) 

    doc_texts = []
    metadata = []
    for item_idx, item in enumerate(corpus):
        doc = item["retrieved_docs"][0]
        doc_texts.append(doc['text'])
        metadata.append({
            "question": item["question"],
            "answer": item["answer"],
            "doc_id": doc.get("id", None),
            "doc_title": doc.get("title", None),
            "texts": doc.get("text", None)
        })

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

    doc_embeddings_np = embed_texts(doc_texts, tokenizer, model).numpy().astype("float32") # FAISS needs float32

    # Build FAISS index
    index = faiss.IndexFlatL2(doc_embeddings_np.shape[1])
    index.add(doc_embeddings_np)

    # Save FAISS index
    faiss.write_index(index, index_output_path)

    # Save as JSONL
    with open(metadata_output_path, "wb") as f:
        pickle.dump(metadata, f)

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-ip", type=str) # "./data/train/retrieval_corpus/knowledge_intensive/retrieval/asqa.jsonl"
    parser.add_argument("--index_path", "-ip", type=str, help="Path to save FAISS index") # "./data/train/embedding/asqa.index"
    parser.add_argument("--meta_path", "-mp", type=str, help="Path to save doc metadata") # /home/user4/SpeRAG/data/train/embedding/asqa_meta.pkl
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_argument()
    InbedderEmbedding(args.input_path, args.index_output_path, args.metadata_output_path)
