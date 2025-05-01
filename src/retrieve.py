import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import DPRQuestionEncoder, DPRContextEncoder
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle
import faiss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

query_encoder = SentenceTransformer("sentence-transformers/msmarco-MiniLM-L6-cos-v5", device=device)
context_encoder = SentenceTransformer("sentence-transformers/msmarco-MiniLM-L6-cos-v5", device=device)

def embed_documents(documents, batch_size=64):
    embeddings = []
    for i in tqdm(range(0, len(documents), batch_size), desc="📌 Embedding documents"):
        batch = documents[i:i+batch_size]
        with torch.no_grad():
            batch_embeddings = context_encoder.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

def embed_query(queries, batch_size=64):
    embeddings = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        with torch.no_grad():
            batch_embeddings = query_encoder.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

def retrieve_for_questions_with_faiss(args, questions, doc_texts, doc_ids, doc_titles, top_k=10, use_approx=False, nlist=100):
    """
    questions: list of dicts with keys "question" or "question_stem", "choices", "answerKey"
    doc_texts, doc_ids, doc_titles: list of documents
    """
    results = []

    # 1. 문서 임베딩  # 2. FAISS 인덱스 구축
    if args.load_embedding:
        index = load_faiss_index(args.load_embedding)
    else:
        doc_embeddings = embed_documents(doc_texts)  # (n_docs, dim)
        index = build_faiss_index(doc_embeddings, use_approx=use_approx, nlist=nlist)
        
        # 인덱스 저장
        #if args.save_embedding:
        #    save_faiss_index(index, args.save_embedding)

    # 3. 질문 텍스트 준비
    query_texts = [q.get("question", q.get("question_stem", "")) for q in questions]
    choices_list = [q.get("choices", []) for q in questions]
    answer_keys = [q.get("answerKey", "") for q in questions]

    # 4. 질문 배치 임베딩
    query_embeddings = embed_query(query_texts).astype('float32')
    faiss.normalize_L2(query_embeddings)

    # 5. FAISS를 이용해 top-k 검색
    D, I = index.search(query_embeddings, top_k)  # D: similarities, I: indices

    # 6. 결과 정리
    for q_idx, (indices, choices, answerKey, q_text) in enumerate(zip(I, choices_list, answer_keys, query_texts)):
        retrieved = []
        for idx in indices:
            if idx == -1:  # 검색 실패 (IVF일 때 드물게 나옴)
                continue
            retrieved.append({
                "id": doc_ids[idx],
                "title": doc_titles[idx],
                "text": doc_texts[idx]
            })

        results.append({
            "question": q_text,
            "choices": choices,
            "answerKey": answerKey,
            "retrieved_docs": retrieved
        })

    return results

def build_faiss_index(doc_embeddings, use_approx=False, nlist=100):
    """
    doc_embeddings: np.ndarray of shape (n_docs, dim)
    use_approx: whether to use approximate search (IndexIVFFlat)
    nlist: number of clusters for IVF
    """
    dim = doc_embeddings.shape[1]
    doc_embeddings = doc_embeddings.astype('float32')
    faiss.normalize_L2(doc_embeddings)

    if use_approx:
        quantizer = faiss.IndexFlatIP(dim)  # used for clustering
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(doc_embeddings)
    else:
        index = faiss.IndexFlatIP(dim)

    index.add(doc_embeddings)
    return index

def save_faiss_index(index, save_path):
    """
    인덱스를 지정된 경로에 저장합니다.
    """
    faiss.write_index(index, save_path)
    print(f"✅ FAISS index saved to {save_path}")

def load_faiss_index(load_path):
    """
    저장된 FAISS 인덱스를 불러옵니다.
    """
    index = faiss.read_index(load_path)
    print(f"📥 FAISS index loaded from {load_path}")
    return index

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

def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def parse_argument():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv_path", type=str, help="Required for loading documents for retrieval")
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--save_metadata_path", type=str, help="Path to save metadata (pkl)")
    parser.add_argument("--load_metadata_path", type=str, help="Path to import metadata (pkl)")
    parser.add_argument("--save_embedding", type=str, help="Path to save FAISS index")
    parser.add_argument("--load_embedding", type=str, help="Path to load FAISS index")
    args = parser.parse_args()

    return args

# 메인 실행
def main(args):
    # TSV 파일 로드
    if args.load_metadata_path:
        print("📥 Import saved metadata...")
        with open(args.load_metadata_path, "rb") as f:
            metadata = pickle.load(f)
        doc_ids = metadata["id"]
        doc_texts = metadata["text"]
        doc_titles = metadata["title"]
    else:
        if not args.tsv_path:
            raise ValueError("Need TSV file path (--tsv_path)")
        
        print("📄 Load TSV document...")
        doc_df = safe_load_tsv(args.tsv_path)
        doc_texts = doc_df["text"].tolist()
        doc_ids = doc_df["id"].tolist()
        doc_titles = doc_df["title"].tolist()

        if args.save_metadata_path:
            metadata = {
                "id": doc_ids,
                "text": doc_texts,
                "title": doc_titles,
            }
            with open(args.save_metadata_path, "wb") as f:
                pickle.dump(metadata, f)
            print(f"✅ Save metadata as pickle: {args.save_metadata_path}")

    # 질문 로드
    print("📂 Loading questions from JSONL...")
    questions = load_questions(args.jsonl_path)

    # 검색 실행
    print("🔍 Retrieving top-k documents per question using FAISS...")
    retrieved_results = retrieve_for_questions_with_faiss(args, questions, doc_texts, doc_ids, doc_titles, top_k=args.top_k)

    # 결과 저장
    print(f"💾 Saving retrieval results to {args.output_path}...")
    with open(args.output_path, "w", encoding="utf-8") as f:
        for item in retrieved_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("✅ Retrieval complete.")

if __name__ == "__main__":
    args = parse_argument()
    main(args)
