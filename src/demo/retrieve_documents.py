import json
import torch
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# -------- Config --------
corpus_path = "./data/train/retrieval_corpus/zebra_documents.jsonl"
embedding_model = "KomeijiForce/inbedder-roberta-large"
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# -------- Load Model (for both query & document) --------
tokenizer = AutoTokenizer.from_pretrained(embedding_model)
model = AutoModel.from_pretrained(embedding_model).to(device).eval()

# -------- Load Corpus --------
corpus = []
with open(corpus_path, "r") as f:
    for line in f:
        item = json.loads(line)
        corpus.append({"id": item["id"], "text": item["text"]})

# -------- Embed Corpus --------
def embed_texts(texts, tokenizer, model, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding corpus"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs).last_hidden_state.mean(dim=1)
        embeddings.append(outputs.cpu())
    return torch.cat(embeddings, dim=0)

# -------- Document Embedding --------
doc_texts = [item["text"] for item in corpus]
doc_embeddings = embed_texts(doc_texts, tokenizer, model)
doc_embeddings_np = doc_embeddings.numpy()

# -------- FAISS Indexing -------- # FAISS / Qdrant
dimension = doc_embeddings_np.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner Product = cosine similarity if normalized
faiss.normalize_L2(doc_embeddings_np)
index.add(doc_embeddings_np)

# -------- QA Pair Retrieval --------
qa_path = "./data/train/qa_pair/ASQA_qa_pairs_only.json"
with open(qa_path, "r") as f:
    qa_pairs = json.load(f)

results = []

for qa in tqdm(qa_pairs, desc="Retrieving"):
    q = qa["question"]
    inputs = tokenizer(q, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        query_vec = model(**inputs).last_hidden_state.mean(dim=1)
        query_vec = torch.nn.functional.normalize(query_vec, p=2, dim=1).cpu().numpy()

    D, I = index.search(query_vec, 5)

    retrieved = []
    for idx in I[0]:
        retrieved.append({
            "id": corpus[idx]["id"],
            "text": corpus[idx]["text"],
            "embedding": doc_embeddings_np[idx].tolist()  # numpy array → list for JSON serialization
        })

    results.append({
        "question": q,
        "retrieved": retrieved,
        "short_answers": qa.get("short_answers", [])
    })

# -------- Save Results --------
with open("./data/train/retrieval_output/retrieved_results_5.json", "w") as f:
    json.dump(results, f, indent=2)
