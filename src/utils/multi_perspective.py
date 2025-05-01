import random
from typing import List, Dict, Any
import faiss
from sklearn.cluster import KMeans
import json
import numpy as np
import pickle
import os
import contextlib
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as hf_logging

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# transformers 로깅 레벨을 ERROR로 설정: progress bar 끄기
hf_logging.set_verbosity_error()

# Redirect print outputs (stdout/stderr) when loading models
@contextlib.contextmanager
def suppress_transformers_output():
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield

# K-means clustering
def documents_to_clusters(
    docs: List[Dict[str, Any]], m: int, seed: int = 1399
) -> List[List[Dict[str, Any]]]:
    
    random.seed(seed)

    model = SentenceTransformer("all-MiniLM-L6-v2") 
    texts = [doc["text"] for doc in docs]
    embeddings = model.encode(texts, show_progress_bar=True)

    #print(f"texts: {texts}")
    
    # print(metadata[0])
    # print(len(metadata[0]))
    # quit()
    kmeans = KMeans(n_clusters=m, random_state=seed)
    clusters = kmeans.fit_predict(embeddings)

    #print(f"clusters: {clusters}")

    clustered_docs = [[] for _ in range(m)]
    for doc_meta, cluster_id in zip(docs, clusters):
        clustered_docs[cluster_id].append(doc_meta)

    #print(f"cluster_docs:{clustered_docs}")

    return clustered_docs

# Multi-Perspective Sampling (make subsets)
def multi_perspective_sampling(
    clusters: List[List[Dict[str, Any]]], k: int, seed: int = 1399
) -> List[List[str]]:

    random.seed(seed)
    all_sampled_ids = []

    for item in clusters:
        #print(f"item: {item}")
        
        doc_ids = [doc["doc_id"] for doc in item if isinstance(doc, dict)]

        if not doc_ids:
            continue

        sampled_ids = random.sample(doc_ids, 1)
        all_sampled_ids.append(sampled_ids)

    return all_sampled_ids
