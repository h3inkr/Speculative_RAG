
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
import sys

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = fnull
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# Supprissing tqdm
with suppress_stdout():
    model = SentenceTransformer("all-MiniLM-L6-v2")

# K-means clustering
def documents_to_clusters(
    docs: List[Dict[str, Any]], m: int, seed: int = 1399
) -> List[List[Dict[str, Any]]]:
    
    random.seed(seed)
    docs = [{"doc_id": i, "text": doc} for i, doc in enumerate(docs)]
    texts = [doc["text"] for doc in docs]
   # texts = [doc for doc in docs]
    embeddings = model.encode(texts, show_progress_bar=False)

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
'''

def documents_to_clusters(
    docs: List[str], m: int, seed: int = 1399
) -> List[List[Dict[str, Any]]]:
    random.seed(seed)

    # 임의의 doc_id를 부여하면서 dict로 변환
    structured_docs = [{"doc_id": i, "text": doc} for i, doc in enumerate(docs)]

    texts = [doc["text"] for doc in structured_docs]
    embeddings = model.encode(texts, show_progress_bar=False)

    kmeans = KMeans(n_clusters=m, random_state=seed)
    clusters = kmeans.fit_predict(embeddings)

    clustered_docs = [[] for _ in range(m)]
    for doc_meta, cluster_id in zip(structured_docs, clusters):
        clustered_docs[cluster_id].append(doc_meta)

    return clustered_docs
'''
'''
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
'''

def multi_perspective_sampling(
    clusters: List[List[Dict[str, Any]]], k: int, seed: int = 1399
) -> List[List[Dict[str, Any]]]:
    random.seed(seed)
    subsets = [[] for _ in range(k)]

    # 각 cluster를 복사하여 sampling 시 원본을 유지
    cluster_pool = [list(cluster) for cluster in clusters]

    for i in range(k):
        for cluster_idx, cluster in enumerate(cluster_pool):
            if not cluster:
                continue  # 이미 다 뽑혀서 비어 있으면 skip

            sampled_doc = random.choice(cluster)
            subsets[i].append(sampled_doc)

            # 선택된 문서는 제거
            cluster_pool[cluster_idx].remove(sampled_doc)

    return subsets

'''
def multi_perspective_sampling(
    clusters: List[List[Dict[str, Any]]], k: int, seed: int = 1399
) -> List[List[str]]:
    random.seed(seed)
    subsets = []

    # 가장 큰 클러스터 기준으로 subset 개수 계산
    max_len = max(len(cluster) for cluster in clusters)
    num_subsets = (max_len + k - 1) // k  # 올림 나눗셈

    for i in range(num_subsets):
        current_subset = []
        for cluster in clusters:
            # 샘플링 시작 인덱스 계산
            start_idx = i * k
            end_idx = min(start_idx + k, len(cluster))
            current_docs = cluster[start_idx:end_idx]
            current_subset.append([doc["doc_id"] for doc in current_docs])
            subsets.append(current_subset)

    return subsets
'''

'''
def multi_perspective_sampling(
    clusters: List[List[str]], k: int, seed: int = 1399
) -> List[List[str]]:
    
    random.seed(seed)
    all_sampled_docs = []

    for item in clusters:
        # item은 각 클러스터의 문서 내용들 리스트
        if not item:
            continue
        
        # 무작위로 1개의 문서를 샘플링
        sampled_doc = random.choice(item)
        all_sampled_docs.append(sampled_doc)

    return all_sampled_docs

'''
