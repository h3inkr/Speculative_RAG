import random
from typing import List, Dict, Any
import faiss
from sklearn.cluster import KMeans
import json
import numpy as np
import pickle

# K-means clustering
def documents_to_clusters(
    index_path:str, meta_path:str, k: int, seed: int = 1399
) -> List[List[Dict[str, Any]]]:
    """
    For each cluster, it contains k documents

    Args:
        embedded_file: Retrival documents to cluster
        k: 
        1. The number of clusters during clustering -> documents_to_clusters
        2. Each subset contains k documents -> 각 cluster 당 하나씩 샘플링
        seed: Random seed for reproducibility

    Returns: A list of lists(클러스터링된 문서들)
        
    """
    
    random.seed(seed)

    index = faiss.read_index(index_path)
    embeddings = index.reconstruct_n(0, index.ntotal)

    # Load metadata
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    kmeans = KMeans(n_clusters=k, random_state=seed)
    clusters = kmeans.fit_predict(embeddings)

    clustered_docs = [[] for _ in range(k)]
    for doc_meta, cluster_id in zip(metadata, clusters):
        clustered_docs[cluster_id].append(doc_meta)

    return clustered_docs

# Multi-Perspective Sampling (make subsets)
def multi_perspective_sampling(
    clusters: List[List[Dict[str, Any]]], m: int, seed: int = 1399
) -> List[List[str]]:
    """
    For each question in the doc, sample k documents from its retrieved list.

    Args
        clusters:
        m: The number of subsets
        seed: Random seed for reproducibility

    Returns:
        A list of lists, each containing the IDs of k sampled documents per question.
    """

    random.seed(seed)
    all_sampled_ids = []

    for item in clusters:
        if not isinstance(item, list):
            continue  # 혹시 item이 리스트가 아니면 skip

        doc_ids = [
            retrieved["doc_id"]
            for doc in item
            if isinstance(doc, dict)
            for retrieved in doc.get("retrieved_docs", [])
            if "doc_id" in retrieved
        ]

        if not doc_ids:
            continue

        sampled_ids = random.sample(doc_ids, 1)
        all_sampled_ids.append(sampled_ids)

    return all_sampled_ids
