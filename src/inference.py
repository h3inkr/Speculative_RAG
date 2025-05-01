import numpy as np
import torch
import argparse
import faiss
import pickle
import re
from tqdm import tqdm
from itertools import islice

import sys
sys.path.append('./src/utils')
from multi_perspective import documents_to_clusters, multi_perspective_sampling
from rag_drafter import generate_draft
from rag_verifier import compute_score
from metrics import batched_select_best_choice, batched_select_best_choice_open

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", "-ip", type=str, required=True, help="faiss index")
    parser.add_argument("--meta_path", "-mp", type=str, required=True) 
    parser.add_argument("--k", "-k", type=int, default=2) # the number of clusters, 2
    parser.add_argument("--m", "-m", type=int, default=5) # the number of drafts, 5
    parser.add_argument("--drafter", "-dr", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--drafter_path", "-dp", type=str, default=None, help="Path to locally fine-tuned RAG drafter directory")
    parser.add_argument("--verifier", "-vr", type=str, default="mistralai/Mistral-7B-v0.1") # mistralai/Mistral-7B-v0.1 or mistralai/Mixtral-8x7B-v0.1
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_argument()

    # Load embedding
    index = faiss.read_index(args.index_path)
    with open(args.meta_path, "rb") as f:
        metadata = pickle.load(f)

    # Extract instruction-following QA pairs
    qa_pairs = {}
    for item in metadata:
        q = item.get("question")
        a = item.get("answer")
        choices = item.get("choices")
        if q and a:
            qa_pairs[q] = {
                "answer": a,
                "choices": choices 
            }

    if args.drafter_path:
        drafter_path = args.drafter_path
    else:
        drafter_path = args.drafter

    correct = 0
    total = 0
    outputs = []
    print(f"\n❗ Total QA Pairs: {len(qa_pairs)}\n")
    #for question, answithchoi in tqdm(islice(qa_pairs.items(), 5), desc="QA Evaluation", unit="pair"): # test
    for question, answithchoi in tqdm(qa_pairs.items(), desc="QA Evaluation", unit="pair"):

        answer = answithchoi["answer"]
        choices = answithchoi["choices"]

        clusters = documents_to_clusters(index_path=args.index_path, meta_path=args.meta_path, k=args.k)
        subsets = multi_perspective_sampling(clusters=clusters, m=args.m)

        drafts = []
        generated_answer = ""
        for i, subset in enumerate(subsets):
            response, rationale, draft_score = generate_draft(
                question=question, 
                answer=answer, 
                drafter_path=drafter_path,
                subset=subset, 
                metadata=metadata, 
                device=device)
            drafts.append((response, rationale, draft_score))
            generated_answer += f"{response}"
        print(f'drafts: {drafts}')
        print(f'number of draft: {len(drafts)}')

        scores = []
        for i, (response, rationale, draft_score) in enumerate(drafts):
            score = compute_score(
                question=question, 
                verifier=args.verifier,
                answer=response, 
                rationale=rationale, 
                draft_score=draft_score,
                device=device)
            scores.append(score)

        best_idx = np.argmax(scores)
        best_response = drafts[best_idx][0]
        print(f"best_idx: {best_idx}")
        print(f"best_response: {best_response}")
    
        d = {
            "ground_truth": answer,
            "choices": choices,
            "generated_answer": best_response,
            }
        outputs.append(d)
        
    #batched_select_best_choice(outputs) # Closed-set QA
    batched_select_best_choice_open(outputs) # Open-set QA
