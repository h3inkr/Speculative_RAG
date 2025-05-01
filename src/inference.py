import numpy as np
import torch
import argparse
import faiss
import pickle
import re
from tqdm import tqdm
from itertools import islice
import logging

import sys
sys.path.append('./src/utils')
from multi_perspective import documents_to_clusters, multi_perspective_sampling
from rag_drafter import generate_draft
from rag_verifier import compute_score
from metrics import batched_select_best_choice, batched_select_best_choice_open

# Set up logging
logging.basicConfig(
    filename='evaluation_log_51.txt',  # Log file name
    level=logging.INFO,  # Log level
    format='%(asctime)s - %(levelname)s - %(message)s'
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", "-ip", type=str, required=True, help="faiss index")
    parser.add_argument("--meta_path", "-mp", type=str, required=True) 
    parser.add_argument("--k", "-k", type=int, default=2) # draft based k documents, 2
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
        retrieved_docs = item.get("retrieved_docs")
        if q and a:
            qa_pairs[q] = {
                "answer": a,
                "choices": choices, 
                "docs": retrieved_docs
            }

    if args.drafter_path:
        drafter_path = args.drafter_path
    else:
        drafter_path = args.drafter

    correct = 0
    total = 0
    outputs = []
    logging.info(f"\n❗ Total QA Pairs: {len(qa_pairs)}\n")
    #for question, others in tqdm(islice(qa_pairs.items(), 10), desc="QA Evaluation", unit="pair"): # test
    for question, others in tqdm(qa_pairs.items(), desc="QA Evaluation", unit="pair"):

        #print(f"question: {question}")
        #print(f"ground_truth: {answer}")

        answer = others["answer"]
        choices = others["choices"]
        docs = others["docs"]
        #print(f"docs: {docs}")

        clusters = documents_to_clusters(docs=docs, m=args.m)
        subsets = multi_perspective_sampling(clusters=clusters, k=args.k)
        # print(f"clusters: {clusters}")
        #print(f"subsets: {subsets}")


        #print(f"args k is ... {args.k}")
  
        
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
        logging.info(f"best_idx: {best_idx}")
        logging.info(f"best_response: {best_response}")
    
        d = {
            "ground_truth": answer,
            "choices": choices,
            "generated_answer": best_response,
            }
        outputs.append(d)
        
    #batched_select_best_choice(outputs) # Closed-set QA
    batched_select_best_choice_open(outputs) # Open-set QA
