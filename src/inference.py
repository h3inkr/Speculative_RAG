import numpy as np
import torch
torch.cuda.empty_cache()
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
from rag_verifier import load_model_silently, compute_score
from metrics import batched_select_best_choice, batched_select_best_choice_open
from embedding import DPR, BasicRAG
from utils import load_drafter_model

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = '1'

# Set up logging
logging.basicConfig(
    filename='evaluation_log_56.txt',  # Log file name
    level=logging.INFO,  # Log level
    format='%(asctime)s - %(levelname)s - %(message)s'
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", "-ip", type=str, required=True, help="faiss index")
    parser.add_argument("--meta_path", "-mp", type=str, required=True) 
    parser.add_argument("--k", "-k", type=int, default=5) # draft based k documents, 2
    parser.add_argument("--m", "-m", type=int, default=2) # the number of drafts, 5
    parser.add_argument("--drafter", "-dr", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--drafter_path", "-dp", type=str, default=None, help="Path to locally fine-tuned RAG drafter directory")
    parser.add_argument("--verifier", "-vr", type=str, default="mistralai/Mistral-7B-v0.1") # mistralai/Mistral-7B-v0.1 or mistralai/Mixtral-8x7B-v0.1
    parser.add_argument("--model_name_or_path", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--retrieval_model_name_or_path", default="/mnt2/user4/coconut_documents/question_encoder")
    parser.add_argument("--embedding_path", default="/mnt2/user4/coconut_documents/")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_argument()

    rag = BasicRAG(args)
 
 # Load embedding
    index = faiss.read_index(args.index_path)
    with open(args.meta_path, "rb") as f:
        metadata = pickle.load(f)

    # Extract instruction-following QA pairs
    qa_pairs = {}
    for item in metadata:
   # for idx, item  in enumerate(metadata):
   #     if idx >= 5:
   #         break
        q = item.get("question")
        a = item.get("answer")
        choices = item.get("choices")
       # retrieved_docs = item.get("retrieved_docs")
        retrieved_docs = rag.retrieve(query=q, topk=10)
        if q and a:
            qa_pairs[q] = {
                "answer": a,
                "choices": choices, 
                "docs": retrieved_docs
            }
   # print(f"qa_pairs: {qa_pairs}")
    if args.drafter_path:
        drafter_path = args.drafter_path
    else:
        drafter_path = args.drafter

    drafter_model, drafter_tokenizer = load_drafter_model(drafter_path, device)
    verifier_model, verifier_tokenizer = load_model_silently(args.verifier, device)

    correct = 0
    total = 0
    outputs = []
    logging.info(f"\n❗ Total QA Pairs: {len(qa_pairs)}\n")
    #for question, others in tqdm(islice(qa_pairs.items(), 5), desc="QA Evaluation", unit="pair"): # test
    for question, others in tqdm(qa_pairs.items(), desc="QA Evaluation", unit="pair"):
        logging.info(f"Question: {question}\n")

        answer = others["answer"]
        logging.info(f"Answer: {answer}\n")

        choices = others["choices"]
        logging.info(f"Choices: {choices}\n")

        docs = others["docs"]

        clusters = documents_to_clusters(docs=docs, m=args.m)
        logging.info(f"clusters: {clusters}\n")
        subsets = multi_perspective_sampling(clusters=clusters, k=args.k)
        logging.info(f"subsets: {subsets}\n")
  
        drafts = []
        generated_answer = ""
        for i, subset in enumerate(subsets):
            response, rationale, draft_score = generate_draft(
                question=question, 
                answer=answer, 
                subset=subset, 
                model=drafter_model,
                tokenizer=drafter_tokenizer, 
                device=device)
            drafts.append((response, rationale, draft_score))
            generated_answer += f"{response}"
        logging.info(f'drafts: {drafts}')
        
        scores = []
        for i, (response, rationale, draft_score) in enumerate(drafts):
            score = compute_score(
                answer=response, 
                rationale=rationale, 
                draft_score=draft_score,
                question =question,
                device=device,
                tokenizer=verifier_tokenizer,
                model=verifier_model)
            scores.append(score)
        logging.info(f"scores: {scores}")
        
        best_idx = np.argmax(scores)
        best_response = drafts[best_idx][0]
        best_rationale = drafts[best_idx][1]
        logging.info(f"best_idx: {best_idx}")
        logging.info(f"best_response: {best_response}")
        logging.info(f"best_rationale: {best_rationale}")
    
        d = {
            "ground_truth": answer,
            "choices": choices,
            #"generated_answer": best_response,
            "generated_answer": best_response
            }
        outputs.append(d)
        
    batched_select_best_choice(outputs) # Closed-set QA
   # batched_select_best_choice_open(outputs) # Open-set QA (piqa)
    print("39 w/o finetuning")
