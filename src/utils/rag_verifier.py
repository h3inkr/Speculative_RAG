



import numpy as np
import torch
import torch.nn.functional as F
import os
import contextlib
import math

from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
def load_model_silently(model_name: str, device: str):
    """
    Load the model and tokenizer while suppressing any stdout/stderr output.
    """
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16
            ).to(device)
    return tokenizer, model

def get_token_probs(full_input, device, tokenizer, model):
    """
    Returns token-level log-probs from a single forward pass.
    """
    inputs = tokenizer(full_input, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # shape: [1, seq_len, vocab_size]
    
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[0, :-1, :]  # predict token i+1 using position i

    chosen_tokens = input_ids[0, 1:]  # drop the first token for alignment
    chosen_log_probs = token_log_probs.gather(1, chosen_tokens.unsqueeze(-1)).squeeze(-1)
    
    return chosen_log_probs, input_ids[0]

def compute_segment_indices(prompt, segments, tokenizer):
    """
    Given the full prompt string and a list of segment strings [Q, α, β, R, "Yes"],
    returns a list of (start_idx, end_idx) for each segment in token space.
    """
    tokenized_prompt = tokenizer(prompt, add_special_tokens=False).input_ids
    indices = []
    offset = 0
    for segment in segments:
        tokenized_segment = tokenizer(segment, add_special_tokens=False).input_ids
        seg_len = len(tokenized_segment)
        # Find sublist starting from current offset
        while offset < len(tokenized_prompt):
            if tokenized_prompt[offset:offset + seg_len] == tokenized_segment:
                indices.append((offset, offset + seg_len))
                offset += seg_len
                break
            offset += 1
            
    return indices

def compute_rho_values(Q, alpha, beta, R, device, tokenizer, model):
    segments = [Q, alpha, beta, R, " Yes"]  # include leading space if needed
    full_prompt = f"Question: {Q}\nDraft: {alpha}\nRationale: {beta}\nReflection: {R}\nAnswer: Yes"
    
    log_probs, input_ids = get_token_probs(full_prompt, device, tokenizer, model)
    segment_indices = compute_segment_indices(full_prompt, segments, tokenizer)

    # Get log-probs for α, β, and "Yes"
    alpha_start, alpha_end = segment_indices[1]
    beta_start, beta_end = segment_indices[2]
    yes_start, yes_end = segment_indices[4]

    rho_sc = math.exp(log_probs[alpha_start:alpha_end].sum() +
                       log_probs[beta_start:beta_end].sum())

    rho_sr = math.exp(log_probs[yes_start:yes_end].sum())

    return rho_sc, rho_sr


def compute_score(
    question: str, verifier: str, answer: str, rationale: str, draft_score: str, device: str
) -> float:
    
    tokenizer, model = load_model_silently(verifier, device)
    
    R = "Is the rationale good enough to support the answer? (Yes or No)"
    
    self_contain, self_reflect = compute_rho_values(question, answer, rationale, R, device, tokenizer, model)
    #final_score = draft_score * self_contain * self_reflect
    #print(f"Self consistency score: {self_contain}")
    #print(f"Self reflection score: {self_reflect}")
    
    if any(math.isnan(x) for x in [draft_score, self_contain, self_reflect]):
        final_score = 0.0
    else:
        final_score = draft_score * self_contain * self_reflect
    #print(f"Final score: {final_score}")

    return final_score