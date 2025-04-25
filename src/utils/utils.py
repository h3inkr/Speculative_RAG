from typing import List, Dict, Tuple, Any
import torch
import torch.nn.functional as F
import numpy as np

def extract_rationale_and_response(response: str) -> Tuple[str, str]:
    rationale_start = response.find("## Rationale:")
    response_start = response.find("## Response:")

    if rationale_start == -1 and response_start == -1:
        # 둘 다 없으면 전부 그냥 response로 처리
        return "", response.strip()
    
    if rationale_start != -1 and (response_start == -1 or rationale_start < response_start):
        # Rationale 먼저 나옴
        rationale = response[rationale_start + len("## Rationale:"):response_start].strip() if response_start != -1 else response[rationale_start + len("## Rationale:"):].strip()
        generated_response = response[response_start + len("## Response:"):].strip() if response_start != -1 else ""
    
    elif response_start != -1 and (rationale_start == -1 or response_start < rationale_start):
        # Response 먼저 나옴
        generated_response = response[response_start + len("## Response:"):rationale_start].strip() if rationale_start != -1 else response[response_start + len("## Response:"):].strip()
        rationale = response[rationale_start + len("## Rationale:"):].strip() if rationale_start != -1 else ""
    
    return rationale, generated_response

def compute_log_probability(prompt: str, continuation: str, tokenizer: str, model: str, device: str) -> float:
    """
    Computes the log probability of the `continuation` given the `prompt`
    """
    input_text = prompt + continuation
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :]  # Shift for causal LM
        target_ids = input_ids[:, 1:]       # Predict next token

        # Get log probs
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

        # Only sum the logprobs corresponding to continuation part
        prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        continuation_log_probs = target_log_probs[0, prompt_len-1:]  # -1 to align shift

        return np.exp(continuation_log_probs.mean().item())  # log probability total

def generate_rho_draft(Q: str, docs: list[str], rationale: str, answer: str, tokenizer: str, model: str, device: str) -> float:
    context = "\n".join(docs)

    rationale_prompt = f"Question: {Q}\nDocuments:\n{context}\nRationale:"
    answer_prompt = f"Question: {Q}\nDocuments:\n{context}\nRationale: {rationale}\nAnswer:"

    logp_rationale = compute_log_probability(rationale_prompt, rationale, tokenizer, model, device)
    logp_answer = compute_log_probability(answer_prompt, answer, tokenizer, model, device)

    rho_draft = logp_rationale + logp_answer
    return rho_draft