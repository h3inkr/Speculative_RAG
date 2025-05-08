from typing import List, Dict, Tuple, Any
import torch
import torch.nn.functional as F
import numpy as np
import math
import sys
import os
'''
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
'''

def extract_rationale_and_response(response: str) -> Tuple[str, str]:
    rationale_header = "## Rationale:"
    response_header  = "## Response:"
    rationale_start  = response.find(rationale_header)
    response_start   = response.find(response_header)

    # 둘 다 없으면 통째로 Response
    if response_start == -1 and rationale_start == -1:
        return "", response.strip()

    # Response 헤더만 있는 경우
    if response_start != -1 and rationale_start == -1:
        generated_response = response[response_start + len(response_header):].strip()
        return "", generated_response

    # Rationale 헤더만 있는 경우 (헤더 앞 부분을 Response로)
    if rationale_start != -1 and response_start == -1:
        generated_response = response[:rationale_start].strip()
        rationale = response[rationale_start + len(rationale_header):].strip()
        return rationale, generated_response

    # 둘 다 있는 경우
    # — Response가 먼저 나오는 경우
    if response_start < rationale_start:
        generated_response = response[response_start + len(response_header):rationale_start].strip()
        rationale          = response[rationale_start + len(rationale_header):].strip()
    # — Rationale이 먼저 나오는 경우
    else:
        rationale          = response[rationale_start + len(rationale_header):response_start].strip()
        generated_response = response[response_start + len(response_header):].strip()

    return rationale, generated_response

def extract_json_response(output: str) -> Tuple[str, str]:
    """
    Extracts 'rationale' and 'response' fields from a JSON-formatted output string.
    """
    try:
        # JSON 부분만 추출 
        json_start = output.find("{")
        json_str = output[json_start:]
        parsed = json.loads(json_str)
        rationale = parsed.get("rationale", "").strip()
        response = parsed.get("response", "").strip()
    except Exception:
        rationale, response = "", ""
    return rationale, response

def compute_log_probability(prompt: str, continuation: str,
                            tokenizer, model, device) -> float:
    input_text = prompt + continuation
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :]
        target_ids = input_ids[:, 1:]

        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

        prompt_len = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"].shape[1]
        continuation_log_probs = target_log_probs[0, prompt_len - 1:]

        return continuation_log_probs.sum().item()
'''
def generate_rho_draft(Q: str, docs: list[str], rationale: str, answer: str,
                       tokenizer, model, device) -> float:
    context = "\n".join(docs)

    rationale_prompt = f"Question: {Q}\nDocuments:\n{context}\nRationale:"
    answer_prompt = f"Question: {Q}\nDocuments:\n{context}\nRationale: {rationale}\nAnswer:"

    logp_rationale = compute_log_probability(rationale_prompt, rationale, tokenizer, model, device)
    logp_answer = compute_log_probability(answer_prompt, answer, tokenizer, model, device)

    return logp_rationale + logp_answer
'''
def generate_rho_draft(Q: str, docs: list[str], rationale: str, answer: str,
                       tokenizer, model, device) -> float:
    context = "\n".join(docs)

    rationale_prompt = f"Question: {Q}\nDocuments:\n{context}\nRationale:"
    answer_prompt = f"Question: {Q}\nDocuments:\n{context}\nRationale: {rationale}\nAnswer:"

    logp_rationale = compute_log_probability(rationale_prompt, rationale, tokenizer, model, device)
    logp_answer = compute_log_probability(answer_prompt, answer, tokenizer, model, device)

    # 확률로 변환
    prob_rationale = math.exp(logp_rationale)
    prob_answer = math.exp(logp_answer)

    # 확률 평균 (또는 가중합도 가능)
    rho_score = (prob_rationale + prob_answer) / 2
    return rho_score

def load_drafter_model(drafter_path: str, device: torch.device):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel, PeftConfig

    is_peft = os.path.isdir(drafter_path) and os.path.exists(os.path.join(drafter_path, "adapter_config.json"))

    if is_peft:	
        peft_config = PeftConfig.from_pretrained(drafter_path)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            quantization_config=bnb_config,
            device_map={"": device.index if device.type == "cuda" else "cpu"},
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, drafter_path)
    else:
            model = AutoModelForCausalLM.from_pretrained(
            drafter_path,
            torch_dtype=torch.float16,
            device_map={"": device.index if device.type == "cuda" else "cpu"},
        )

    tokenizer = AutoTokenizer.from_pretrained(drafter_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer

