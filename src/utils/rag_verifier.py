import numpy as np
import torch
import torch.nn.functional as F
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4" 
import contextlib
import math

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

def load_model_silently(model_name: str, device: str):
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            model.gradient_checkpointing_enable()

    return model, tokenizer

def get_token_probs(full_input, device, tokenizer, model):
    inputs = tokenizer(full_input, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    with torch.no_grad():
        outputs = model(**input_ids)
        logits = outputs.logits  # shape: [1, seq_len, vocab_size]
    
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[0, :-1, :]  # predict token i+1 using position i

    chosen_tokens = input_ids[0, 1:]
    chosen_log_probs = token_log_probs.gather(1, chosen_tokens.unsqueeze(-1)).squeeze(-1)
    
    return chosen_log_probs, input_ids[0]

def compute_rho_values(Q, alpha, beta, R, device, tokenizer, model):

    q_ids = tokenizer("Question: " + Q, add_special_tokens=False, return_tensors="pt")["input_ids"]
    a_ids = tokenizer("Draft: " + alpha, add_special_tokens=False, return_tensors="pt")["input_ids"]
    b_ids = tokenizer("Rationale: " + beta, add_special_tokens=False, return_tensors="pt")["input_ids"]
    r_ids = tokenizer("Reflection: " + R, add_special_tokens=False, return_tensors="pt")["input_ids"]
    y_ids = tokenizer("Answer: Yes", add_special_tokens=False, return_tensors="pt")["input_ids"]

    input_ids = torch.cat([q_ids, a_ids, b_ids, r_ids, y_ids], dim=1).to("cuda:1")  # shape: [1, total_seq_len]
    model.eval()
    model = model.to("cuda:1")

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[0, :-1, :]  # shift
    chosen_tokens = input_ids[0, 1:]        # shift
    chosen_log_probs = token_log_probs.gather(1, chosen_tokens.unsqueeze(-1)).squeeze(-1)

    # Range 계산
    q_len = q_ids.shape[1]
    a_len = a_ids.shape[1]
    b_len = b_ids.shape[1]
    y_len = y_ids.shape[1]

    alpha_range = (q_len, q_len + a_len)
    beta_range = (alpha_range[1], alpha_range[1] + b_len)
    yes_range = (input_ids.shape[1] - y_len, input_ids.shape[1])

    # rho 값 계산
    rho_sc = math.exp(chosen_log_probs[alpha_range[0] - 1:alpha_range[1] - 1].sum().item() +
                      chosen_log_probs[beta_range[0] - 1:beta_range[1] - 1].sum().item())
    rho_sr = math.exp(chosen_log_probs[yes_range[0] - 1:yes_range[1] - 1].sum().item())

    return rho_sc, rho_sr


def compute_score(answer, rationale, draft_score, question, device, tokenizer, model):

    if answer == '' or rationale == '': # response와 rationale 둘 중 하나라도 빈 string이면 후보에서 제외
        return 0.0
    else: 
        R = "Is the rationale good enough to support the answer? (Yes or No)"
        self_contain, self_reflect = compute_rho_values(question, answer, rationale, R, device, tokenizer, model)

        if any(math.isnan(x) for x in [draft_score, self_contain, self_reflect]):
            return 0.0
        else:
            score = draft_score * self_contain * self_reflect
            torch.cuda.empty_cache()
            return score

