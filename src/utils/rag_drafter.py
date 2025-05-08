import numpy as np 
import torch 
torch.cuda.empty_cache()
torch.cuda.ipc_collect() 
import torch.nn.functional as F 
import re 
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4" 
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import contextlib
from typing import Dict, List, Tuple, Any

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel
from transformers.utils import logging as hf_logging

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = '1'

from transformers import AutoModelForCausalLM, BitsAndBytesConfig


# transformers 로깅 레벨을 ERROR로 설정: progress bar 끄기
hf_logging.set_verbosity_error()

# Redirect print outputs (stdout/stderr) when loading models
@contextlib.contextmanager
def suppress_transformers_output():
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield

from utils import extract_rationale_and_response, extract_json_response, compute_log_probability, generate_rho_draft
def generate_draft(question: str, answer: str, subset: List[Dict[str, Any]],
                   model, tokenizer, device: torch.device) -> Tuple[str, str, float]:

    context_text = "\n".join([d['text'] for d in subset])

    draft_prompt = f"""Response to the instruction. Also provide rationale for your response. 
## Instruction: {question}

## Evidence: {context_text}
"""

    pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(draft_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = output_text[len(draft_prompt):].strip()
    print(f"response: {response}/n/n/n/n/n")

    rationale, gen_response = extract_rationale_and_response(response)
    #rationale, gen_response = extract_json_response(response)

    rho = generate_rho_draft(question, [d['text'] for d in subset], rationale, gen_response, tokenizer, model, device)
    torch.cuda.empty_cache()
    return gen_response, rationale, rho