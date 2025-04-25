import numpy as np
import torch
import torch.nn.functional as F
import re
import os
import contextlib

from typing import List, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel

from transformers.utils import logging as hf_logging

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# transformers 로깅 레벨을 ERROR로 설정: progress bar 끄기
hf_logging.set_verbosity_error()

# Redirect print outputs (stdout/stderr) when loading models
@contextlib.contextmanager
def suppress_transformers_output():
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield

from utils import extract_rationale_and_response, compute_log_probability, generate_rho_draft

def generate_draft(
    question: str, answer:str, drafter_path: str, subset: List[List[str]], metadata:list, device:str
) -> Tuple[str, str, float]:

    # drafter_path가 디렉토리이고 adapter_config.json이 있으면 사전학습된 PEFT 모델로 간주
    is_peft_model = os.path.isdir(drafter_path) and os.path.exists(os.path.join(drafter_path, "adapter_config.json"))

    if device.type == "cuda" and torch.cuda.is_available():
        gpu_index = device.index
        device_map = {"": gpu_index}
    else:
        device_map = {"": "cpu"} 

    with suppress_transformers_output():
        if is_peft_model:
            peft_config = PeftConfig.from_pretrained(drafter_path)               
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                torch_dtype=torch.float16,
                device_map=device_map,
            )
            draft_model = PeftModel.from_pretrained(base_model, drafter_path)
            tokenizer = AutoTokenizer.from_pretrained(drafter_path)
        else:
            draft_model = AutoModelForCausalLM.from_pretrained(
                drafter_path,
                torch_dtype=torch.float16,
                device_map=device_map,
            )
            tokenizer = AutoTokenizer.from_pretrained(drafter_path)

    tokenizer.pad_token = tokenizer.eos_token
    draft_model.eval()

    # Incorporate each document subset into the prompt
    subset_ids = [id_[0] for id_ in subset]

    # 모든 retrieved_docs에서 id가 subset_ids에 있는 text만 추출
    texts = ""
    n = 0
    for item in metadata:
        if item.get("doc_id") in subset_ids:
            n += 1
            texts += f"[{n}] {item.get('doc_title', '')}\n{item.get('texts', '')}\n"

    question = re.search(r"## Input:\s*(.*)", question, re.DOTALL).group(1).strip()
    ## RAG Drafting           
    draft_prompt = f"""Response to the instruction. Also provide rationale for your response. 

    ## Instruction: {question}

    ## Evidence:
    {texts}
    """ 

    draft_inputs = tokenizer(draft_prompt, return_tensors="pt").to(device)  
    draft_outputs = draft_model.generate(
        **draft_inputs,
        max_new_tokens=512,
        #temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )  

    output = tokenizer.decode(draft_outputs[0], skip_special_tokens=True)
    response = output[len(draft_prompt):].strip()

    # Extract the rationale and response from the output
    rationale, generated_response = extract_rationale_and_response(response)
    rho = generate_rho_draft(question, texts, rationale, generated_response, tokenizer, draft_model, device)
    
    return generated_response, rationale, rho