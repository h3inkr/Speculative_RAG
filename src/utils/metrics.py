import re, json, string
from tqdm import tqdm
import numpy as np
from collections import Counter

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_metrics(data):
    correct = 0
    data_len = 0
    for d in tqdm(data):
        data_len += 1
        idx = d["choices"]["label"].index(d["answerKey"])
        truth = normalize_answer(d["choices"]["text"][idx])
        answer = normalize_answer(d['generated_answer'])
        if truth == answer:
            correct += 1
    
    acc = correct / data_len * 100 if data_len !=0 else 0

    print(f"Accuracy: {acc:.1f}%")