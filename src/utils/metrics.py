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

def exact_presence(answers, context):
    """Verify if any of the answers is present in the given context."""

    answers = [normalize_answer(ans) for ans in answers]
    context = normalize_answer(context)

    print(f"answers: {answers}")
    print(f"context: {context}")

    for ans in answers:
        if ans in context:
            return True

    return False

def get_metrics(data):
    idx = 0
    num_accurate = 0
    print('Evaluating results...')

    for d in tqdm(data):
        idx += 1
        print(f"ground_truth: {d['ground_truth']}")
        print(f"generated_answer: {d['generated_answer']}")
        is_accurate = exact_presence(d['ground_truth'], d['generated_answer'])
        num_accurate += 1 if is_accurate else 0

    accuracy = num_accurate / idx * 100
    print(f"Accuracy: {accuracy:.1f}%")