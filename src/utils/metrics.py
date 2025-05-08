import re, string
import numpy as np
from sentence_transformers import SentenceTransformer, util

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
'''
def batched_select_best_choice(datalist): # Closed-set QA
    """
    datalist: list of dicts, each dict has 'generated_answer' and 'choices' keys
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')

    all_generated_answers = []
    all_choices = []

    # flatten choices
    for d in datalist:
        generated_answer = normalize_answer(d['generated_answer'])
        choices = d['choices']['text']
        all_generated_answers.append(generated_answer)
        all_choices.append(choices)

    # Encode all generated answers
    gen_embs = model.encode(all_generated_answers, convert_to_tensor=True, batch_size=64)

    # Encode all choices
    flat_choices = sum(all_choices, [])  # flatten: list of all choices
    choice_embs = model.encode(flat_choices, convert_to_tensor=True, batch_size=64)

    # Now, reshape choice_embs to [num_samples, num_choices, emb_dim]
    num_samples = len(datalist)
    num_choices = len(all_choices[0])  # assume fixed number, like 4
    emb_dim = choice_embs.shape[-1]

    choice_embs = choice_embs.reshape(num_samples, num_choices, emb_dim)

    correct = 0
    total = num_samples

    for i, d in enumerate(datalist):
        gen_emb = gen_embs[i]
        choices_emb = choice_embs[i]

        sims = util.cos_sim(gen_emb, choices_emb).squeeze(0)  # (4,)
        best_idx = sims.argmax().item()

        predicted_label = d['choices']['label'][best_idx]
        if predicted_label == d['ground_truth']:
            correct += 1

        print(f'{i+1}th QA...')
        print(f"ground truth: {d['ground_truth']}")
        print(f"predicted_label: {predicted_label}")

    acc = correct / total * 100 if total != 0 else 0
    print(f"\n🔑 Accuracy: {acc:.1f}%")
'''

from sentence_transformers import SentenceTransformer, util

def batched_select_best_choice(datalist):  # Closed-set QA
    """
    datalist: list of dicts, each dict has 'generated_answer' and 'choices' keys
    """
    model = SentenceTransformer('all-mpnet-base-v2')

    all_generated_answers = []
    flat_choices = []
    choice_lengths = []  # 각 샘플의 선택지 개수 저장

    for d in datalist:
        generated_answer = normalize_answer(d['generated_answer'])
        choices = d['choices']['text']

        all_generated_answers.append(generated_answer)
        flat_choices.extend(choices)
        choice_lengths.append(len(choices))

    # 임베딩 계산
    gen_embs = model.encode(all_generated_answers, convert_to_tensor=True, batch_size=64)
    choice_embs = model.encode(flat_choices, convert_to_tensor=True, batch_size=64)

    # 각 샘플별로 선택지 임베딩을 나눠서 평가
    correct = 0
    total = len(datalist)
    idx = 0  # flat choice_embs에서 위치 추적

    for i, d in enumerate(datalist):
        num_choices = choice_lengths[i]
        gen_emb = gen_embs[i]
        choices_emb = choice_embs[idx:idx+num_choices]
        idx += num_choices

        sims = util.cos_sim(gen_emb, choices_emb).squeeze(0)  # (num_choices,)
        best_idx = sims.argmax().item()

        predicted_label = d['choices']['label'][best_idx]
        if predicted_label == d['ground_truth']:
            correct += 1

        print(f'{i+1}th QA...')
        print(f"ground truth: {d['ground_truth']}")
        print(f"predicted_label: {predicted_label}")

    acc = correct / total * 100 if total != 0 else 0
    print(f"\n🔑 Accuracy: {acc:.1f}%")
'''
def batched_select_best_choice_exact_math(datalist):
    """
    datalist: list of dicts, each dict has:
        - 'generated_answer': model's answer as a string
        - 'choices': dict with 'text' (list of choices) and 'label' (e.g., ['A', 'B', 'C', 'D'])
        - 'ground_truth': the correct label (e.g., 'B')
    """
    correct = 0
    total = len(datalist)

    for i, d in enumerate(datalist):
        generated_answer = normalize_answer(d['generated_answer'])  # Normalize to improve matching
        choices = [normalize_answer(choice) for choice in d['choices']['text']]

        try:
            best_idx = choices.index(generated_answer)
            predicted_label = d['choices']['label'][best_idx]
        except ValueError:
            # generated_answer does not exactly match any of the choices
            predicted_label = None

        if predicted_label == d['ground_truth']:
            correct += 1

        print(f'{i+1}th QA...')
        print(f"ground truth: {d['ground_truth']}")
        print(f"predicted_label: {predicted_label}")

    acc = correct / total * 100 if total != 0 else 0
    print(f"\n🔑 Accuracy (Exact Match): {acc:.1f}%")
'''
def batched_select_best_choice_open(datalist): # Open-set QA
    """
    datalist: list of dicts, each dict has 'generated_answer', 'choices', and 'ground_truth' keys.
    Assumes 'choices' is a list of strings, and 'ground_truth' is a string.
    """
    model = SentenceTransformer('all-mpnet-base-v2')

    correct = 0
    total = len(datalist)

    for i, d in enumerate(datalist):
        generated_answer = normalize_answer(d['generated_answer'])
        ground_truth = normalize_answer(d['ground_truth'])

        if ground_truth in generated_answer:
            correct += 1

        #print(f'{i+1}th QA...')
        #print(f"ground truth: {d['ground_truth']}")
        #print(f"generated_answer: {d['generated_answer']}")
        #print(f"✅ Matched: {ground_truth in generated_answer}")

    acc = correct / total * 100 if total != 0 else 0
    print(f"\n🔑 Accuracy: {acc:.1f}%")
