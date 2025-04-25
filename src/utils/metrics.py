import re, json, string
from tqdm import tqdm
import numpy as np
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

def select_best_choice_based_on_answer(d):
    """
    Select the most appropriate choice based on the generated answer.
    """
    generated_answer = d['generated_answer']
    choices = d['choices']
    
    # Prepare the text data for comparison
    choice_texts = choices['text']
    
    # Vectorize 'generated_answer' and choices using TF-IDF
    tfidf_vectorizer = TfidfVectorizer().fit_transform([generated_answer] + choice_texts)
    
    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(tfidf_vectorizer[0:1], tfidf_vectorizer[1:])
    
    # Select highest option
    best_choice_idx = cosine_similarities.argmax()
    best_choice = choice_texts[best_choice_idx]
    
    return best_choice, choices['label'][best_choice_idx]

def compute_metrics(data):
    correct = 0
    data_len = 0
    for d in tqdm(data):
        data_len += 1
        
        # Select the most appropriate choice based on the generated answer
        best_choice, best_choice_label = select_best_choice_based_on_answer(d)
        
        # Compare the selected choice with the ground truth
        if best_choice_label == d["ground_truth"]:
            correct += 1

    acc = correct / data_len * 100 if data_len != 0 else 0
    print(f"Accuracy: {acc:.1f}%")