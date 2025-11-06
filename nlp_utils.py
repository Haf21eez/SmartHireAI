# nlp_utils.py
"""
Lightweight NLP utilities for SmartHireAI.

This module avoids performing network downloads or heavy operations at import time
so Streamlit won't be blocked by missing NLTK resources. It provides safe
fallbacks for tokenization and stopwords when NLTK corpora aren't available.
"""
import re
from typing import List
import os

# Try to use NLTK if available, but don't crash at import time if resources are
# missing (this previously caused Streamlit to hang or show a blank page).
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    # Do NOT call nltk.download() here (network + UI at import time is bad).
    try:
        STOPWORDS = set(stopwords.words('english'))
    except Exception:
        STOPWORDS = set()
except Exception:
    # Minimal fallbacks
    STOPWORDS = set()

    def word_tokenize(text: str):
        # split on word characters
        return re.findall(r"\w+", text)

    def sent_tokenize(text: str):
        if not text:
            return []
        # naive sentence split on punctuation followed by whitespace
        return re.split(r'(?<=[.!?])\s+', text.strip())


def simple_clean(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^0-9A-Za-z\.\,\-\+\s#]', ' ', text)
    return text.lower()


def tokenize_words(text: str) -> List[str]:
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum() and t.lower() not in STOPWORDS]
    return tokens


def load_skills(skills_path: str = 'skills_list.txt') -> List[str]:
    if not os.path.exists(skills_path):
        return []
    with open(skills_path, 'r', encoding='utf-8') as f:
        skills = [line.strip().lower() for line in f if line.strip()]
    return skills


def extract_skills_from_text(text: str, skills_vocab: List[str]) -> List[str]:
    text_l = text.lower()
    found = set()
    for skill in skills_vocab:
        # match whole words and short forms
        skill_safe = skill.replace('.', r'\.')
        pattern = r'\b' + re.escape(skill_safe) + r'\b'
        if re.search(pattern, text_l):
            found.add(skill)
    return sorted(found)


def compute_match_score(resume_text: str, job_desc: str) -> float:
    # guard against None or empty inputs
    corpus = [resume_text or "", job_desc or ""]
    try:
        # import sklearn lazily so module import won't fail if sklearn is missing
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vectorizer = TfidfVectorizer(stop_words='english')
        tf = vectorizer.fit_transform(corpus)
        sim = cosine_similarity(tf[0:1], tf[1:2])[0][0]
        return float(sim)
    except Exception:
        # if sklearn is not available or TF-IDF fails, return 0.0
        return 0.0


def top_sentences_by_tfidf(text: str, top_n=2) -> List[str]:
    # Simple extractive ranking: TF-IDF over sentences
    sents = sent_tokenize(text)
    if not sents:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        tf = vectorizer.fit_transform(sents)
        # sentence score = sum of tfidf weights
        scores = tf.sum(axis=1).A1
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top = [sents[i] for i in ranked_idx[:top_n]]
        return top
    except Exception:
        # sklearn not available or TF-IDF failed: fallback to first N sentences
        return sents[:top_n]
