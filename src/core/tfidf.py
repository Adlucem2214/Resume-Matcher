import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class TFIDFManager:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.vectorizer = None
        if self.model_path.exists():
            self.load()
        else:
            # user should run train_from_corpus(...) to create vectorizer
            self.vectorizer = None

    def train_from_corpus(self, texts: List[str]):
        # bigrams + trigrams (updated per request)
        vect = TfidfVectorizer(ngram_range=(2,3), max_features=50000, analyzer="word")
        vect.fit(texts)
        self.vectorizer = vect
        self.save()

    # def train_from_corpus(self, texts: List[str]):
    #     # unigrams + bigrams
    #     vect = TfidfVectorizer(ngram_range=(1,2), max_features=50000)
    #     vect.fit(texts)
    #     self.vectorizer = vect
    #     self.save()

    def save(self):
        with open(self.model_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self):
        with open(self.model_path, "rb") as f:
            self.vectorizer = pickle.load(f)

    def transform(self, texts: List[str]):
        if self.vectorizer is None:
            raise RuntimeError("TF-IDF vectorizer not trained/loaded.")
        return self.vectorizer.transform(texts)

    def transform_single(self, text: str):
        return self.transform([text])[0]

    def cosine_similarities(self, job_text: str, resume_texts: List[str]):
        # compute job and resumes tfidf then cosine similarity
        if self.vectorizer is None:
            raise RuntimeError("TF-IDF vectorizer not trained/loaded.")
        X = self.vectorizer.transform([job_text] + resume_texts)
        job_vec = X[0]
        resumes = X[1:]
        sims = cosine_similarity(job_vec, resumes)[0]
        return sims

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()