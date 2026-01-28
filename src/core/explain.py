from src.core.tfidf import TFIDFManager
import numpy as np

class Explainer:
    def __init__(self, tfidf_manager: TFIDFManager):
        self.tfidf = tfidf_manager

    def top_tfidf_terms(self, job_text: str, resume_text: str, top_k: int = 10):
        """
        Compute element-wise product of tfidf vectors and pick top contributing terms.
        """
        X = self.tfidf.vectorizer.transform([job_text, resume_text])
        job_vec = X[0].toarray().ravel()
        res_vec = X[1].toarray().ravel()
        contrib = job_vec * res_vec  # high when both high
        if contrib.sum() == 0:
            return []
        idx = np.argsort(contrib)[::-1][:top_k]
        terms = self.tfidf.get_feature_names()
        top = []
        for i in idx:
            if contrib[i] > 0:
                top.append((terms[i], float(contrib[i])))
        return top