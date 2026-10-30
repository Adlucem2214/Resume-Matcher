from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np

class SBERTManager:
    """
    SBERT manager that keeps a pretrained SentenceTransformer and
    an in-memory cache of text -> embedding to avoid recomputation.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self._cache: Dict[str, np.ndarray] = {}

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts. Uses cache for previously seen texts.
        Returns a 2D numpy array shape (len(texts), embedding_dim).
        """
        to_compute: List[str] = []
        compute_indices: List[int] = []
        embeddings: List[np.ndarray] = [None] * len(texts)  # type: ignore

        for i, t in enumerate(texts):
            if t in self._cache:
                embeddings[i] = self._cache[t]
            else:
                to_compute.append(t)
                compute_indices.append(i)

        if to_compute:
            new_embs = self.model.encode(to_compute, convert_to_numpy=True, show_progress_bar=False)
            # assign back and update cache
            for idx, emb in zip(compute_indices, new_embs):
                embeddings[idx] = emb
                self._cache[to_compute[compute_indices.index(idx)]] = emb

        # final stack
        return np.vstack(embeddings)