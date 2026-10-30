from typing import Optional, Dict

class Scorer:
    def __init__(self, weights: Dict[str, float]):
        # expected keys: 'tfidf', 'sbert', 'skill' (skill optional)
        self.weights = weights
        # Normalize weights (for provided keys only)
        total = sum(v for v in weights.values() if v is not None)
        if total > 0:
            self.weights = {k: v / total for k, v in weights.items()}

    def combine(self, tfidf_score: float, sbert_score: float, skill_score: Optional[float] = None) -> float:
        w_t = self.weights.get("tfidf", 0.0)
        w_s = self.weights.get("sbert", 0.0)
        w_k = self.weights.get("skill", 0.0)
        # if skill_score is None, ignore weight by scaling others
        if skill_score is None:
            denom = (w_t + w_s) if (w_t + w_s) > 0 else 1.0
            return (tfidf_score * w_t + sbert_score * w_s) / denom
        else:
            return tfidf_score * w_t + sbert_score * w_s + skill_score * w_k