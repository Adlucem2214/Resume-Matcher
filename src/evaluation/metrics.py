from typing import List
import numpy as np

def precision_at_k(ranked_labels: List[int], k: int) -> float:
    # ranked_labels: 1 for relevant, 0 for not
    topk = ranked_labels[:k]
    return sum(topk) / k if k > 0 else 0.0

def recall_at_k(ranked_labels: List[int], k: int, total_relevant: int) -> float:
    topk = ranked_labels[:k]
    return sum(topk) / total_relevant if total_relevant > 0 else 0.0

def mean_reciprocal_rank(all_ranked_labels: List[List[int]]) -> float:
    rr = []
    for labels in all_ranked_labels:
        try:
            idx = labels.index(1)
            rr.append(1.0 / (idx + 1))
        except ValueError:
            rr.append(0.0)
    return float(np.mean(rr)) if rr else 0.0