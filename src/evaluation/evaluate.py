from .metrics import precision_at_k, recall_at_k, mean_reciprocal_rank

def evaluate_single(ranked_labels, k):
    prec = precision_at_k(ranked_labels, k)
    rec = recall_at_k(ranked_labels, k, total_relevant=sum(ranked_labels))
    return {"precision@k": prec, "recall@k": rec}

def evaluate_many(all_ranked_labels, k):
    p_at_k = [precision_at_k(labels, k) for labels in all_ranked_labels]
    r_at_k = [recall_at_k(labels, k, total_relevant=sum(labels)) for labels in all_ranked_labels]
    mrr = mean_reciprocal_rank(all_ranked_labels)
    return {"mean_precision@k": sum(p_at_k)/len(p_at_k), "mean_recall@k": sum(r_at_k)/len(r_at_k), "mrr": mrr}