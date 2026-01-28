"""
Run evaluation using labeled job-resume relevance pairs.

Usage:
    python -m src.evaluation.run_evaluation
    or
    python src\evaluation\run_evaluation.py
"""
from pathlib import Path
import sys
import pandas as pd
from typing import List, Dict
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from src.core.tfidf import TFIDFManager
from src.core.sbert import SBERTManager
from src.core.cleaning import clean_text_pipeline
from src.core.skills import extract_skills_from_text
from src.core.scoring import Scorer
from src.evaluation.metrics import precision_at_k, recall_at_k, mean_reciprocal_rank
from sklearn.metrics.pairwise import cosine_similarity

def build_job_text(row: pd.Series) -> str:
    parts = [str(row.get("Title", "")), str(row.get("Responsibilities", "")), str(row.get("ExperienceLevel", ""))]
    for f in ("Skills", "Keywords"):
        val = row.get(f, None)
        if pd.notna(val) and val:
            parts.append(str(val))
    return " ".join([p for p in parts if p])

def load_resume_map(resume_csv: Path) -> Dict[str, str]:
    df = pd.read_csv(resume_csv)
    if "resume_id" not in df.columns or "resume_text" not in df.columns:
        raise RuntimeError("resume CSV must contain 'resume_id' and 'resume_text' columns")
    return dict(zip(df["resume_id"].astype(str), df["resume_text"].astype(str)))

def main() -> None:
    data_root = ROOT / "data"
    labels_path = data_root / "evaluation" / "labels.csv"
    jobs_path = data_root / "jobs" / "jobs.csv"
    resumes_path = data_root / "resumes" / "train.csv"

    if not labels_path.exists():
        raise FileNotFoundError(f"labels.csv not found at {labels_path}")
    if not jobs_path.exists():
        raise FileNotFoundError(f"jobs.csv not found at {jobs_path}")
    if not resumes_path.exists():
        raise FileNotFoundError(f"resumes train.csv not found at {resumes_path}")

    labels_df = pd.read_csv(labels_path)
    jobs_df = pd.read_csv(jobs_path)
    resume_map = load_resume_map(resumes_path)

    models_dir = ROOT / "src" / "app" / "models"
    tfidf_mgr = TFIDFManager(models_dir / "tfidf_vectorizer.pkl")
    sbert_mgr = SBERTManager()

    grouped = labels_df.groupby("job_id")
    all_prec_k5, all_rec_k5, all_prec_k10, all_rec_k10, all_rr = [], [], [], [], []

    weights = {"tfidf": 0.4, "sbert": 0.6}
    scorer = Scorer(weights)

    for job_id, group in grouped:
        job_row = jobs_df[jobs_df["JobID"].astype(str) == str(job_id)]
        if job_row.empty:
            continue
        job_row = job_row.iloc[0]
        job_text_raw = build_job_text(job_row)
        job_clean = clean_text_pipeline(job_text_raw)

        resume_ids = group["resume_id"].astype(str).tolist()
        resume_texts_raw = [resume_map.get(rid, "") for rid in resume_ids]
        cleaned_texts = [clean_text_pipeline(t) for t in resume_texts_raw]

        tfidf_sims = tfidf_mgr.cosine_similarities(job_clean, cleaned_texts)
        job_emb = sbert_mgr.embed_texts([job_clean])[0]
        resume_embs = sbert_mgr.embed_texts(cleaned_texts)
        sbert_sims = cosine_similarity([job_emb], resume_embs)[0]

        skills_field = job_row.get("Skills", None)
        job_skills = []
        if pd.notna(skills_field) and skills_field:
            job_skills = [s.strip() for s in str(skills_field).replace(";", ",").split(",") if s.strip()]

        skill_scores = []
        for txt in cleaned_texts:
            if job_skills:
                ex = extract_skills_from_text(txt, job_skills)
                skill_scores.append(len(ex["matched"]) / max(len(job_skills), 1))
            else:
                skill_scores.append(None)

        final_scores = [scorer.combine(float(t), float(s), k) for t, s, k in zip(tfidf_sims, sbert_sims, skill_scores)]
        sorted_indices = list(np.argsort(final_scores)[::-1])

        relevance_map = {str(rid): int(lbl) for rid, lbl in zip(group["resume_id"].astype(str), group["relevant"])}
        ranked_labels = [relevance_map.get(resume_ids[i], 0) for i in sorted_indices]

        prec5 = precision_at_k(ranked_labels, 5)
        rec5 = recall_at_k(ranked_labels, 5, total_relevant=sum(ranked_labels))
        prec10 = precision_at_k(ranked_labels, 10)
        rec10 = recall_at_k(ranked_labels, 10, total_relevant=sum(ranked_labels))
        all_prec_k5.append(prec5); all_rec_k5.append(rec5)
        all_prec_k10.append(prec10); all_rec_k10.append(rec10)

        try:
            idx = ranked_labels.index(1)
            all_rr.append(1.0 / (idx + 1))
        except ValueError:
            all_rr.append(0.0)

    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    print("Jobs evaluated:", len(all_prec_k5))
    print(f"Precision@5 (mean): {mean(all_prec_k5):.4f}")
    print(f"Recall@5    (mean): {mean(all_rec_k5):.4f}")
    print(f"Precision@10 (mean): {mean(all_prec_k10):.4f}")
    print(f"Recall@10    (mean): {mean(all_rec_k10):.4f}")
    print(f"Mean Reciprocal Rank (MRR): {mean(all_rr):.4f}")

if __name__ == "__main__":
    main()