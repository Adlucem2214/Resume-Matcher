"""
CLI: rank PDF resumes against a job (title + description required).
Usage (from project root):
  python scripts\rank_from_pdfs.py --title "Backend Dev" --description "python flask" --pdf C:\resumes\a.pdf C:\resumes\b.pdf
  python scripts\rank_from_pdfs.py --title "Dev" --description "C#" --pdf-dir C:\resumes_dir
Outputs JSON to stdout.
"""
from pathlib import Path
import sys
import argparse
import json
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(ROOT / "src"))

from src.core.pdf_parser_ocr import extract_text_from_pdf_path
from src.core.cleaning import clean_text_pipeline
from src.core.tfidf import TFIDFManager
from src.core.sbert import SBERTManager
from src.core.skills import extract_skills_from_text
from src.core.scoring import Scorer
from src.core.explain import Explainer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

DEFAULT_WEIGHTS = {"tfidf": 0.4, "sbert": 0.6}

def rank_job_against_texts(job_title: str, job_description: str, resume_texts: List[str], resume_ids: List[str], required_skills: Optional[List[str]] = None):
    job_raw = " ".join([job_title, job_description] + (required_skills or []))
    job_clean = clean_text_pipeline(job_raw)

    models_dir = ROOT / "src" / "app" / "models"
    tfidf_mgr = TFIDFManager(models_dir / "tfidf_vectorizer.pkl")
    sbert_mgr = SBERTManager()

    if tfidf_mgr.vectorizer is None:
        raise RuntimeError("TF-IDF vectorizer not trained. Run scripts/train_tfidf.py and ensure src/app/models/tfidf_vectorizer.pkl exists.")

    tfidf_sims = tfidf_mgr.cosine_similarities(job_clean, [clean_text_pipeline(t) for t in resume_texts])
    job_emb = sbert_mgr.embed_texts([job_clean])[0]
    cleaned_resumes = [clean_text_pipeline(t) for t in resume_texts]
    resume_embs = sbert_mgr.embed_texts(cleaned_resumes)
    sbert_sims = cosine_similarity([job_emb], resume_embs)[0]

    matched_list = []
    missing_list = []
    skill_scores = []
    for txt in cleaned_resumes:
        if required_skills:
            ex = extract_skills_from_text(txt, required_skills)
            matched_list.append(ex["matched"])
            missing_list.append(ex["missing"])
            skill_scores.append(len(ex["matched"]) / max(len(required_skills), 1))
        else:
            matched_list.append([])
            missing_list.append([])
            skill_scores.append(None)

    scorer = Scorer(DEFAULT_WEIGHTS)
    final_scores = [scorer.combine(float(tfidf_sims[i]), float(sbert_sims[i]), skill_scores[i]) for i in range(len(cleaned_resumes))]

    order = np.argsort(final_scores)[::-1]
    explainer = Explainer(tfidf_mgr)
    ranked = []
    for idx in order:
        ranked.append({
            "resume_id": resume_ids[idx],
            "final_score": float(final_scores[idx]),
            "tfidf_score": float(tfidf_sims[idx]),
            "sbert_score": float(sbert_sims[idx]),
            "skill_score": (skill_scores[idx] if skill_scores[idx] is not None else None),
            "matched_skills": matched_list[idx],
            "missing_skills": missing_list[idx],
            "top_tfidf_terms": explainer.top_tfidf_terms(job_clean, cleaned_resumes[idx], top_k=10)
        })
    return {"job_title": job_title, "job_description": job_description, "ranked": ranked}

def main(argv: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--pdf", nargs="*", help="paths to PDF files")
    parser.add_argument("--pdf-dir", help="directory with PDFs (recursive)")
    parser.add_argument("--skills", help="comma-separated required skills (optional)")
    args = parser.parse_args(argv)

    pdf_paths: List[Path] = []
    if args.pdf:
        pdf_paths += [Path(p) for p in args.pdf]
    if args.pdf_dir:
        pdf_paths += list(Path(args.pdf_dir).expanduser().rglob("*.pdf"))

    if not pdf_paths:
        print("No PDFs provided. Use --pdf <file> or --pdf-dir <dir>")
        sys.exit(1)

    texts = []
    ids = []
    for p in pdf_paths:
        txt = extract_text_from_pdf_path(p)
        
        # Debug: Save extracted text to verify content
        try:
            debug_out = p.with_name(f"{p.stem}_debug.txt")
            with open(debug_out, "w", encoding="utf-8") as f:
                f.write(txt or "")
        except Exception as e:
            print(f"Failed to save debug text for {p.name}: {e}", file=sys.stderr)

        texts.append(txt or "")
        ids.append(p.name)

    skills = [s.strip() for s in args.skills.split(",")] if args.skills else None

    out = rank_job_against_texts(args.title, args.description, texts, ids, required_skills=skills)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main(sys.argv[1:])