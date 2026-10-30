from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from typing import List, Optional, Any
from src.models.request_response import JobInput, ResumeInput, RankRequest, RankResponse, RankedResume
from src.core.cleaning import clean_text_pipeline
from src.core.tfidf import TFIDFManager
from src.core.sbert import SBERTManager
from src.core.skills import extract_skills_from_text
from src.core.scoring import Scorer
from src.core.explain import Explainer
from src.core.pdf_parser import extract_text_from_bytes
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

router = APIRouter()

DEFAULT_WEIGHTS = {"tfidf": 0.4, "sbert": 0.6}  # fixed; payload weights ignored per request

def get_managers(request: Request):
    return request.app.state.tfidf, request.app.state.sbert

def _perform_ranking(
    job_clean: str,
    resumes_processed: List[dict],
    required_skills: Optional[List[str]],
    tfidf_mgr: TFIDFManager,
    sbert_mgr: SBERTManager
) -> List[RankedResume]:
    """Helper function to perform the core ranking logic."""
    resume_texts = [r["cleaned"] for r in resumes_processed]

@router.post("/rank", response_model=RankResponse)
def rank_endpoint(payload: RankRequest, managers: tuple = Depends(get_managers)) -> RankResponse:
    """
    Accepts JSON RankRequest (job + list of resumes as text). Returns RankResponse.
    We ignore any weights passed and use DEFAULT_WEIGHTS.
    """
    tfidf_mgr, sbert_mgr = managers
    if tfidf_mgr is None:
        raise HTTPException(status_code=503, detail="TF-IDF model not available. Train and place at src/app/models/tfidf_vectorizer.pkl")
    if sbert_mgr is None:
        raise HTTPException(status_code=503, detail="SBERT model not available")

    # required fields
    if not payload.job or not payload.job.job_title or not payload.job.job_description:
        raise HTTPException(status_code=422, detail="job.job_title and job.job_description are required")

    # Build job text and clean
    job_fields = [payload.job.job_title, payload.job.job_description]
    if payload.job.required_skills:
        job_fields.append(" ".join(payload.job.required_skills) if isinstance(payload.job.required_skills, list) else str(payload.job.required_skills))
    if payload.job.keywords:
        job_fields.append(" ".join(payload.job.keywords) if isinstance(payload.job.keywords, list) else str(payload.job.keywords))
    if payload.job.experience_level:
        job_fields.append(str(payload.job.experience_level))

    job_raw = " ".join([f for f in job_fields if f])
    job_clean = clean_text_pipeline(job_raw)

    # prepare resumes
    resumes_processed = []
    for r in payload.resumes:
        raw = r.resume_text or ""
        cleaned = clean_text_pipeline(raw)
        resumes_processed.append({"resume_id": r.resume_id, "raw": raw, "cleaned": cleaned})

    if not resumes_processed:
        return RankResponse(job_id=payload.job.job_id or None, ranked_resumes=[])

    resume_texts = [r["cleaned"] for r in resumes_processed]

    # TF-IDF similarities
    tfidf_sims = tfidf_mgr.cosine_similarities(job_clean, resume_texts)

    # SBERT similarities
    job_emb = sbert_mgr.embed_texts([job_clean])[0]
    resume_embs = sbert_mgr.embed_texts(resume_texts)
    sbert_sims = cosine_similarity([job_emb], resume_embs)[0]

    # Skill matching
    job_skills = []
    if payload.job.required_skills:
        job_skills = payload.job.required_skills if isinstance(payload.job.required_skills, list) else [s.strip() for s in str(payload.job.required_skills).split(",") if s.strip()]

    skill_scores: List[Optional[float]] = []
    matched_list: List[List[str]] = []
    missing_list: List[List[str]] = []
    for txt in resume_texts:
        if job_skills:
            ex = extract_skills_from_text(txt, job_skills)
            matched, missing = ex["matched"], ex["missing"]
            matched_list.append(matched)
            missing_list.append(missing)
            skill_scores.append(len(matched) / max(len(job_skills), 1))
        else:
            matched_list.append([])
            missing_list.append([])
            skill_scores.append(None)

    # Combine scores (fixed weights)
    scorer = Scorer(DEFAULT_WEIGHTS)
    final_scores = [scorer.combine(float(tfidf_sims[i]), float(sbert_sims[i]), skill_scores[i]) for i in range(len(resume_texts))]

    # Rank and build response models
    order = np.argsort(final_scores)[::-1]
    explainer = Explainer(tfidf_mgr)
    ranked_models: List[RankedResume] = []
    for idx in order:
        top_terms = explainer.top_tfidf_terms(job_clean, resumes_processed[idx]["cleaned"], top_k=10)
        rr = RankedResume(
            resume_id=resumes_processed[idx]["resume_id"],
            final_score=float(final_scores[idx]),
            tfidf_score=float(tfidf_sims[idx]),
            sbert_score=float(sbert_sims[idx]),
            skill_score=(skill_scores[idx] if skill_scores[idx] is not None else None),
            matched_skills=matched_list[idx],
            missing_skills=missing_list[idx],
            top_tfidf_terms=top_terms
        )
        ranked_models.append(rr)

    return RankResponse(job_id=payload.job.job_id or None, ranked_resumes=ranked_models)


@router.post("/rank/upload", response_model=RankResponse)
async def rank_upload_endpoint(
    request: Request,
    job_title: str = Form(...),
    job_description: str = Form(...),
    required_skills: Optional[str] = Form(None),
    files: List[UploadFile] = File(...)
) -> RankResponse:
    """
    Accept multipart/form-data:
      - job_title (required)
      - job_description (required)
      - required_skills (optional): comma-separated or JSON list
      - files: one or more PDF files
    Returns same RankResponse as /rank.
    """
    tfidf_mgr, sbert_mgr = request.app.state.tfidf, request.app.state.sbert
    if tfidf_mgr is None:
        raise HTTPException(status_code=503, detail="TF-IDF model not available")
    if sbert_mgr is None:
        raise HTTPException(status_code=503, detail="SBERT model not available")

    # parse skills
    skills_parsed: Optional[List[str]] = None
    if required_skills:
        s = required_skills.strip()
        if s.startswith("["):
            try:
                import json
                skills_parsed = json.loads(s)
            except Exception:
                skills_parsed = [x.strip() for x in s.strip("[]").split(",") if x.strip()]
        else:
            skills_parsed = [x.strip() for x in s.split(",") if x.strip()]

    # extract texts from uploaded files
    resumes = []
    for f in files:
        content = await f.read()
        text = extract_text_from_bytes(content) if content else ""
        cleaned = clean_text_pipeline(text)
        resumes.append({"resume_id": f.filename, "raw": text, "cleaned": cleaned})

    if not resumes:
        return RankResponse(job_id=None, ranked_resumes=[])

    # Build job text and clean
    job_raw = " ".join([job_title, job_description] + (skills_parsed or []))
    job_clean = clean_text_pipeline(job_raw)

    resume_texts = [r["cleaned"] for r in resumes]

    tfidf_sims = tfidf_mgr.cosine_similarities(job_clean, resume_texts)
    job_emb = sbert_mgr.embed_texts([job_clean])[0]
    resume_embs = sbert_mgr.embed_texts(resume_texts)
    sbert_sims = cosine_similarity([job_emb], resume_embs)[0]

    matched_list: List[List[str]] = []
    missing_list: List[List[str]] = []
    skill_scores: List[Optional[float]] = []
    for txt in resume_texts:
        if skills_parsed:
            ex = extract_skills_from_text(txt, skills_parsed)
            matched_list.append(ex["matched"])
            missing_list.append(ex["missing"])
            skill_scores.append(len(ex["matched"]) / max(len(skills_parsed), 1))
        else:
            matched_list.append([])
            missing_list.append([])
            skill_scores.append(None)

    scorer = Scorer(DEFAULT_WEIGHTS)
    final_scores = [scorer.combine(float(tfidf_sims[i]), float(sbert_sims[i]), skill_scores[i]) for i in range(len(resume_texts))]

    order = np.argsort(final_scores)[::-1]
    explainer = Explainer(tfidf_mgr)
    ranked_models: List[RankedResume] = []
    for idx in order:
        top_terms = explainer.top_tfidf_terms(job_clean, resumes[idx]["cleaned"], top_k=10)
        rr = RankedResume(
            resume_id=resumes[idx]["resume_id"],
            final_score=float(final_scores[idx]),
            tfidf_score=float(tfidf_sims[idx]),
            sbert_score=float(sbert_sims[idx]),
            skill_score=(skill_scores[idx] if skill_scores[idx] is not None else None),
            matched_skills=matched_list[idx],
            missing_skills=missing_list[idx],
            top_tfidf_terms=top_terms
        )
        ranked_models.append(rr)

    return RankResponse(job_id=None, ranked_resumes=ranked_models)