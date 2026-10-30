from pydantic import BaseModel
from typing import List, Optional, Any

class JobInput(BaseModel):
    job_id: Optional[str] = None
    job_title: str
    job_description: str
    required_skills: Optional[Any] = None  # string or list
    experience_level: Optional[str] = None
    keywords: Optional[Any] = None

class ResumeInput(BaseModel):
    resume_id: str
    resume_text: str

class RankRequest(BaseModel):
    job: JobInput
    resumes: List[ResumeInput]
    weights: Optional[dict] = None  # e.g. {"tfidf":0.4,"sbert":0.5,"skill":0.1}

class RankedResume(BaseModel):
    resume_id: str
    final_score: float
    tfidf_score: float
    sbert_score: float
    skill_score: Optional[float] = None
    matched_skills: List[str]
    missing_skills: List[str]
    top_tfidf_terms: List[Any]

class RankResponse(BaseModel):
    job_id: Optional[str] = None
    ranked_resumes: List[RankedResume]
    

# from pydantic import BaseModel
# from typing import List, Optional, Dict, Any

# class JobInput(BaseModel):
#     job_id: Optional[str] = None
#     job_title: str
#     job_description: str
#     required_skills: Optional[Any] = None  # string or list
#     experience_level: Optional[str] = None
#     keywords: Optional[Any] = None

# class ResumeInput(BaseModel):
#     resume_id: str
#     resume_text: str

# class RankRequest(BaseModel):
#     job: JobInput
#     resumes: List[ResumeInput]
#     weights: Optional[Dict[str, float]] = None  # e.g. {"tfidf":0.4,"sbert":0.5,"skill":0.1}

# class RankedResume(BaseModel):
#     resume_id: str
#     final_score: float
#     tfidf_score: float
#     sbert_score: float
#     skill_score: Optional[float] = None
#     matched_skills: List[str]
#     missing_skills: List[str]
#     top_tfidf_terms: List[Any]

# class RankResponse(BaseModel):
#     job_id: Optional[str] = None
#     ranked_resumes: List[Dict[str, Any]]