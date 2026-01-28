import re
from typing import List, Dict
from difflib import SequenceMatcher

def _normalize_skill(s: str) -> str:
    return re.sub(r"[^a-z0-9\s\+\#\.-]", "", s.lower()).strip()

def extract_skills_from_text(text: str, job_skills: List[str]) -> Dict:
    """
    Returns matched and missing lists.
    Uses simple substring and fuzzy matching fallback.
    """
    normalized_text = text.lower()
    job_skills_norm = [_normalize_skill(s) for s in job_skills]
    matched = []
    missing = []
    for orig, norm in zip(job_skills, job_skills_norm):
        if not norm:
            continue
        # exact token boundary match
        pattern = r"\b" + re.escape(norm) + r"\b"
        if re.search(pattern, normalized_text):
            matched.append(orig)
            continue
        # substring match
        if norm in normalized_text:
            matched.append(orig)
            continue
        # fuzzy: ratio > 0.85
        found = False
        for token in set(re.findall(r"[a-zA-Z0-9\+\#\.-]+", normalized_text)):
            ratio = SequenceMatcher(None, token, norm).ratio()
            if ratio > 0.85:
                matched.append(orig)
                found = True
                break
        if not found:
            missing.append(orig)
    return {"matched": matched, "missing": missing}