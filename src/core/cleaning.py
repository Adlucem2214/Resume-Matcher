# ...existing code...
import re
import unicodedata
from typing import Optional

# try to import project's tokenizer/lemmatizer, fallback handled later
try:
    from src.core.preprocessing import simple_preprocess
except Exception:
    simple_preprocess = None

# Common mojibake and unicode fixes
_MOJIBAKE_MAP = {
    "Ã¯Â¿Â½": " ",
    "Ã¢ÂÂ¢": " ",
    "Ã¢ÂÂ": "-",
    "â¢": " ",
    "â": "-",
    "â": "-",
    "Â": "",
    "Ã": "A",
    "â": " ",
    "\uFFFD": " ",
    "âª": " ",
    "â": " ",
    "â": " ",
    "Ã¼": "u",
    "Ã©": "e",
    "Ã ": "a",
}

# Patterns for PII and noise
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b")
_LONG_NUM_RE = re.compile(r"\b\d{6,}\b")

# Remove common section headers and labels that repeat across resumes
_SECTION_HEADERS = [
    r"\b(skill(s)?|skills?)\b\s*[:\-]?", r"\b(education|education details)\b",
    r"\b(company|company details|company -)\b", r"\b(description)\b\s*[:\-]?",
    r"\b(profile summary|profile)\b", r"\b(achievement(s)?|achievements)\b",
    r"\b(experience|work experience)\b", r"\b(projects?)\b", r"\b(personal profile)\b",
    r"\b(contact|email|e-?mail|phone)\b\s*[:\-]?", r"\b(skill details)\b",
    r"\b(multiple data science and analytic projects)\b", r"\bcurriculum vitae\b",
    r"\bdata science\b"  # careful: kept broad to remove standalone header lines
]
_SECTION_RE = re.compile("|".join(_SECTION_HEADERS), flags=re.IGNORECASE)

# Remove decorative bullets / symbols
_SYMBOLS_RE = re.compile(
    r"[\u2022\u2023\u25E6\u2043\u2219\u25AA\u25AB\u25CF\u25CB"
    r"\u2500-\u257F\uFFFD\*•·►✔✓■▲▼◦]"
)

# Normalize whitespace and punctuation
_PUNCT_RE = re.compile(r"[=+_/\\@#$%^&*<>~`{}|\[\];:\"'“”‘’—––]"
)

def _normalize_unicode(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for k, v in _MOJIBAKE_MAP.items():
        text = text.replace(k, v)
    # Replace weird control sequences and NBSP with space
    text = text.replace("\xa0", " ")
    text = text.replace("\ufeff", " ")
    return text

def _remove_pii_and_urls(text: str) -> str:
    t = _EMAIL_RE.sub(" ", text)
    t = _URL_RE.sub(" ", t)
    t = _PHONE_RE.sub(" ", t)
    t = _LONG_NUM_RE.sub(" ", t)
    return t

def _strip_bracketed(text: str) -> str:
    # remove content inside parentheses/brackets that is often metadata
    return re.sub(r"\[.*?\]|\(.*?\)|\<.*?\>", " ", text)

def _collapse_repeated_punct(text: str) -> str:
    # replace runs of punctuation/ornaments with a single space
    text = re.sub(r"[^\w\s]{2,}", " ", text)
    return text

def _reduce_repeated_letters(text: str, max_repeat: int = 2) -> str:
    # reduce sequences like coooool -> coool (keeping up to max_repeat)
    return re.sub(r"(.)\1{" + str(max_repeat) + r",}", r"\1" * max_repeat, text)

def _clean_basic(text: str) -> str:
    t = text
    t = _normalize_unicode(t)
    t = t.replace("\r", " ").replace("\n", " ")
    t = _strip_bracketed(t)
    t = _remove_pii_and_urls(t)
    t = _SECTION_RE.sub(" ", t)
    t = _SYMBOLS_RE.sub(" ", t)
    t = t.replace("*", " ")  # remove asterisks used as bullets
    t = _PUNCT_RE.sub(" ", t)
    t = _collapse_repeated_punct(t)
    t = _reduce_repeated_letters(t, max_repeat=2)
    # collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    # Remove long underline separators like ________
    t = re.sub(r"_{2,}", " ", t)

    return t

def clean_text_pipeline(text: Optional[str]) -> str:
    """
    Clean a single resume text and return a lowercased, whitespace-normalized string.
    Keeps words and simple tokens; removes PII, bullets, headers, weird encodings and long numeric noise.
    """
    if text is None:
        return ""
    t = str(text)
    t = _clean_basic(t)
    # Lowercase after basic cleaning
    t = t.lower().strip()
    # Hand off to simple_preprocess if available (may return str or list)
    if simple_preprocess:
        try:
            proc = simple_preprocess(t)
            if isinstance(proc, list):
                t = " ".join(proc)
            else:
                t = str(proc).strip()
        except Exception:
            # fallback to simple tokenization
            t = " ".join(re.findall(r"\b[a-zA-Z0-9]+\b", t))
    else:
        # fallback tokenization: keep alnum words
        t = " ".join(re.findall(r"\b[a-zA-Z0-9]+\b", t))
    # final collapse
    t = re.sub(r"\s+", " ", t).strip()
    return t



# # ...existing code...
# import re
# import unicodedata
# from src.core.preprocessing import simple_preprocess

# # Steps must be applied in order described.

# # Step 1: Unicode & Encoding Normalization
# def normalize_unicode(text: str) -> str:
#     if not text:
#         return ""
#     # Normalize to NFKC for compatibility (NFKD then recombine)
#     text = unicodedata.normalize("NFKC", text)
#     # common mojibake / weird sequences -> remove or map to ascii-friendly
#     fixes = {
#         "Ã¯Â¿Â½": "�",
#         "Ã¢ÂÂ¢": " ",
#         "Ã¢ÂÂ": "-",
#         "â¢": " ",
#         "â": "-",
#         "â": "-",
#         "Â": "",
#         "Ã": "A",
#         "â": "a",
#         "�": " "
#     }
#     for k, v in fixes.items():
#         text = text.replace(k, v)
#     return text

# # helper: remove html tags
# def remove_html_tags(text: str) -> str:
#     return re.sub(r"<[^>]+>", " ", text)

# # helper: remove emails, urls, phones
# def remove_pii_like(text: str) -> str:
#     txt = text
#     # emails
#     txt = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", " ", txt)
#     # urls
#     txt = re.sub(r"https?://\S+|www\.\S+", " ", txt)
#     # phone numbers (various formats)
#     txt = re.sub(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b", " ", txt)
#     return txt

# # Step 2: Noise Removal (declarations, signatures, dates, page numbers, generic statements)
# NOISE_PATTERNS = [
#     r"i hereby declare[^\n]*", r"signature[^\n]*",
#     r"date\s*[:\-]\s*\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}",
#     r"page\s*\d+\s*of\s*\d+", r"curriculum vitae", r"\bresume\b", r"contact\s*[:\-]",
#     r"linkedin\.com[^\s,]*", r"github\.com[^\s,]*"
# ]

# def remove_noise(text: str) -> str:
#     txt = text
#     # strip HTML, PII-like and bracketed metadata first
#     txt = remove_html_tags(txt)
#     txt = re.sub(r"\[.*?\]|\(.*?\)|\<.*?\>", " ", txt)
#     txt = remove_pii_like(txt)
#     # apply noise patterns
#     for p in NOISE_PATTERNS:
#         txt = re.sub(p, " ", txt, flags=re.IGNORECASE)
#     # remove long numeric sequences (serial numbers, some dates)
#     txt = re.sub(r"\b\d{6,}\b", " ", txt)
#     return txt

# # Step 3: Symbol & Bullet Cleanup
# def remove_symbols_and_bullets(text: str) -> str:
#     txt = text
#     # replace common bullets and arrows with space
#     bullets = ["•", "â¢", "Â•", "·", "•", "â€”", "–", "—", "•", "•"]
#     for b in bullets:
#         txt = txt.replace(b, " ")
#     # remove other odd control characters
#     txt = re.sub(r"[\u2022\u2023\u25E6\u2043\u2219\uFFFD]", " ", txt)
#     # remove repeated punctuation like "----", "....", "***"
#     txt = re.sub(r"([^\w\s])\1{1,}", r"\1", txt)
#     # remove stray non-printable/control chars
#     txt = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", txt)
#     # replace many punctuation with space but keep simple sentence punctuation to preserve tokens
#     txt = re.sub(r"[=+_/\\@#$%^&*<>~`{}|\[\];:\"']", " ", txt)
#     # normalize whitespace
#     txt = re.sub(r"\s+", " ", txt)
#     return txt.strip()

# # helper: reduce repeated letters (cooool -> cool)
# def reduce_repeated_chars(text: str, max_repeat: int = 2) -> str:
#     return re.sub(r"(.)\1{" + str(max_repeat) + r",}", r"\1" * max_repeat, text)

# # Step 4 & 5 done in API & pipeline (job construction & lowercasing)
# # Step 6: linguistic preprocessing delegated to preprocessing.simple_preprocess

# def clean_text_pipeline(text: str) -> str:
#     if text is None:
#         return ""
#     t = str(text)
#     t = normalize_unicode(t)
#     t = remove_noise(t)
#     t = remove_symbols_and_bullets(t)
#     t = t.lower()
#     t = reduce_repeated_chars(t, max_repeat=2)
#     # delegate tokenization / lemmatization / stopword removal to simple_preprocess
#     try:
#         t = simple_preprocess(t)
#     except Exception:
#         # fallback: basic tokenization and join
#         t = " ".join(re.findall(r"\b\w+\b", t))
#     return t.strip()




# import re
# import unicodedata
# from src.core.preprocessing import simple_preprocess

# # Steps must be applied in order described.

# # Step 1: Unicode & Encoding Normalization
# def normalize_unicode(text: str) -> str:
#     if not text:
#         return ""
#     # normalize to NFKD and remove weird control chars
#     text = unicodedata.normalize("NFKD", text)
#     # remove common corrupted sequences
#     text = text.replace("Ã¯Â¿Â½", "").replace("Ã¢ÂÂ¢", " ").replace("Â", "")
#     return text

# # Step 2: Noise Removal (declarations, signatures, dates, page numbers, generic statements)
# NOISE_PATTERNS = [
#     r"i hereby declare[^\n]*", r"signature[^\n]*", r"date\s*[:\-]\s*\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}",
#     r"page\s*\d+\s*of\s*\d+", r"curriculum vitae", r"resume", r"contact\s*[:\-]"
# ]

# def remove_noise(text: str) -> str:
#     txt = text
#     for p in NOISE_PATTERNS:
#         txt = re.sub(p, " ", txt, flags=re.IGNORECASE)
#     # remove long sequences of non-letters except punctuation kept for next steps
#     return txt

# # Step 3: Symbol & Bullet Cleanup
# def remove_symbols_and_bullets(text: str) -> str:
#     # common bullets and weird symbols
#     bullets = ["•", "â¢", "Â•", "·", "-", "•", "*", "â€”"]
#     txt = text
#     for b in bullets:
#         txt = txt.replace(b, " ")
#     # remove other odd control characters
#     txt = re.sub(r"[•·•\u2022\u2023\u25E6\u2043\u2219]", " ", txt)
#     # normalize whitespace
#     txt = re.sub(r"\s+", " ", txt)
#     return txt.strip()

# # Step 4 & 5 done in API & pipeline (job construction & lowercasing)
# # Step 6: linguistic preprocessing delegated to preprocessing.simple_preprocess

# def clean_text_pipeline(text: str) -> str:
#     t = normalize_unicode(text)
#     t = remove_noise(t)
#     t = remove_symbols_and_bullets(t)
#     t = t.lower()
#     t = simple_preprocess(t)
#     return t