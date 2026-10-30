import re
from typing import List
import spacy
from spacy.cli import download as spacy_download
from pathlib import Path

# Try loading small model
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
except Exception:
    spacy_download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

STOPWORDS = nlp.Defaults.stop_words

def simple_preprocess(text: str) -> str:
    """
    Keep lemmatized tokens, allow alphanumeric tokens, drop stopwords,
    keep tokens longer than 2 chars (helps keep short meaningful tokens).
    """
    doc = nlp(text)
    toks = []
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        lemma = token.lemma_.strip().lower()
        if not lemma:
            continue
        if lemma in STOPWORDS:
            continue
        # allow alphanumeric tokens (letters + digits + +,#,-,.)
        if re.fullmatch(r"[a-z0-9\+\#\.-]+", lemma) and len(lemma) > 2:
            toks.append(lemma)
    return " ".join(toks)