"""
PDF text extraction helpers. Primary: pdfplumber. Fallback: PyPDF2 if available.
"""
from pathlib import Path
from typing import Optional
import io

def _try_import_pdfplumber():
    try:
        import pdfplumber  # type: ignore
        return pdfplumber
    except Exception:
        return None

def extract_text_from_pdf_path(path: Path) -> str:
    """
    Extract plain text from a PDF file path. Returns empty string on failure.
    """
    pdfplumber = _try_import_pdfplumber()
    if pdfplumber:
        try:
            with pdfplumber.open(str(path)) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            return "\n".join(pages).strip()
        except Exception:
            pass
    # fallback to PyPDF2
    try:
        from PyPDF2 import PdfReader  # type: ignore
        reader = PdfReader(str(path))
        texts = []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                texts.append("")
        return "\n".join(texts).strip()
    except Exception:
        return ""

def extract_text_from_bytes(b: bytes) -> str:
    """
    Extract text from PDF bytes (useful for uploaded files).
    """
    pdfplumber = _try_import_pdfplumber()
    stream = io.BytesIO(b)
    if pdfplumber:
        try:
            with pdfplumber.open(stream) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            return "\n".join(pages).strip()
        except Exception:
            pass
    # fallback PyPDF2
    try:
        from PyPDF2 import PdfReader  # type: ignore
        reader = PdfReader(stream)
        texts = []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                texts.append("")
        return "\n".join(texts).strip()
    except Exception:
        return ""