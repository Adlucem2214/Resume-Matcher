"""
Robust PDF parser with OCR fallback.
Requires: pip install pdfplumber pdf2image pytesseract pillow
System Requires: Poppler (for pdf2image) and Tesseract-OCR installed and in PATH.
"""
import sys
import logging
import pdfplumber
from pathlib import Path

# Configure logging to stderr so it doesn't break JSON output on stdout
logger = logging.getLogger("pdf_parser")
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter('[PDF-PARSER] %(levelname)s: %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

OCR_AVAILABLE = False
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    logger.warning("OCR dependencies (pytesseract, pdf2image, pillow) not installed. Scanned PDFs will be empty.")

def extract_text_from_pdf_path(path: Path) -> str:
    path = Path(path)
    if not path.exists():
        logger.error(f"File not found: {path}")
        return ""

    text = ""
    
    # 1. Try extracting text directly (pdfplumber) - fast and accurate for digital PDFs
    try:
        with pdfplumber.open(path) as pdf:
            pages = []
            for p in pdf.pages:
                extracted = p.extract_text()
                if extracted:
                    pages.append(extracted)
            text = "\n".join(pages)
    except Exception as e:
        logger.warning(f"pdfplumber error on {path.name}: {e}")

    # If we got a good amount of text, return it
    if text and len(text.strip()) > 50:
        return text

    # 2. Fallback to OCR if text is empty or too short
    if not OCR_AVAILABLE:
        logger.warning(f"Skipping OCR for {path.name} (missing libraries). Text length: {len(text)}")
        return text

    logger.info(f"Text layer empty/insufficient for {path.name}. Attempting OCR...")
    
    try:
        # Convert PDF to images
        # Note: This requires Poppler installed and in PATH
        images = convert_from_path(str(path))
        
        ocr_texts = []
        for i, img in enumerate(images):
            # pytesseract needs tesseract executable in PATH
            page_text = pytesseract.image_to_string(img)
            ocr_texts.append(page_text)
        
        ocr_full = "\n".join(ocr_texts)
        
        if ocr_full.strip():
            logger.info(f"OCR successful for {path.name} ({len(ocr_full)} chars).")
            return ocr_full
        else:
            logger.warning(f"OCR produced no text for {path.name}.")
            
    except Exception as e:
        logger.error(f"OCR failed for {path.name}. Error: {e}")
        if "poppler" in str(e).lower():
            logger.error("Ensure Poppler is installed and in PATH.")
        if "tesseract" in str(e).lower():
            logger.error("Ensure Tesseract-OCR is installed and in PATH.")

    return text
