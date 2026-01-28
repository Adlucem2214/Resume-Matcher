"""
Train TF-IDF vectorizer from resume training data and persist to models directory.

Usage:
    python scripts/train_tfidf.py
"""
from pathlib import Path
import sys
import pandas as pd
from typing import List

# make src importable when running from project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from src.core.tfidf import TFIDFManager
from src.core.cleaning import clean_text_pipeline

def load_resume_texts(train_csv: Path) -> List[str]:
    df = pd.read_csv(train_csv)
    if "resume_text" not in df.columns:
        raise RuntimeError(f"train csv at {train_csv} missing 'resume_text' column")
    return df["resume_text"].astype(str).tolist()

def main() -> None:
    data_dir = ROOT / "data" / "resumes"
    train_csv = data_dir / "train_1.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"{train_csv} not found. Place resume train CSV at this path.")

    print("Loading resume training data from:", train_csv)
    texts = load_resume_texts(train_csv)
    print(f"Loaded {len(texts)} resumes. Running cleaning pipeline...")

    cleaned = [clean_text_pipeline(t) for t in texts]
    print("Cleaning complete. Training TF-IDF vectorizer (unigrams + bigrams)...")

    models_dir = ROOT / "src" / "app" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "tfidf_vectorizer.pkl"

    tfidf_mgr = TFIDFManager(model_path)
    tfidf_mgr.train_from_corpus(cleaned)
    print("TF-IDF training complete. Vectorizer saved to:", model_path)

if __name__ == "__main__":
    main()