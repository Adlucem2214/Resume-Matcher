from pathlib import Path
import pandas as pd
import uuid
from src.core.cleaning import clean_text_pipeline

ROOT = Path(__file__).resolve().parents[2]  # project root (resume-job-matcher)
RAW = ROOT / "src" / "data" / "raw" / "Resume_Dataset.csv"
OUT_DIR = ROOT / "src" / "data" / "resumes"
PROCESSED_DIR = ROOT / "src" / "data" / "processed"

OUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_raw_resume_df(path: Path) -> pd.DataFrame:
    # Try common CSV variants and fallback to pandas defaults
    df = pd.read_csv(path, dtype=str, encoding="utf-8", keep_default_na=False)
    # Expect column named 'Resume' (as in your file); fallback to first text-like column
    if "Resume" in df.columns:
        txt_col = "Resume"
    else:
        # pick the second column if first is 'Category'
        txt_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    return df[[txt_col]].rename(columns={txt_col: "resume_text"})

def prepare():
    if not RAW.exists():
        raise FileNotFoundError(f"Raw resume CSV not found at {RAW}")
    raw_df = load_raw_resume_df(RAW)
    # create stable ids
    raw_df = raw_df.reset_index(drop=True)
    raw_df["resume_id"] = raw_df.index.to_series().apply(lambda i: f"resume_{i+1}")
    # ensure text is str
    raw_df["resume_text"] = raw_df["resume_text"].astype(str)

    # run cleaning pipeline and keep both raw and cleaned
    raw_df["cleaned_resume"] = raw_df["resume_text"].apply(lambda t: clean_text_pipeline(t))

    # save standardized train.csv used by training/eval (columns: resume_id, resume_text)
    train_csv = OUT_DIR / "train_1.csv" 
    raw_df[["resume_id", "cleaned_resume"]].rename(columns={"cleaned_resume": "resume_text"}).to_csv(train_csv, index=False, encoding="utf-8")


    # save processed parquet with cleaned text for faster reuse
    proc_path = PROCESSED_DIR / "resumes.parquet"
    raw_df[["resume_id", "resume_text", "cleaned_resume"]].to_parquet(proc_path, index=False)
    print("Wrote processed resumes parquet ->", proc_path)

if __name__ == "__main__":
    prepare()