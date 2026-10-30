from pathlib import Path
import numpy as np
import pandas as pd
from src.core.sbert import SBERTManager

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "src" / "data" / "processed" / "resumes.parquet"
OUT_DIR = ROOT / "src" / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main(batch_size: int = 64, model_name: str = "all-MiniLM-L6-v2"):
    df = pd.read_parquet(PROC)
    ids = df["resume_id"].astype(str).tolist()
    texts = df["cleaned_resume"].astype(str).tolist()
    mgr = SBERTManager(model_name=model_name)
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = mgr.embed_texts(batch)
        embeddings.append(emb)
        print(f"Computed embeddings {i}..{i+len(batch)}")
    emb_mat = np.vstack(embeddings)
    np.save(OUT_DIR / "resume_embeddings.npy", emb_mat)
    pd.DataFrame({"resume_id": ids}).to_csv(OUT_DIR / "resume_ids.csv", index=False)
    print("Saved embeddings ->", OUT_DIR / "resume_embeddings.npy")
    print("Saved ids ->", OUT_DIR / "resume_ids.csv")

if __name__ == "__main__":
    main()