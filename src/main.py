from fastapi import FastAPI
from contextlib import asynccontextmanager
from pathlib import Path
import logging
from typing import Optional

# local imports (src package must be on PYTHONPATH when running)
from src.api import rank
from src.core.tfidf import TFIDFManager
from src.core.sbert import SBERTManager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("resume-job-matcher")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan manager: loads TF-IDF and SBERT managers once at startup and attaches
    them to app.state. Replaces deprecated @app.on_event("startup").
    """
    models_dir = PROJECT_ROOT / "src" / "app"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load TF-IDF manager
    try:
        app.state.tfidf = TFIDFManager(models_dir / "tfidf_vectorizer.pkl")
        logger.info("Loaded TF-IDF vectorizer at %s", str(models_dir / "tfidf_vectorizer.pkl"))
    except Exception as e:
        app.state.tfidf = None
        logger.exception("Failed to initialize TFIDFManager: %s", e)

    # Load SBERT manager
    try:
        app.state.sbert = SBERTManager()
        logger.info("Loaded SBERT model")
    except Exception as e:
        app.state.sbert = None
        logger.exception("Failed to initialize SBERTManager: %s", e)

    yield

    # Optionally add cleanup here
    logger.info("Application shutdown complete")


app = FastAPI(title="Resume-Job Matcher", lifespan=lifespan)
app.include_router(rank.router, prefix="/api")