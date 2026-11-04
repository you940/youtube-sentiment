import logging
import re
import time
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import BatchRequest, BatchResponse, Prediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentiment-api")

LABELS = {-1: "négatif", 0: "neutre", 1: "positif"}
state = {}


def clean_text(t: str) -> str:
    # Même nettoyage que dans preprocess.py (indispensable)
    t = str(t).lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"[^a-z0-9\s'!?]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Le modèle est chargé UNE SEULE FOIS au démarrage
    state["model"] = joblib.load("models/model.joblib")
    state["vec"] = joblib.load("models/vectorizer.joblib")
    logger.info("Modèle et vectoriseur chargés")
    yield
    state.clear()


app = FastAPI(
    title="YouTube Sentiment API",
    description="Analyse le sentiment de commentaires YouTube",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS : autorise l'extension Chrome à appeler l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    t0 = time.perf_counter()
    try:
        texts = [clean_text(c) for c in req.comments]
        X = state["vec"].transform(texts)
        probs = state["model"].predict_proba(X)
        classes = state["model"].classes_
    except Exception as e:
        logger.exception("Erreur de prédiction")
        raise HTTPException(status_code=500, detail=str(e))

    predictions = []
    counts = {-1: 0, 0: 0, 1: 0}
    for original, p in zip(req.comments, probs):
        i = int(p.argmax())
        s = int(classes[i])
        counts[s] += 1
        predictions.append(Prediction(
            text=original,
            sentiment=s,
            label=LABELS[s],
            confidence=round(float(p[i]), 4),
        ))

    n = len(predictions)
    stats = {
        "total": n,
        "counts": {LABELS[k]: v for k, v in counts.items()},
        "percentages": {LABELS[k]: round(100 * v / n, 1) for k, v in counts.items()},
    }
    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    logger.info("Batch de %d commentaires traité en %s ms", n, elapsed)

    return BatchResponse(predictions=predictions, statistics=stats,
                         processing_time_ms=elapsed)