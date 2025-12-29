from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from prometheus_fastapi_instrumentator import Instrumentator

from heart_disease_mlops.config import FEATURE_COLUMNS
from heart_disease_mlops.logging_utils import configure_logging
from heart_disease_mlops.serving.schemas import PredictRequest, PredictResponse

logger = logging.getLogger("heart_disease_mlops.api")


@lru_cache(maxsize=1)
def _load_model():
    model_path = Path(os.environ.get("MODEL_PATH", "artifacts/model/model.joblib")).resolve()
    if not model_path.exists():
        raise FileNotFoundError(
            f"MODEL_PATH not found: {model_path}. Run the training pipeline first to generate it."
        )
    return joblib.load(model_path)


app = FastAPI(title="Heart Disease Risk API", version="0.1.0")
Instrumentator().instrument(app).expose(app)


@app.on_event("startup")
def _startup() -> None:
    configure_logging()
    _load_model()
    logger.info("Model loaded")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    dur_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s status=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        dur_ms,
    )
    return response


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    _load_model()
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    model = _load_model()
    row = payload.model_dump()
    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    proba = float(model.predict_proba(df)[:, 1][0])
    pred = int(proba >= 0.5)
    return PredictResponse(prediction=pred, confidence=proba)
