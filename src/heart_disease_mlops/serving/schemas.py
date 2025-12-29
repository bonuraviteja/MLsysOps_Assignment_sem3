from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    age: float = Field(..., ge=0)
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: float
    thal: float


class PredictResponse(BaseModel):
    prediction: int
    confidence: float
