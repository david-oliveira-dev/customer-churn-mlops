"""API FastAPI para servir o modelo de churn.

Endpoints:
    GET  /health   -> status do serviço e se o modelo está carregado
    POST /predict  -> recebe features de um cliente e devolve prob. + classe

O modelo (Pipeline preprocessor + classificador) é carregado uma vez na
inicialização. O caminho pode ser sobrescrito pela env var ``MODEL_PATH``.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.build_features import prepare_features

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/churn_model.joblib"))
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))

# Carregado no startup; None até lá (ou se o arquivo não existir).
_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = load_model()
    yield


app = FastAPI(
    title="Customer Churn API",
    description="Prevê a probabilidade de cancelamento (churn) de um cliente.",
    version="1.0.0",
    lifespan=lifespan,
)


class Customer(BaseModel):
    """Features de entrada de um cliente (dados brutos, sem derivadas)."""

    tenure_months: int = Field(..., ge=0, le=72, examples=[5])
    monthly_charges: float = Field(..., ge=0, examples=[89.9])
    total_charges: float = Field(..., ge=0, examples=[450.0])
    contract: str = Field(..., examples=["Month-to-month"])
    internet_service: str = Field(..., examples=["Fiber optic"])
    payment_method: str = Field(..., examples=["Electronic check"])
    senior_citizen: int = Field(..., ge=0, le=1, examples=[0])
    partner: int = Field(..., ge=0, le=1, examples=[0])
    dependents: int = Field(..., ge=0, le=1, examples=[0])
    paperless_billing: int = Field(..., ge=0, le=1, examples=[1])
    tech_support: int = Field(..., ge=0, le=1, examples=[0])
    online_security: int = Field(..., ge=0, le=1, examples=[0])


class Prediction(BaseModel):
    churn_probability: float
    churn: bool
    threshold: float


def load_model(path: Path = MODEL_PATH):
    """Carrega o pipeline salvo, se existir."""
    if path.exists():
        return joblib.load(path)
    return None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> Prediction:
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Rode `python -m src.models.train` primeiro.",
        )
    X = prepare_features(pd.DataFrame([customer.model_dump()]))
    proba = float(_model.predict_proba(X)[0, 1])
    return Prediction(
        churn_probability=round(proba, 4),
        churn=proba >= DECISION_THRESHOLD,
        threshold=DECISION_THRESHOLD,
    )
