"""Testes da API FastAPI (Etapa 6).

Garante que exista um modelo treinado e valida os contratos de /health e
/predict via TestClient (que dispara o lifespan e carrega o modelo).
"""
import joblib
import pytest
from fastapi.testclient import TestClient

import app.main as api
from src.models.train import BEST_MODEL_PATH, train_all

VALID_PAYLOAD = {
    "tenure_months": 3, "monthly_charges": 95.0, "total_charges": 285.0,
    "contract": "Month-to-month", "internet_service": "Fiber optic",
    "payment_method": "Electronic check", "senior_citizen": 0, "partner": 0,
    "dependents": 0, "paperless_billing": 1, "tech_support": 0, "online_security": 0,
}


@pytest.fixture(scope="module")
def client():
    if not BEST_MODEL_PATH.exists():
        train_all(n=1500, seed=1)
    api.MODEL_PATH = BEST_MODEL_PATH
    api._model = joblib.load(BEST_MODEL_PATH)
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_probability(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn"], bool)


def test_predict_high_vs_low_risk(client):
    """Sanidade de negócio: mensal+fibra+novo deve churnar mais que 2 anos+antigo."""
    low = dict(
        VALID_PAYLOAD, tenure_months=60, contract="Two year",
        internet_service="DSL", payment_method="Credit card",
        monthly_charges=45.0, total_charges=2700.0, tech_support=1, online_security=1,
    )
    high_p = client.post("/predict", json=VALID_PAYLOAD).json()["churn_probability"]
    low_p = client.post("/predict", json=low).json()["churn_probability"]
    assert high_p > low_p


def test_predict_validation_error(client):
    bad = dict(VALID_PAYLOAD, tenure_months=999)  # fora de [0, 72]
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422
