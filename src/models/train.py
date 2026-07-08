"""Treino e comparação de modelos de churn com tracking em MLflow.

Compara RandomForest, XGBoost e LightGBM dentro de um Pipeline
(preprocessor + classificador), avalia em teste holdout (ROC-AUC, PR-AUC, F1 e
recall na classe churn — a métrica de negócio mais importante, pois errar um
cliente que vai cancelar é caro) e salva o melhor modelo em ``models/``.

Uso:
    python -m src.models.train --n 8000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, f1_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.generate_synthetic import generate_customers
from src.features.build_features import build_preprocessor, split_X_y

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MLRUNS_DIR = ROOT / "mlruns"
# MLflow 3.x aposentou o file store; usamos backend SQLite (recomendado).
# Sobrescrevível por env var para apontar a um servidor MLflow em produção.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
BEST_MODEL_PATH = MODELS_DIR / "churn_model.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"

EXPERIMENT = "churn-classification"


def get_models() -> dict[str, object]:
    """Instancia os classificadores a comparar. Importa XGB/LGBM aqui para
    manter o módulo importável mesmo se algum extra não estiver instalado."""
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, class_weight="balanced",
            random_state=42, n_jobs=-1, verbose=-1,
        ),
    }


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Calcula as métricas de avaliação no conjunto de teste."""
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "f1": float(f1_score(y_test, pred)),
        "recall_churn": float(recall_score(y_test, pred)),
    }


def train_all(n: int = 8000, seed: int = 42) -> dict:
    """Treina e compara todos os modelos; salva o melhor (por ROC-AUC)."""
    df = generate_customers(n=n, seed=seed)
    X, y = split_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed,
    )

    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    results: dict[str, dict[str, float]] = {}
    best_name, best_pipeline, best_auc = None, None, -1.0

    for name, clf in get_models().items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", clf),
        ])
        with mlflow.start_run(run_name=name):
            pipeline.fit(X_train, y_train)
            metrics = evaluate(pipeline, X_test, y_test)
            mlflow.log_param("model", name)
            mlflow.log_metrics(metrics)
            results[name] = metrics
            print(f"{name:>14}: " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
            if metrics["roc_auc"] > best_auc:
                best_name, best_pipeline, best_auc = name, pipeline, metrics["roc_auc"]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, BEST_MODEL_PATH)

    summary = {"best_model": best_name, "best_roc_auc": best_auc, "all": results}
    METRICS_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nMelhor modelo: {best_name} (ROC-AUC={best_auc:.3f}) -> {BEST_MODEL_PATH}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina modelos de churn.")
    parser.add_argument("--n", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_all(n=args.n, seed=args.seed)


if __name__ == "__main__":
    main()
