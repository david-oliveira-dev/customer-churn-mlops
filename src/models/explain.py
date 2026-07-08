"""Interpretabilidade do modelo de churn com SHAP.

Gera um gráfico de importância global (summary plot) em ``reports/`` a partir do
modelo salvo, ajudando a explicar *por que* o modelo prevê churn — essencial
para um projeto de portfólio orientado a negócio.

Uso:
    python -m src.models.explain --n 2000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # backend sem display (roda em servidor/CI)
import matplotlib.pyplot as plt
import shap

from src.data.generate_synthetic import generate_customers
from src.features.build_features import split_X_y
from src.models.train import BEST_MODEL_PATH, REPORTS_DIR

SHAP_PLOT_PATH = REPORTS_DIR / "shap_summary.png"


def explain(n: int = 2000, seed: int = 123, model_path: Path = BEST_MODEL_PATH) -> Path:
    """Calcula valores SHAP numa amostra e salva o summary plot."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} não existe. Rode antes: python -m src.models.train"
        )

    pipeline = joblib.load(model_path)
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    df = generate_customers(n=n, seed=seed)
    X, _ = split_X_y(df)
    X_trans = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()

    # TreeExplainer cobre RF/XGB/LGBM — todos os candidatos são baseados em árvore.
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_trans)
    # Alguns explainers retornam lista [classe0, classe1]; usamos a classe churn.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X_trans, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_PLOT_PATH, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary salvo em {SHAP_PLOT_PATH}")
    return SHAP_PLOT_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicabilidade SHAP do churn.")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    explain(n=args.n, seed=args.seed)


if __name__ == "__main__":
    main()
