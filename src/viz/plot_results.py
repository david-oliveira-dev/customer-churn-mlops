"""Gera as figuras de resultado usadas no README.

Salva em `reports/figures/comparacao_modelos.png` o placar dos três modelos a
partir de `reports/metrics.json` (não retreina).

A figura destaca o ponto do projeto: o modelo **não** foi escolhido pelo F1 nem
pela acurácia, e sim pelo **recall na classe churn** — em retenção, o falso
negativo é o erro caro. O gráfico deixa esse critério visível, porque no F1 o
vencedor seria outro.

Uso:
    python -m src.viz.plot_results
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # backend sem display: roda em servidor e no CI
import matplotlib.pyplot as plt  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("plot_results")

ROOT = Path(__file__).resolve().parents[2]
FIGURAS = ROOT / "reports" / "figures"
METRICS = ROOT / "reports" / "metrics.json"

ROTULOS = {
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "f1": "F1",
    "recall_churn": "Recall (churn)",
}


def plot_comparacao_modelos() -> Path:
    dados = json.loads(METRICS.read_text())
    resultados = dados["all"]
    campeao = dados["best_model"]
    modelos = list(resultados)

    x = np.arange(len(ROTULOS))
    largura = 0.8 / len(modelos)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, nome in enumerate(modelos):
        valores = [resultados[nome][k] for k in ROTULOS]
        destaque = nome == campeao
        # Sem emoji no rótulo: a fonte padrão do matplotlib não tem o glifo e
        # renderiza um quadrado vazio.
        barras = ax.bar(x + i * largura, valores, largura,
                        label=f"{nome} (escolhido)" if destaque else nome,
                        color="#c44e52" if destaque else None,
                        alpha=1.0 if destaque else 0.75)
        for barra, valor in zip(barras, valores):
            ax.text(barra.get_x() + barra.get_width() / 2, valor,
                    f"{valor:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + largura * (len(modelos) - 1) / 2, list(ROTULOS.values()))
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("valor da métrica")
    ax.set_title("Comparação dos modelos — o critério de escolha é o recall na classe churn,\n"
                 "não o F1: em retenção, o falso negativo é o erro caro")
    ax.legend()
    fig.tight_layout()

    FIGURAS.mkdir(parents=True, exist_ok=True)
    destino = FIGURAS / "comparacao_modelos.png"
    fig.savefig(destino, dpi=120)
    plt.close(fig)
    logger.info("Salvo %s", destino.name)
    return destino


if __name__ == "__main__":
    plot_comparacao_modelos()
