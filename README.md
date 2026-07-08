# Customer Churn MLOps Platform

Plataforma ponta a ponta para prever **churn** (cancelamento de clientes) —
de dados sintéticos a modelo servido em API, com dashboard e monitoramento.
Projeto de portfólio para vaga de **Cientista de Dados Pleno**.

> 🚧 **Em construção incremental.** O roteiro completo está em
> [`BUILD_BRIEF.md`](BUILD_BRIEF.md). Etapa atual: **1 — geração de dados**.

## Stack
Python 3.12 · Pandas/NumPy · scikit-learn · XGBoost · LightGBM · SHAP ·
MLflow · FastAPI · Streamlit · SQLAlchemy/PostgreSQL · Docker · pytest

## Arquitetura (alvo)
```
Dados sintéticos → ETL → PostgreSQL → Feature Engineering
      → Treino (RF / XGBoost / LightGBM) + SHAP → MLflow
      → API (FastAPI) → Dashboard (Streamlit) → Monitoramento
```

## Como rodar (local)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Etapa 1 — gerar dados sintéticos
python -m src.data.generate_synthetic --n 8000 --seed 42

# Testes
pytest -q
```

## Estrutura
```
data/        dados brutos e processados
src/         código (data, features, models)
notebooks/   EDA e experimentos
app/         API (FastAPI) e dashboard (Streamlit)
models/      modelos treinados
tests/       testes automatizados
reports/     relatórios, diagramas, SHAP
```

## Status das etapas
- [x] 1 — Geração de dados sintéticos
- [ ] 2 — ETL + carga no banco
- [ ] 3 — EDA
- [ ] 4 — Feature Engineering
- [ ] 5 — Treino e comparação de modelos
- [ ] 6 — API FastAPI
- [ ] 7 — Dashboard Streamlit
- [ ] 8 — Docker + Compose
- [ ] 9 — Testes, CI e documentação
