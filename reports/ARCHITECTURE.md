# Arquitetura & Relatório Técnico — Customer Churn MLOps

## 1. Visão geral
Plataforma de previsão de churn projetada como um fluxo reprodutível de dados →
modelo → serviço. Cada etapa é um módulo isolado e testável.

```
┌──────────────────┐     ┌──────────┐     ┌────────────────────────┐
│ generate_synthetic│────▶│   ETL    │────▶│ PostgreSQL + parquet   │
│  (modelo logístico│     │ (clean/  │     │ (fallback SQLite)      │
│   latente)        │     │  valida) │     └────────────────────────┘
└──────────────────┘     └──────────┘                 │
                                                       ▼
┌───────────────────────────┐     ┌──────────────────────────────────┐
│  Feature Engineering       │────▶│  Treino: RF / XGBoost / LightGBM │
│  charges_per_tenure,       │     │  ColumnTransformer + Pipeline    │
│  buckets de tenure         │     │  → MLflow (métricas) + SHAP      │
└───────────────────────────┘     └──────────────────────────────────┘
                                                       │
                                       melhor modelo (churn_model.joblib)
                                                       │
                        ┌──────────────────────────────┴───────────────┐
                        ▼                                                ▼
              ┌───────────────────┐                        ┌─────────────────────┐
              │  API — FastAPI    │◀───── HTTP /predict ───│ Dashboard Streamlit │
              │  /health /predict │                        │ KPIs · EDA · simul. │
              └───────────────────┘                        └─────────────────────┘
```

## 2. Dados
Dados **sintéticos** gerados por código (`src/data/generate_synthetic.py`). O
alvo `churn` não é aleatório: vem de um **modelo logístico latente** cujos
coeficientes codificam intuição de negócio (contrato mês a mês ↑, tenure ↓,
cobrança alta ↑, fibra ↑, suporte/segurança ↓). Isso garante que exista sinal
real e interpretável para o ML aprender. Taxa de churn resultante ≈ 15%.

## 3. Feature Engineering
- `charges_per_tenure` — gasto mensal normalizado pelo tempo de casa; separa o
  cliente "caro e novo" (alto risco) do "caro e antigo" (fidelizado).
- `tenure_bucket` — discretização do tenure para capturar não-linearidade.
- `ColumnTransformer`: `StandardScaler` nas numéricas, `OneHotEncoder`
  (`drop="first"`, `handle_unknown="ignore"`) nas categóricas, passthrough nas
  binárias. Encapsulado no mesmo `Pipeline` do modelo → sem *training/serving skew*.

## 4. Modelagem
Três classificadores baseados em árvore comparados em holdout estratificado (20%):

| Modelo | ROC-AUC | PR-AUC | F1 | Recall (churn) |
|---|---|---|---|---|
| **RandomForest** | **0.828** | 0.426 | 0.456 | **0.730** |
| XGBoost | 0.814 | 0.429 | 0.302 | 0.215 |
| LightGBM | 0.808 | 0.408 | 0.466 | 0.652 |

**Critério de escolha:** ROC-AUC como métrica de ranqueamento + **recall na
classe churn** como métrica de negócio. O RandomForest (com `class_weight=
"balanced"`) vence nas duas. O XGBoost, sem rebalanceamento de classe, ficou
conservador demais (recall 0.21) — bom AUC, mas fraco para acionar retenção.

Todos os runs são registrados no **MLflow** (`mlflow.db`), permitindo comparar
parâmetros e métricas. A interpretabilidade global usa **SHAP** (`TreeExplainer`)
— ver `shap_summary.png`.

## 5. Serviço
- **API FastAPI**: `/health` (liveness + modelo carregado) e `/predict` (features
  → probabilidade + classe). Validação de entrada via Pydantic; modelo carregado
  uma vez no `lifespan`.
- **Dashboard Streamlit**: KPIs, gráficos de EDA e um simulador que consome a API.

## 6. Qualidade & Operação
- **Testes** (`pytest`): gerador, ETL (incl. validações), features, e API
  (contratos + sanidade de negócio alto vs. baixo risco).
- **CI** (GitHub Actions): instala dependências + roda a suíte a cada push/PR.
- **Docker Compose**: Postgres + API + dashboard, com healthchecks.
- **Config por env var**: `DATABASE_URL`, `MODEL_PATH`, `MLFLOW_TRACKING_URI`,
  `DECISION_THRESHOLD` — nada de segredo no código.

## 7. Limitações e próximos passos
- Dados sintéticos: úteis para demonstrar engenharia, mas não substituem dados
  reais (calibração de probabilidade seria o próximo passo com dados reais).
- **Monitoramento de drift** e re-treino agendado ficariam como Etapa 10.
- Ajuste de **threshold** por custo de negócio (hoje fixo em 0.5) e calibração
  (`CalibratedClassifierCV`).
- Registro de modelo no **MLflow Model Registry** para promoção staging→prod.
