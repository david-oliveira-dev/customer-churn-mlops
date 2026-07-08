# Build Brief — Customer Churn MLOps Platform

> Este arquivo é o **roteiro de construção** do projeto. Um agente (local ou na
> nuvem via claude.ai/code) deve seguir estas etapas na ordem, commitando ao
> final de cada uma. O objetivo é qualidade de **portfólio para vaga de
> Cientista de Dados Pleno**.

## Objetivo
Plataforma ponta a ponta para prever **churn** (cancelamento de clientes),
pronta para produção: dados → ETL → features → modelo → API → dashboard →
monitoramento.

## Decisões já tomadas (não reabrir)
- **Dados:** sintéticos, gerados por código (nada de download). O gerador deve
  produzir sinal realista — churn correlacionado com contrato mensal, tenure
  baixo, cobrança alta e ausência de suporte técnico.
- **Dashboard:** Streamlit (roda em Linux, versiona no Git, funciona na nuvem).
- **Banco:** PostgreSQL (via SQLAlchemy; permitir fallback SQLite p/ testes/CI).
- **Modelos:** comparar RandomForest, XGBoost e LightGBM; interpretar com SHAP.
- **Tracking:** MLflow (armazenamento local em `mlruns/`).

## Etapas (construção incremental — commit ao fim de cada uma)

### Etapa 1 — Geração de dados sintéticos ✅ (feita no scaffold local)
- `src/data/generate_synthetic.py`: gera um DataFrame de clientes com features
  realistas e um alvo `churn` derivado de um modelo logístico latente.
- Salva `data/raw/customers.csv`.
- Teste em `tests/test_generate_synthetic.py` verificando shape, colunas,
  taxa de churn plausível (10%–35%) e reprodutibilidade (seed).

### Etapa 2 — ETL + carga no banco
- `src/data/etl.py`: lê `data/raw`, limpa/tipa, valida, grava em PostgreSQL
  (tabela `customers`) e também em `data/processed/customers.parquet`.
- Conexão via env var `DATABASE_URL` (fallback SQLite).

### Etapa 3 — EDA
- `notebooks/01-eda.ipynb`: distribuições, correlações, churn por segmento,
  insights de negócio escritos em markdown.

### Etapa 4 — Feature Engineering
- `src/features/build_features.py`: encoding, scaling, novas features
  (ex.: `charges_per_tenure`, buckets de tenure). Persistir o preprocessor.

### Etapa 5 — Treino e comparação de modelos
- `src/models/train.py`: treina RF/XGBoost/LightGBM, valida (ROC-AUC, PR-AUC,
  F1, recall na classe churn), registra tudo no MLflow, salva o melhor em
  `models/`. `src/models/explain.py` gera SHAP em `reports/`.

### Etapa 6 — API FastAPI
- `app/main.py`: endpoints `/health` e `/predict` (recebe features → prob. de
  churn + classe). Carrega modelo + preprocessor salvos.

### Etapa 7 — Dashboard Streamlit
- `app/dashboard.py`: KPIs de churn, gráficos da EDA, simulador que chama a API.

### Etapa 8 — Docker + Compose
- `Dockerfile` (API) e `docker-compose.yml` (API + Postgres + dashboard).

### Etapa 9 — Testes, CI, docs
- Cobertura de testes nas peças críticas; CI verde no GitHub Actions.
- `README.md` profissional: arquitetura, como rodar, resultados, trade-offs.
- Diagrama de arquitetura + relatório técnico em `reports/`.

## Padrões
- Código modular e tipado; funções pequenas; docstrings.
- Nada de segredos no repo; configs por env var.
- Commits pequenos, um por etapa. **Sem linha de Co-Authored-By.**
