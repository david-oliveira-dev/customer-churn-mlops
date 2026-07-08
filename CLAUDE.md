# CLAUDE.md — Customer Churn MLOps Platform

Contexto e convenções para qualquer sessão do Claude Code (local ou nuvem)
trabalhando neste repositório.

## O que é
Projeto de portfólio (Cientista de Dados Pleno): plataforma ponta a ponta de
previsão de churn. O roteiro completo de construção está em **`BUILD_BRIEF.md`**
— siga-o etapa a etapa.

## Como trabalhar aqui
- Construção **incremental**: uma etapa do BUILD_BRIEF por vez, com commit ao fim.
- Explique decisões técnicas nos commits e no README (é portfólio).
- **Commits sem a linha `Co-Authored-By`.**
- Não comitar dados grandes nem segredos. Configs sempre por variável de ambiente.

## Stack
Python 3.12, Pandas/NumPy, scikit-learn, XGBoost, LightGBM, SHAP, MLflow,
FastAPI, Streamlit, SQLAlchemy + PostgreSQL (fallback SQLite), Docker, pytest.

## Ambiente
- Use `venv`; o `pip` global da máquina do dono é bloqueado (PEP 668).
- Testes e CI devem funcionar **sem** PostgreSQL (fallback SQLite via `DATABASE_URL`).

## Rodando na nuvem (PC desligado)
Este repo é preparado para rodar via **claude.ai/code** ou rotina agendada.
O agente da nuvem deve ler `BUILD_BRIEF.md` e continuar da próxima etapa pendente.
