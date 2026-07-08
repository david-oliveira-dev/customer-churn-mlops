"""ETL: lê os dados brutos, limpa/tipa, valida e carrega no banco.

Fluxo:
    data/raw/customers.csv  ->  validação/tipagem  ->  PostgreSQL (tabela
    `customers`) + data/processed/customers.parquet

A conexão com o banco vem da env var ``DATABASE_URL``. Se ela não estiver
definida, cai no fallback **SQLite** (arquivo local), de modo que testes e CI
rodem sem um PostgreSQL de verdade.

Uso:
    python -m src.data.etl
    DATABASE_URL=postgresql+psycopg2://user:pass@host/db python -m src.data.etl
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "raw" / "customers.csv"
PROCESSED_PARQUET = ROOT / "data" / "processed" / "customers.parquet"
SQLITE_FALLBACK = ROOT / "data" / "processed" / "churn.db"
TABLE = "customers"

# Tipos-alvo por coluna: garante consistência entre CSV, parquet e banco.
BINARY_COLS = [
    "senior_citizen", "partner", "dependents", "paperless_billing",
    "tech_support", "online_security", "churn",
]
CATEGORICAL_COLS = ["contract", "internet_service", "payment_method"]
NUMERIC_COLS = ["tenure_months", "monthly_charges", "total_charges"]


def get_engine(database_url: str | None = None) -> Engine:
    """Cria um Engine do SQLAlchemy a partir de ``DATABASE_URL``.

    Sem a variável, usa SQLite local (fallback para testes/CI).
    """
    url = database_url or os.getenv("DATABASE_URL")
    if not url:
        SQLITE_FALLBACK.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{SQLITE_FALLBACK}"
    return create_engine(url)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa e tipa o DataFrame bruto, validando invariantes de negócio."""
    df = df.copy()

    # Tipagem explícita.
    df["customer_id"] = df["customer_id"].astype("string")
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="raise")
    for col in BINARY_COLS:
        df[col] = df[col].astype("int8")
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    # Validações — falhar cedo é melhor que treinar em lixo.
    if df["customer_id"].duplicated().any():
        raise ValueError("customer_id duplicado encontrado no CSV bruto.")
    if df[NUMERIC_COLS].isna().any().any():
        raise ValueError("valores nulos em colunas numéricas.")
    if not df["tenure_months"].between(0, 72).all():
        raise ValueError("tenure_months fora do intervalo esperado [0, 72].")
    if not set(df["churn"].unique()).issubset({0, 1}):
        raise ValueError("churn deve ser binário (0/1).")

    return df.reset_index(drop=True)


def run_etl(
    raw_csv: Path = RAW_CSV,
    parquet_out: Path = PROCESSED_PARQUET,
    engine: Engine | None = None,
) -> pd.DataFrame:
    """Executa o ETL completo e retorna o DataFrame limpo."""
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"{raw_csv} não existe. Rode antes: python -m src.data.generate_synthetic"
        )

    df = clean(pd.read_csv(raw_csv))

    parquet_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_out, index=False)

    engine = engine or get_engine()
    df.to_sql(TABLE, engine, if_exists="replace", index=False)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="ETL de clientes para churn.")
    parser.add_argument("--raw", type=Path, default=RAW_CSV)
    parser.add_argument("--parquet", type=Path, default=PROCESSED_PARQUET)
    args = parser.parse_args()

    engine = get_engine()
    df = run_etl(raw_csv=args.raw, parquet_out=args.parquet, engine=engine)
    print(f"ETL concluído: {len(df)} linhas")
    print(f"  -> parquet: {args.parquet}")
    print(f"  -> banco:   {engine.url}")


if __name__ == "__main__":
    main()
