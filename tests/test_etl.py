"""Testes do ETL (Etapa 2)."""
import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.data.etl import TABLE, clean, run_etl
from src.data.generate_synthetic import generate_customers


def _raw_frame(n: int = 200) -> pd.DataFrame:
    return generate_customers(n=n, seed=5)


def test_clean_types_and_passes_valid_data():
    df = clean(_raw_frame())
    assert df["monthly_charges"].dtype.kind == "f"
    assert df["churn"].dtype == "int8"
    assert str(df["contract"].dtype) == "category"


def test_clean_rejects_bad_tenure():
    df = _raw_frame()
    df.loc[0, "tenure_months"] = 999
    with pytest.raises(ValueError, match="tenure_months"):
        clean(df)


def test_clean_rejects_duplicate_id():
    df = _raw_frame()
    df.loc[1, "customer_id"] = df.loc[0, "customer_id"]
    with pytest.raises(ValueError, match="duplicado"):
        clean(df)


def test_run_etl_writes_parquet_and_db(tmp_path):
    raw = tmp_path / "customers.csv"
    parquet = tmp_path / "customers.parquet"
    _raw_frame(300).to_csv(raw, index=False)
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")

    df = run_etl(raw_csv=raw, parquet_out=parquet, engine=engine)

    assert parquet.exists()
    from_db = pd.read_sql(f"SELECT * FROM {TABLE}", engine)
    assert len(from_db) == len(df) == 300


def test_run_etl_missing_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_etl(raw_csv=tmp_path / "nope.csv", parquet_out=tmp_path / "x.parquet",
                engine=create_engine("sqlite://"))
