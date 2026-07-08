"""Testes de feature engineering (Etapa 4)."""
from src.data.generate_synthetic import generate_customers
from src.features.build_features import (
    FEATURE_COLS, add_derived_features, build_preprocessor, prepare_features,
)


def test_derived_features_present():
    df = add_derived_features(generate_customers(n=100, seed=2))
    assert "charges_per_tenure" in df.columns
    assert "tenure_bucket_code" in df.columns
    # charges_per_tenure nunca deve ser infinito (tenure é clipado em 1).
    assert df["charges_per_tenure"].notna().all()
    assert (df["charges_per_tenure"] < float("inf")).all()


def test_prepare_features_returns_canonical_columns():
    X = prepare_features(generate_customers(n=50, seed=3))
    assert list(X.columns) == FEATURE_COLS


def test_preprocessor_fits_and_transforms():
    df = generate_customers(n=200, seed=4)
    X = prepare_features(df)
    pre = build_preprocessor()
    Xt = pre.fit_transform(X)
    # Deve produzir uma matriz densa com uma linha por cliente.
    assert Xt.shape[0] == len(df)
    assert Xt.shape[1] >= len(FEATURE_COLS)
