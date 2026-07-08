"""Feature engineering para o modelo de churn.

Cria features derivadas com sentido de negócio e monta um ``ColumnTransformer``
(one-hot para categóricas, padronização para numéricas) que é persistido junto
com o modelo — garantindo que treino e inferência apliquem exatamente a mesma
transformação.

Uso:
    from src.features.build_features import add_derived_features, build_preprocessor
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
PREPROCESSOR_PATH = ROOT / "models" / "preprocessor.joblib"

TARGET = "churn"
# Colunas ignoradas como feature (identificador).
ID_COLS = ["customer_id"]

CATEGORICAL = ["contract", "internet_service", "payment_method"]
BINARY = [
    "senior_citizen", "partner", "dependents", "paperless_billing",
    "tech_support", "online_security",
]
# Numéricas de base + derivadas (preenchidas por add_derived_features).
NUMERIC = [
    "tenure_months", "monthly_charges", "total_charges",
    "charges_per_tenure", "tenure_bucket_code",
]

TENURE_BUCKETS = [0, 12, 24, 48, 72]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49-72"]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features derivadas ao DataFrame (não muta o original)."""
    df = df.copy()

    # Gasto médio por mês de permanência: separa cliente caro-e-novo de
    # cliente caro-e-antigo, que têm risco de churn bem diferente.
    df["charges_per_tenure"] = df["monthly_charges"] / df["tenure_months"].clip(lower=1)

    # Faixa de tempo de casa (bucket) — captura não-linearidade do tenure.
    df["tenure_bucket"] = pd.cut(
        df["tenure_months"], bins=TENURE_BUCKETS, labels=TENURE_LABELS,
        include_lowest=True,
    )
    df["tenure_bucket_code"] = df["tenure_bucket"].cat.codes

    return df


def build_preprocessor() -> ColumnTransformer:
    """Monta o ColumnTransformer (one-hot + scaling). Binárias passam direto."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL),
            ("bin", "passthrough", BINARY),
        ],
        remainder="drop",
    )


# Ordem canônica das colunas de entrada do modelo (usada em treino e inferência).
FEATURE_COLS = NUMERIC + CATEGORICAL + BINARY


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica derivadas e devolve só as colunas de feature, na ordem canônica.

    Usada tanto no treino quanto na inferência (API) para garantir consistência.
    """
    return add_derived_features(df)[FEATURE_COLS]


def split_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features (X) e alvo (y) já com as derivadas aplicadas."""
    return prepare_features(df), df[TARGET]


def save_preprocessor(preprocessor: ColumnTransformer, path: Path = PREPROCESSOR_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)


def load_preprocessor(path: Path = PREPROCESSOR_PATH) -> ColumnTransformer:
    return joblib.load(path)
