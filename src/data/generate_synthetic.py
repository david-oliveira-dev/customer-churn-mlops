"""Gerador de dados sintéticos de clientes para previsão de churn.

A ideia central: em vez de sortear o churn de forma aleatória, derivamos a
probabilidade de churn de um *modelo logístico latente* baseado nas features.
Assim o dataset tem **sinal real** — contrato mensal, tenure baixo, cobrança
alta e ausência de suporte aumentam o churn — e um modelo de ML consegue
aprender relações que fazem sentido de negócio.

Uso:
    python -m src.data.generate_synthetic --n 8000 --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET = ["DSL", "Fiber optic", "No"]
PAYMENTS = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_customers(n: int = 8000, seed: int = 42) -> pd.DataFrame:
    """Gera `n` clientes sintéticos com um alvo `churn` correlacionado.

    Retorna um DataFrame pronto para o ETL.
    """
    rng = np.random.default_rng(seed)

    # --- Features demográficas / de conta ---
    tenure = rng.integers(0, 72, size=n)  # meses como cliente
    contract = rng.choice(CONTRACTS, size=n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(INTERNET, size=n, p=[0.35, 0.45, 0.20])
    payment = rng.choice(PAYMENTS, size=n, p=[0.35, 0.20, 0.22, 0.23])
    senior = rng.binomial(1, 0.16, size=n)
    partner = rng.binomial(1, 0.48, size=n)
    dependents = rng.binomial(1, 0.30, size=n)
    paperless = rng.binomial(1, 0.59, size=n)
    tech_support = rng.binomial(1, 0.38, size=n)
    online_security = rng.binomial(1, 0.35, size=n)

    # Cobrança mensal depende do serviço de internet
    base_charge = np.where(internet == "Fiber optic", 75,
                           np.where(internet == "DSL", 55, 20))
    monthly_charges = np.round(
        base_charge + rng.normal(0, 8, size=n)
        + tech_support * 6 + online_security * 5, 2
    ).clip(18, 130)
    total_charges = np.round(monthly_charges * np.maximum(tenure, 1)
                             * rng.uniform(0.9, 1.05, size=n), 2)

    # --- Modelo logístico latente para o churn ---
    # Coeficientes escolhidos para refletir intuição de negócio.
    is_month = (contract == "Month-to-month").astype(float)
    is_two_year = (contract == "Two year").astype(float)
    is_fiber = (internet == "Fiber optic").astype(float)
    is_echeck = (payment == "Electronic check").astype(float)

    logit = (
        -1.4                                  # intercepto (base ~20% churn)
        + 1.5 * is_month                      # contrato mensal -> muito mais churn
        - 1.1 * is_two_year                   # contrato 2 anos -> retém
        - 0.045 * tenure                      # quanto mais tempo, menos churn
        + 0.015 * (monthly_charges - 65)      # cobrança alta -> churn
        + 0.6 * is_fiber                      # fibra cara insatisfaz
        + 0.5 * is_echeck                     # cheque eletrônico -> churn
        - 0.7 * tech_support                  # suporte retém
        - 0.5 * online_security               # segurança retém
        + 0.3 * senior                        # sênior churna um pouco mais
        - 0.25 * partner
    )
    prob = _sigmoid(logit)
    churn = rng.binomial(1, prob)

    df = pd.DataFrame({
        "customer_id": [f"C{100000 + i}" for i in range(n)],
        "tenure_months": tenure,
        "contract": contract,
        "internet_service": internet,
        "payment_method": payment,
        "senior_citizen": senior,
        "partner": partner,
        "dependents": dependents,
        "paperless_billing": paperless,
        "tech_support": tech_support,
        "online_security": online_security,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "churn": churn,
    })
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera dados sintéticos de churn.")
    parser.add_argument("--n", type=int, default=8000, help="nº de clientes")
    parser.add_argument("--seed", type=int, default=42, help="semente aleatória")
    parser.add_argument("--out", type=Path, default=RAW_DIR / "customers.csv")
    args = parser.parse_args()

    df = generate_customers(n=args.n, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    rate = df["churn"].mean()
    print(f"Gerados {len(df)} clientes -> {args.out}")
    print(f"Taxa de churn: {rate:.1%}")


if __name__ == "__main__":
    main()
