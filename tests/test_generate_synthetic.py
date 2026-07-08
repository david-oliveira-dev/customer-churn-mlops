"""Testes do gerador de dados sintéticos (Etapa 1)."""
from src.data.generate_synthetic import generate_customers

EXPECTED_COLUMNS = {
    "customer_id", "tenure_months", "contract", "internet_service",
    "payment_method", "senior_citizen", "partner", "dependents",
    "paperless_billing", "tech_support", "online_security",
    "monthly_charges", "total_charges", "churn",
}


def test_shape_and_columns():
    df = generate_customers(n=500, seed=1)
    assert len(df) == 500
    assert set(df.columns) == EXPECTED_COLUMNS
    assert df["customer_id"].is_unique


def test_churn_rate_plausible():
    df = generate_customers(n=5000, seed=42)
    rate = df["churn"].mean()
    assert 0.10 <= rate <= 0.35, f"taxa de churn fora do esperado: {rate:.2%}"


def test_reproducible_with_seed():
    a = generate_customers(n=300, seed=7)
    b = generate_customers(n=300, seed=7)
    assert a.equals(b)


def test_signal_month_to_month_churns_more():
    # Sanidade: contrato mensal deve churnar mais que contrato de 2 anos.
    df = generate_customers(n=8000, seed=3)
    monthly = df.loc[df["contract"] == "Month-to-month", "churn"].mean()
    two_year = df.loc[df["contract"] == "Two year", "churn"].mean()
    assert monthly > two_year
