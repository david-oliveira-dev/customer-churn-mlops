"""Dashboard Streamlit para o projeto de churn.

Três seções:
    1. KPIs de churn sobre a base sintética.
    2. Gráficos de EDA (churn por contrato, por tenure, por cobrança).
    3. Simulador que chama a API FastAPI (/predict) para um cliente hipotético.

Rodar:
    streamlit run app/dashboard.py
A URL da API é lida de ``API_URL`` (default http://localhost:8000).
"""
from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

from src.data.generate_synthetic import generate_customers

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Churn Analytics", page_icon="📉", layout="wide")


@st.cache_data
def load_data(n: int = 8000) -> pd.DataFrame:
    return generate_customers(n=n, seed=42)


df = load_data()

st.title("📉 Customer Churn — Analytics & Simulador")
st.caption("Plataforma de previsão de churn · dados sintéticos · modelo em produção via API")

# --- 1. KPIs ---
total = len(df)
churn_rate = df["churn"].mean()
avg_charges = df["monthly_charges"].mean()
avg_tenure = df["tenure_months"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Clientes", f"{total:,}")
c2.metric("Taxa de churn", f"{churn_rate:.1%}")
c3.metric("Cobrança média", f"R$ {avg_charges:.2f}")
c4.metric("Tenure médio", f"{avg_tenure:.0f} meses")

st.divider()

# --- 2. EDA ---
st.subheader("Churn por segmento")
col_a, col_b = st.columns(2)
with col_a:
    by_contract = df.groupby("contract")["churn"].mean().sort_values(ascending=False)
    st.bar_chart(by_contract, y_label="taxa de churn")
    st.caption("Contrato mês a mês concentra o churn.")
with col_b:
    by_internet = df.groupby("internet_service")["churn"].mean().sort_values(ascending=False)
    st.bar_chart(by_internet, y_label="taxa de churn")
    st.caption("Fibra (mais cara) tende a churnar mais.")

st.subheader("Churn por faixa de tempo de casa (tenure)")
df_t = df.copy()
df_t["faixa_tenure"] = pd.cut(
    df_t["tenure_months"], bins=[0, 12, 24, 48, 72],
    labels=["0-12", "13-24", "25-48", "49-72"], include_lowest=True,
)
st.bar_chart(df_t.groupby("faixa_tenure", observed=True)["churn"].mean(), y_label="taxa de churn")

st.divider()

# --- 3. Simulador ---
st.subheader("🔮 Simulador de churn (chama a API)")
with st.form("simulador"):
    c1, c2, c3 = st.columns(3)
    with c1:
        tenure = st.slider("Tenure (meses)", 0, 72, 5)
        monthly = st.slider("Cobrança mensal (R$)", 18, 130, 90)
        total = st.number_input("Cobrança total (R$)", 0.0, 12000.0, 450.0)
    with c2:
        contract = st.selectbox("Contrato", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet", ["Fiber optic", "DSL", "No"])
        payment = st.selectbox(
            "Pagamento",
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        )
    with c3:
        senior = st.checkbox("Idoso")
        partner = st.checkbox("Tem parceiro(a)")
        dependents = st.checkbox("Tem dependentes")
        paperless = st.checkbox("Fatura digital", value=True)
        tech = st.checkbox("Suporte técnico")
        security = st.checkbox("Segurança online")
    submitted = st.form_submit_button("Prever churn")

if submitted:
    payload = {
        "tenure_months": tenure, "monthly_charges": float(monthly),
        "total_charges": float(total), "contract": contract,
        "internet_service": internet, "payment_method": payment,
        "senior_citizen": int(senior), "partner": int(partner),
        "dependents": int(dependents), "paperless_billing": int(paperless),
        "tech_support": int(tech), "online_security": int(security),
    }
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        prob = result["churn_probability"]
        st.metric("Probabilidade de churn", f"{prob:.1%}")
        if result["churn"]:
            st.error("⚠️ Cliente com ALTO risco de churn — acionar retenção.")
        else:
            st.success("✅ Cliente com baixo risco de churn.")
    except requests.RequestException as exc:
        st.warning(f"Não foi possível chamar a API em {API_URL}. Ela está no ar? ({exc})")
