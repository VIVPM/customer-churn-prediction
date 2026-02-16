"""
Streamlit UI for Churn Prediction
==================================
Frontend that talks to the FastAPI backend.

Run:
  1. Start API:  uvicorn api:app --reload --port 8000
  2. Start UI:   streamlit run streamlit_app.py
"""

import streamlit as st
import requests
import pandas as pd
import os

# ---------------------------------------------------------------------------
# Config — reads from Streamlit secrets (deployed) or defaults to localhost
# ---------------------------------------------------------------------------
API_URL = os.environ.get("API_URL", "http://localhost:8000")
try:
    API_URL = st.secrets["API_URL"]
except (FileNotFoundError, KeyError):
    pass  # use env var or default

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Clean light CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Risk badges */
    .risk-high {
        background: #dc3545;
        color: white;
        padding: 10px 24px;
        border-radius: 24px;
        font-weight: bold;
        font-size: 20px;
        display: inline-block;
        text-align: center;
    }
    .risk-medium {
        background: #fd7e14;
        color: white;
        padding: 10px 24px;
        border-radius: 24px;
        font-weight: bold;
        font-size: 20px;
        display: inline-block;
        text-align: center;
    }
    .risk-low {
        background: #28a745;
        color: white;
        padding: 10px 24px;
        border-radius: 24px;
        font-weight: bold;
        font-size: 20px;
        display: inline-block;
        text-align: center;
    }

    /* Result card */
    .result-card {
        background: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: Check API connection
# ---------------------------------------------------------------------------
def check_api():
    try:
        r = requests.get(f"{API_URL}/model/info", timeout=3)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔮 Churn Predictor")
    st.markdown("---")

    api_ok = check_api()
    if api_ok:
        st.success("✅ API Connected")
        info = requests.get(f"{API_URL}/model/info").json()
        st.markdown(f"**Model:** `{info['model_name']}`")
        st.markdown(f"**Features:** `{info['num_features']}`")
    else:
        st.error("❌ API Offline")
        st.code("uvicorn api:app --reload", language="bash")

    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
    1. Start the FastAPI backend
    2. **Single Prediction** — fill form
    3. **Batch Prediction** — upload CSV
    """)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🔮 Customer Churn Prediction")
st.caption("Predict which customers are likely to leave — powered by Machine Learning")
st.markdown("---")


# ---------------------------------------------------------------------------
# Tabs (2 tabs only)
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🎯 Single Prediction", "📊 Batch Prediction"])


# ======================== TAB 1: Single Prediction ========================
with tab1:
    if not api_ok:
        st.warning("⚠️ Start the FastAPI backend to make predictions.")
    else:
        st.subheader("Enter Customer Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            customer_id = st.number_input("Customer ID", min_value=1, value=101, step=1)
            transaction_id = st.number_input("Transaction ID", min_value=1, value=5000, step=1)
            amount_spent = st.number_input("Amount Spent ($)", min_value=0.0, value=250.0, step=10.0)
            interaction_id = st.number_input("Interaction ID", min_value=0.0, value=3000.0, step=1.0)

        with col2:
            login_frequency = st.number_input("Login Frequency", min_value=0, value=15, step=1)
            transaction_year = st.number_input("Transaction Year", min_value=2020, max_value=2030, value=2022)
            interaction_month = st.selectbox("Interaction Month", list(range(1, 13)), index=5)
            product_category = st.selectbox("Product Category",
                                            ["Books", "Clothing", "Electronics", "Furniture", "Groceries"])

        with col3:
            interaction_type = st.selectbox("Interaction Type",
                                           ["Complaint", "Feedback", "Inquiry"])
            resolution_status = st.selectbox("Resolution Status", ["Resolved", "Unresolved"])
            service_usage = st.selectbox("Service Usage", ["Mobile App", "Online Banking", "Website"])

        st.markdown("")
        predict_btn = st.button("🔮 Predict Churn", use_container_width=True, type="primary")

        if predict_btn:
            payload = {
                "CustomerID": customer_id,
                "TransactionID": transaction_id,
                "AmountSpent": amount_spent,
                "InteractionID": interaction_id,
                "LoginFrequency": login_frequency,
                "TransactionYear": transaction_year,
                "InteractionMonth": interaction_month,
                "ProductCategory": product_category,
                "InteractionType": interaction_type,
                "ResolutionStatus": resolution_status,
                "ServiceUsage": service_usage,
            }

            with st.spinner("Predicting..."):
                resp = requests.post(f"{API_URL}/predict", json=payload)

            if resp.status_code == 200:
                result = resp.json()

                st.markdown("---")
                st.subheader("Prediction Result")

                r1, r2, r3 = st.columns(3)

                with r1:
                    risk = result["risk_level"]
                    css_class = f"risk-{risk.lower()}"
                    st.markdown(f'<div class="{css_class}">{risk} Risk</div>', unsafe_allow_html=True)

                with r2:
                    st.metric("Churn Probability", f"{result['churn_probability']:.1%}")

                with r3:
                    label = "⚠️ Will Churn" if result["churn_prediction"] == 1 else "✅ Will Stay"
                    st.metric("Prediction", label)

                st.info(f"**Recommendation:** {result['recommendation']}")
            else:
                st.error(f"API error: {resp.text}")


# ======================== TAB 2: Batch Prediction =========================
with tab2:
    if not api_ok:
        st.warning("⚠️ Start the FastAPI backend to make predictions.")
    else:
        st.subheader("Upload Customer CSV")
        st.info("""
        CSV columns needed: **CustomerID, TransactionID, AmountSpent, InteractionID,
        LoginFrequency, TransactionYear, InteractionMonth, ProductCategory, InteractionType,
        ResolutionStatus, ServiceUsage**
        """)

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            df_preview = pd.read_csv(uploaded_file)
            st.markdown(f"**Loaded {len(df_preview)} customers**")
            st.dataframe(df_preview.head(), use_container_width=True)

            if st.button("🔮 Predict All", use_container_width=True, type="primary"):
                uploaded_file.seek(0)

                with st.spinner(f"Predicting for {len(df_preview)} customers..."):
                    resp = requests.post(
                        f"{API_URL}/predict/batch",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                    )

                if resp.status_code == 200:
                    data = resp.json()
                    results_df = pd.DataFrame(data["predictions"])

                    st.markdown("---")
                    st.subheader("Results Summary")
                    m1, m2, m3, m4 = st.columns(4)
                    total = len(results_df)
                    churners = results_df["churn_prediction"].sum()

                    m1.metric("Total Customers", total)
                    m2.metric("Predicted Churners", int(churners))
                    m3.metric("Churn Rate", f"{churners / total:.1%}")
                    m4.metric("High Risk", int((results_df["risk_level"] == "High").sum()))

                    st.subheader("Detailed Predictions")
                    st.dataframe(results_df, use_container_width=True)

                    csv_out = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results CSV",
                        csv_out,
                        "churn_predictions.csv",
                        "text/csv",
                        use_container_width=True,
                    )
                else:
                    st.error(f"API error: {resp.text}")
