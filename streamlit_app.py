"""
Streamlit UI for Churn Prediction
==================================
Frontend that talks to the FastAPI backend.

Run:
  1. Start API:  cd backend && uvicorn api:app --reload --port 8000
  2. Start UI:   streamlit run streamlit_app.py
"""

import streamlit as st
import requests
import pandas as pd
import os
import time

# ---------------------------------------------------------------------------
# Config — reads from Streamlit secrets (deployed) or defaults to localhost
# ---------------------------------------------------------------------------
API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
try:
    API_URL = st.secrets["API_URL"].rstrip("/")
except (FileNotFoundError, KeyError):
    pass

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
# CSS
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

    /* Training status cards */
    .status-running {
        background: #0d6efd;
        border: 2px solid #0a58ca;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0;
        color: white;
    }
    .status-completed {
        background: #157347;
        border: 2px solid #0f5132;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0;
        color: white;
    }
    .status-failed {
        background: #b02a37;
        border: 2px solid #842029;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0;
        color: white;
    }

    /* Step indicators */
    .step-done   { color: #28a745; font-weight: 600; }
    .step-active { color: #fd7e14; font-weight: 600; }
    .step-wait   { color: #adb5bd; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: Check API connection
# ---------------------------------------------------------------------------
def check_api():
    try:
        r = requests.get(f"{API_URL}/", timeout=15)
        return r.status_code == 200, r.json()
    except (requests.ConnectionError, requests.Timeout):
        return False, {}


def get_model_versions():
    try:
        r = requests.get(f"{API_URL}/model/versions", timeout=10)
        if r.status_code == 200:
            return r.json().get("versions", [])
    except Exception:
        pass
    return []

def get_model_info(version="main"):
    try:
        r = requests.get(f"{API_URL}/model/info?version={version}", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔮 Churn Predictor")
    st.markdown("---")

    api_ok, root_data = check_api()
    model_loaded = root_data.get("model_loaded", False)

    if api_ok:
        st.success("✅ API Connected")
        
        # Version Selection
        versions = get_model_versions()
        
        if not versions:
             st.warning("⚠️ No model versions found on HF Hub. Train a model first.")
             st.session_state["selected_version"] = "local"
        else:
             selected_version = st.selectbox(
                 "📂 Select Model Version",
                 options=reversed(versions), # Show newest first
                 index=0
             )
             st.session_state["selected_version"] = selected_version

             info = get_model_info(selected_version)
             if info:
                 st.markdown(f"**Model:** `{info['model_name']}`")
                 st.markdown(f"**Features:** `{info['num_features']}`")
                 st.markdown(f"**Version:** `{selected_version}`")
             else:
                 st.warning("⚠️ Model not loaded yet — train first")
    else:
        st.error("❌ API Offline")
        st.code("cd backend && uvicorn api:app --reload", language="bash")

    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
    1. **Train Model** — upload Excel data
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
# Tabs
# ---------------------------------------------------------------------------
tab_train, tab1, tab2 = st.tabs([
    "🏋️ Train Model",
    "🎯 Single Prediction",
    "📊 Batch Prediction",
])


# ======================== TAB 0: Train Model ================================
with tab_train:
    st.subheader("🏋️ Retrain the Prediction Model")
    st.markdown("""
    Upload the raw Excel data file to retrain the model from scratch.
    The pipeline runs:
    - **Step 1** — Data Preprocessing (missing values, outlier removal, encoding)
    - **Step 2** — Feature Engineering (correlation analysis, feature selection)
    - **Step 3** — Model Training (GridSearchCV across SVM, Random Forest, Logistic Regression, Decision Tree)
    """)

    if not api_ok:
        st.warning("⚠️ Start the FastAPI backend to use training.")
    else:
        st.markdown("#### 📁 Upload Training Data")
        st.info("Upload your `Customer_Churn_Data_Large.xlsx` file (multi-sheet Excel)")

        uploaded_excel = st.file_uploader(
            "Choose Excel file (.xlsx)",
            type=["xlsx", "xls"],
            key="train_upload"
        )

        col_btn, col_status = st.columns([1, 2])

        with col_btn:
            train_btn = st.button(
                "🚀 Start Training",
                use_container_width=True,
                type="primary",
                disabled=(uploaded_excel is None),
            )

        if train_btn and uploaded_excel is not None:
            with st.spinner("Uploading data and starting training..."):
                resp = requests.post(
                    f"{API_URL}/train",
                    files={"file": (uploaded_excel.name, uploaded_excel.getvalue(),
                                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
                    timeout=30,
                )
            if resp.status_code == 200:
                st.success("✅ Training started! Monitor progress below.")
                st.session_state["training_triggered"] = True
            elif resp.status_code == 409:
                st.warning("⚠️ Training already in progress.")
                st.session_state["training_triggered"] = True
            else:
                st.error(f"❌ Failed to start training: {resp.text}")

        # ── Training Status Panel ──────────────────────────────────────────
        if not st.session_state.get("training_triggered", False):
            st.info("📂 Upload your Excel file and click **Start Training** to begin.")
        else:
            st.markdown("---")
            st.markdown("#### 📊 Training Status")

        status_placeholder = st.empty()
        steps_placeholder = st.empty()
        metrics_placeholder = st.empty()
        refresh_placeholder = st.empty()

        def render_status():
            status = "idle"
            s = {}

            # First, check if there's an active or recently completed training in this session
            if st.session_state.get("training_triggered", False):
                try:
                    r = requests.get(f"{API_URL}/train/status", timeout=10)
                    if r.status_code == 200:
                        s = r.json()
                        status = s.get("status", "idle")
                except Exception:
                    pass

            # If no active training is going on, check if a model version is loaded/selected
            if status == "idle":
                selected = st.session_state.get("selected_version")
                if selected and selected != "local":
                    info = get_model_info(selected)
                    if info:
                        status = "loaded_from_hf"
                        s = {
                            "message": f"Viewing model details for {selected} from Hugging Face Hub.",
                            "model_name": info.get("model_name"),
                            "best_cv_score": info.get("best_cv_score"),
                            "num_features": info.get("num_features")
                        }

            if status == "idle":
                status_placeholder.info("💤 No models trained yet. Upload data and click **Start Training**.")

            elif status == "running":
                message = s.get("message", "")
                status_placeholder.markdown(
                    f'<div class="status-running">🔄 <strong>Training in Progress</strong><br>{message}</div>',
                    unsafe_allow_html=True
                )
                # Step indicators based on message
                msg_lower = message.lower()
                step1 = "✅" if "step 2" in msg_lower or "step 3" in msg_lower or "complete" in msg_lower else (
                    "🔄" if "step 1" in msg_lower else "⏳")
                step2 = "✅" if "step 3" in msg_lower or "complete" in msg_lower else (
                    "🔄" if "step 2" in msg_lower else "⏳")
                step3 = "✅" if "complete" in msg_lower else (
                    "🔄" if "step 3" in msg_lower else "⏳")

                steps_placeholder.markdown(f"""
                | Step | Task | Status |
                |------|------|--------|
                | 1 | Data Preprocessing | {step1} |
                | 2 | Feature Engineering | {step2} |
                | 3 | Model Training (GridSearchCV) | {step3} |
                """)

            elif status in ("completed", "loaded_from_hf"):
                message = s.get("message", "")
                title = "Training Complete!" if status == "completed" else "Model Loaded"
                status_placeholder.markdown(
                    f'<div class="status-completed">✅ <strong>{title}</strong><br>{message}</div>',
                    unsafe_allow_html=True
                )
                steps_placeholder.markdown("""
                #### 📊Training Status
                | Step | Task | Status |
                |------|------|--------|
                | 1 | Data Preprocessing | ✅ |
                | 2 | Feature Engineering | ✅ |
                | 3 | Model Training (GridSearchCV) | ✅ |
                """)
                m1, m2, m3 = metrics_placeholder.columns(3)
                m1.metric("🏆 Best Model", s.get("model_name", "-"))
                cv = s.get("best_cv_score")
                m2.metric("📈 CV Score", f"{cv:.4f}" if cv and pd.notna(cv) else "-")
                m3.metric("🔢 Features Used", s.get("num_features", "-"))

            elif status == "failed":
                message = s.get("message", "")
                status_placeholder.markdown(
                    f'<div class="status-failed">❌ <strong>Training Failed</strong><br>{message}</div>',
                    unsafe_allow_html=True
                )
                if s.get("error"):
                    with st.expander("🔍 Error Details"):
                        st.code(s["error"], language="python")

            return status

        # Manual refresh only — no auto-rerun
        current_status = render_status()

        if current_status == "running":
            refresh_placeholder.markdown("*⏳ Training running in background — click Refresh to check progress.*")
            if st.button("🔄 Refresh Status"):
                st.rerun()
        else:
            refresh_placeholder.empty()
            if current_status in ("completed", "failed"):
                if st.button("🔄 Refresh Status"):
                    st.rerun()


# ======================== TAB 1: Single Prediction ==========================
with tab1:
    if not api_ok:
        st.warning("⚠️ Start the FastAPI backend to make predictions.")

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

    if predict_btn and api_ok:
        if not versions:
            st.warning("⚠️ **Training Required:** No models are available on Hugging Face Hub. Please go to the **Train Model** tab to train and register your first model!")
            st.stop()
            
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
            version = st.session_state.get("selected_version", "main")
            resp = requests.post(f"{API_URL}/predict?version={version}", json=payload)

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


# ======================== TAB 2: Batch Prediction ===========================
with tab2:
    if not api_ok:
        st.warning("⚠️ Start the FastAPI backend to make predictions.")

    st.subheader("Upload Customer CSV")
    st.info("""
    CSV columns needed: **CustomerID, TransactionID, AmountSpent, InteractionID,
    LoginFrequency, TransactionYear, InteractionMonth, ProductCategory, InteractionType,
    ResolutionStatus, ServiceUsage**
    """)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None and api_ok:
        df_preview = pd.read_csv(uploaded_file)
        st.markdown(f"**Loaded {len(df_preview)} customers**")
        st.dataframe(df_preview.head(), use_container_width=True)

        if st.button("🔮 Predict All", use_container_width=True, type="primary"):
            if not versions:
                st.warning("⚠️ **Training Required:** No models are available on Hugging Face Hub. Please go to the **Train Model** tab to train and register your first model!")
                st.stop()
                
            uploaded_file.seek(0)

            with st.spinner(f"Predicting for {len(df_preview)} customers..."):
                version = st.session_state.get("selected_version", "main")
                resp = requests.post(
                    f"{API_URL}/predict/batch?version={version}",
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
