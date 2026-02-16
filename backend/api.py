"""
FastAPI Backend for Churn Prediction
=====================================
Serves the trained model via REST API.

Run locally:  uvicorn api:app --reload --port 8000
Render:       auto-deploys via render.yaml
Docs:         http://localhost:8000/docs
"""

import pandas as pd
import numpy as np
import io
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from joblib import load

# ---------------------------------------------------------------------------
# Paths (relative to this file)
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).parent
MODELS_DIR = BACKEND_DIR / "models"
DATA_DIR = BACKEND_DIR / "data"

# ---------------------------------------------------------------------------
# Global model objects (loaded once at startup)
# ---------------------------------------------------------------------------
model = None
scaler = None
model_name = "unknown"
feature_names = []

# One-hot mapping: user-friendly value → which dummy columns to set to 1
CATEGORY_ENCODINGS = {
    "ProductCategory": {
        "columns": [
            "ProductCategory_Clothing",
            "ProductCategory_Electronics",
            "ProductCategory_Furniture",
            "ProductCategory_Groceries",
        ],
        "base": "Books",
    },
    "InteractionType": {
        "columns": [
            "InteractionType_Feedback",
            "InteractionType_Inquiry",
        ],
        "base": "Complaint",
    },
    "ResolutionStatus": {
        "columns": ["ResolutionStatus_Unresolved"],
        "base": "Resolved",
    },
    "ServiceUsage": {
        "columns": [
            "ServiceUsage_Online Banking",
            "ServiceUsage_Website",
        ],
        "base": "Mobile App",
    },
}


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, model_name, feature_names

    model = load(MODELS_DIR / "best_model.joblib")
    scaler = load(MODELS_DIR / "scaler.joblib")

    name_file = MODELS_DIR / "best_model_name.txt"
    if name_file.exists():
        model_name = name_file.read_text().strip()

    feat_file = DATA_DIR / "feature_names_selected.csv"
    feature_names = pd.read_csv(feat_file)["feature"].tolist()

    print(f"✅ Model loaded: {model_name} ({len(feature_names)} features)")
    yield
    print("🛑 Shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Churn Prediction API",
    description="Predict customer churn using the trained Decision Tree model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class CustomerInput(BaseModel):
    CustomerID: int = Field(..., example=101)
    TransactionID: int = Field(..., example=5000)
    AmountSpent: float = Field(..., example=250.0)
    InteractionID: float = Field(..., example=3000)
    LoginFrequency: int = Field(..., example=15)
    TransactionYear: int = Field(2022, example=2022)
    InteractionMonth: int = Field(6, ge=1, le=12, example=6)
    ProductCategory: str = Field("Electronics", example="Electronics")
    InteractionType: str = Field("Inquiry", example="Inquiry")
    ResolutionStatus: str = Field("Resolved", example="Resolved")
    ServiceUsage: str = Field("Mobile App", example="Mobile App")


class PredictionResult(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str
    recommendation: str


class ModelInfo(BaseModel):
    model_name: str
    num_features: int
    feature_names: list[str]
    categories: dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _encode_customer(data: CustomerInput) -> pd.DataFrame:
    row = {feat: 0 for feat in feature_names}

    row["CustomerID"] = data.CustomerID
    row["TransactionID"] = data.TransactionID
    row["AmountSpent"] = data.AmountSpent
    row["InteractionID"] = data.InteractionID
    row["LoginFrequency"] = data.LoginFrequency
    row["TransactionYear"] = data.TransactionYear
    row["InteractionMonth"] = data.InteractionMonth

    for cat_field, info in CATEGORY_ENCODINGS.items():
        value = getattr(data, cat_field)
        for col in info["columns"]:
            if col.endswith(f"_{value}"):
                row[col] = 1

    return pd.DataFrame([row])[feature_names]


def _get_risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "High"
    elif prob >= 0.4:
        return "Medium"
    return "Low"


def _get_recommendation(risk: str) -> str:
    return {
        "High": "⚠️ Immediate action required! Contact customer with special retention offer.",
        "Medium": "📋 Monitor closely. Consider proactive engagement and loyalty rewards.",
        "Low": "✅ Low risk. Continue regular engagement and satisfaction monitoring.",
    }.get(risk, "")


def _predict_single(df: pd.DataFrame) -> PredictionResult:
    scaled = scaler.transform(df)

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(scaled)[0][1])
        pred = int(prob >= 0.5)
    else:
        pred = int(model.predict(scaled)[0])
        prob = float(pred)

    risk = _get_risk_level(prob)
    return PredictionResult(
        churn_prediction=pred,
        churn_probability=round(prob, 4),
        risk_level=risk,
        recommendation=_get_recommendation(risk),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"status": "running", "model": model_name, "docs": "/docs"}


@app.post("/predict", response_model=PredictionResult)
async def predict(customer: CustomerInput):
    """Predict churn for a single customer."""
    df = _encode_customer(customer)
    return _predict_single(df)


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """Predict churn for multiple customers from CSV upload."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    contents = await file.read()
    df_input = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    results = []
    for _, row in df_input.iterrows():
        customer = CustomerInput(**row.to_dict())
        encoded = _encode_customer(customer)
        result = _predict_single(encoded)
        results.append({
            "CustomerID": int(row.get("CustomerID", 0)),
            **result.model_dump(),
        })

    return {"total": len(results), "predictions": results}


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the deployed model."""
    categories = {}
    for cat_field, info in CATEGORY_ENCODINGS.items():
        options = [col.split("_", 1)[1] for col in info["columns"]]
        options.append(info["base"])
        categories[cat_field] = sorted(options)

    return ModelInfo(
        model_name=model_name,
        num_features=len(feature_names),
        feature_names=feature_names,
        categories=categories,
    )
