"""
FastAPI Backend for Churn Prediction
=====================================
Serves the trained model via REST API.

Run locally:  uvicorn api:app --reload --port 8000
Docs:         http://localhost:8000/docs
"""

import pandas as pd
import numpy as np
import io
import sys
import threading
import traceback
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from joblib import load
import os

# Load .env from backend directory
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars


try:
    from huggingface_hub import HfApi, hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# ---------------------------------------------------------------------------
# HuggingFace Hub config (read from environment)
# ---------------------------------------------------------------------------
HF_TOKEN   = os.environ.get("HF_TOKEN", "")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "vivpm99/customer-churn-model")

# Files to sync with HF Hub
HF_FILES = [
    "best_model.joblib",
    "scaler.joblib",
    "best_model_name.txt",
]

# ---------------------------------------------------------------------------
# Paths (relative to project root, not backend folder)
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent
MODELS_DIR = BACKEND_DIR / "models"
DATA_DIR = BACKEND_DIR / "data"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Add project root to sys.path so src.* imports work
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Global model objects (loaded once at startup)
# ---------------------------------------------------------------------------
model = None
scaler = None
model_name = "unknown"
feature_names = []

# ---------------------------------------------------------------------------
# Training status tracking
# ---------------------------------------------------------------------------
training_status = {
    "status": "idle",           # idle | running | completed | failed
    "started_at": None,
    "completed_at": None,
    "model_name": None,
    "best_cv_score": None,
    "num_features": None,
    "message": "No training has been run yet.",
    "error": None,
}
training_lock = threading.Lock()

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
# HuggingFace Hub helpers
# ---------------------------------------------------------------------------
def _hf_enabled() -> bool:
    """Returns True if HF Hub is configured and available."""
    return HF_AVAILABLE and bool(HF_TOKEN) and HF_TOKEN != "your_hf_token_here"


def _upload_to_hf():
    """Upload model artifacts from MODELS_DIR to HuggingFace Hub."""
    if not _hf_enabled():
        print("⚠️  HF Hub not configured — skipping upload.")
        return False
    try:
        api = HfApi(token=HF_TOKEN)
        # Create repo if it doesn't exist
        api.create_repo(repo_id=HF_REPO_ID, exist_ok=True, private=False)
        for fname in HF_FILES:
            fpath = MODELS_DIR / fname
            if fpath.exists():
                api.upload_file(
                    path_or_fileobj=str(fpath),
                    path_in_repo=fname,
                    repo_id=HF_REPO_ID,
                    token=HF_TOKEN,
                )
                print(f"☁️  Uploaded {fname} → {HF_REPO_ID}")
        return True
    except Exception as e:
        print(f"❌ HF upload failed: {e}")
        return False


def _download_from_hf():
    """Download model artifacts from HuggingFace Hub into MODELS_DIR."""
    if not _hf_enabled():
        return False
        
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Check if the repo exists first
    try:
        api = HfApi(token=HF_TOKEN)
        api.repo_info(repo_id=HF_REPO_ID)
    except Exception as e:
        if "404" in str(e):
            raise FileNotFoundError(f"HuggingFace repo '{HF_REPO_ID}' does not exist yet. Please train a model first.")
        raise
        
    for fname in HF_FILES:
        try:
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=fname,
                token=HF_TOKEN,
                local_dir=str(MODELS_DIR),
                local_dir_use_symlinks=False, # Force actual download, don't just link cache
            )
            print(f"⬇️  Downloaded {fname} from {HF_REPO_ID}")
        except Exception as e:
            if "404" in str(e):
                raise FileNotFoundError(f"File '{fname}' missing in repo '{HF_REPO_ID}'. Please train a model.")
            raise e
    return True


# ---------------------------------------------------------------------------
# Helper: Load model artifacts from local disk (after HF sync)
# ---------------------------------------------------------------------------
def _load_model_artifacts():
    """Download from HF Hub (if configured), then load from local MODELS_DIR."""
    global model, scaler, model_name, feature_names

    # Try to pull latest from HuggingFace Hub first
    if _hf_enabled():
        print(f"🔄 Pulling model from HuggingFace Hub ({HF_REPO_ID})...")
        _download_from_hf()
    else:
        print("ℹ️  HF Hub not configured — loading from local files.")

    model_path  = MODELS_DIR / "best_model.joblib"
    scaler_path = MODELS_DIR / "scaler.joblib"

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            "Model files not found. Please run training first via POST /train."
        )

    model  = load(model_path)
    scaler = load(scaler_path)

    name_file = MODELS_DIR / "best_model_name.txt"
    if name_file.exists():
        model_name = name_file.read_text().strip()

    feat_file = DATA_DIR / "feature_names_selected.csv"
    if not feat_file.exists():
        feat_file = DATA_DIR / "feature_names.csv"
    feature_names = pd.read_csv(feat_file)["feature"].tolist()

    print(f"✅ Model loaded: {model_name} ({len(feature_names)} features)")


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_model_artifacts()
    except FileNotFoundError as e:
        print(f"⚠️  Could not load model on startup: {e}")
        print("    Use POST /train to train and save a model first.")
    yield
    print("🛑 Shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Churn Prediction API",
    description="Predict customer churn and retrain the model via REST API.",
    version="2.0.0",
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


class TrainingStatusResponse(BaseModel):
    status: str
    started_at: Optional[str]
    completed_at: Optional[str]
    model_name: Optional[str]
    best_cv_score: Optional[float]
    num_features: Optional[int]
    message: str
    error: Optional[str]


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
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first via POST /train.",
        )
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
# Background training worker
# ---------------------------------------------------------------------------
def _run_training_pipeline(data_path: Path):
    """
    Run the full training pipeline in a background thread.
    Saves best_model.joblib and scaler.joblib to MODELS_DIR.
    """
    global training_status

    with training_lock:
        training_status["status"] = "running"
        training_status["started_at"] = datetime.now().isoformat()
        training_status["completed_at"] = None
        training_status["error"] = None
        training_status["message"] = "Training started..."

    try:
        from src.preprocessing import preprocess_data
        from src.feature_engineering import run_feature_engineering
        from src.train import train_models

        # Step 1: Preprocess
        with training_lock:
            training_status["message"] = "Step 1/3: Preprocessing data..."
        print("🔄 Running preprocessing...")
        preprocess_data()

        # Step 2: Feature Engineering
        with training_lock:
            training_status["message"] = "Step 2/3: Running feature engineering..."
        print("🔄 Running feature engineering...")
        run_feature_engineering()

        # Step 3: Train
        with training_lock:
            training_status["message"] = "Step 3/3: Training models with GridSearchCV..."
        print("🔄 Training models...")
        best_model, best_scaler, scores_df = train_models()

        # Copy artifacts to backend/models/
        import shutil
        src_models = PROJECT_ROOT / "models"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src_models / "best_model.joblib",   MODELS_DIR / "best_model.joblib")
        shutil.copy2(src_models / "scaler.joblib",        MODELS_DIR / "scaler.joblib")
        shutil.copy2(src_models / "best_model_name.txt",  MODELS_DIR / "best_model_name.txt")

        # Copy feature names to backend/data/
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        processed_dir = PROJECT_ROOT / "data" / "processed"
        feat_selected = processed_dir / "feature_names_selected.csv"
        feat_basic    = processed_dir / "feature_names.csv"
        if feat_selected.exists():
            shutil.copy2(feat_selected, DATA_DIR / "feature_names_selected.csv")
        elif feat_basic.exists():
            shutil.copy2(feat_basic, DATA_DIR / "feature_names.csv")

        # Upload to HuggingFace Hub
        with training_lock:
            training_status["message"] = "Uploading model to HuggingFace Hub..."
        hf_uploaded = _upload_to_hf()

        # Read best model info from scores_df
        best_row = scores_df.loc[scores_df["best_score"].idxmax()]
        best_cv = float(best_row["best_score"])
        best_name = str(best_row["model"])

        # Reload models into memory
        _load_model_artifacts()

        with training_lock:
            training_status["status"]       = "completed"
            training_status["completed_at"] = datetime.now().isoformat()
            training_status["model_name"]   = model_name
            training_status["best_cv_score"] = round(best_cv, 4)
            training_status["num_features"] = len(feature_names)
            hf_note = f" | Uploaded to HF Hub ({HF_REPO_ID})" if hf_uploaded else ""
            training_status["message"] = (
                f"Training complete! Best model: {model_name} "
                f"(CV score: {best_cv:.4f}){hf_note}"
            )

        print(f"✅ Training complete. Best model: {model_name}, CV: {best_cv:.4f}")

    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"❌ Training failed:\n{err_msg}")
        with training_lock:
            training_status["status"] = "failed"
            training_status["completed_at"] = datetime.now().isoformat()
            training_status["message"] = f"Training failed: {str(e)}"
            training_status["error"] = err_msg


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "status": "running",
        "model": model_name,
        "model_loaded": model is not None,
        "docs": "/docs",
    }


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
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
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


@app.post("/train")
async def trigger_training(file: UploadFile = File(...)):
    """
    Upload the raw Excel data file and trigger model retraining.

    - Accepts: .xlsx file (Customer_Churn_Data_Large.xlsx)
    - Saves file to data/raw/, runs full pipeline, saves joblib locally
    - Training runs in background — poll GET /train/status for progress
    """
    global training_status

    # Reject if already running
    with training_lock:
        if training_status["status"] == "running":
            raise HTTPException(
                status_code=409,
                detail="Training is already in progress. Check GET /train/status.",
            )

    # Validate file type
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Only Excel (.xlsx) files are accepted.")

    # Save uploaded file to data/raw/
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_path = RAW_DATA_DIR / "Customer_Churn_Data_Large.xlsx"
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    print(f"📁 Uploaded data saved to: {save_path}")

    # Launch training in background thread
    thread = threading.Thread(
        target=_run_training_pipeline,
        args=(save_path,),
        daemon=True,
    )
    thread.start()

    return {
        "message": "Training started in background.",
        "status": "running",
        "poll_url": "/train/status",
    }


@app.get("/train/status", response_model=TrainingStatusResponse)
async def get_training_status():
    """Poll training progress. Status: idle | running | completed | failed."""
    with training_lock:
        return TrainingStatusResponse(**training_status)
