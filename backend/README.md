# Backend - Churn Prediction API

FastAPI backend that serves the trained churn prediction model.

## Local Development

```bash
cd backend
uvicorn api:app --reload --port 8000
```

API docs: http://localhost:8000/docs

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Single customer prediction |
| POST | `/predict/batch` | Batch prediction (CSV upload) |
| GET | `/model/info` | Model metadata |

## Deploy to Render

1. Push to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect repo → Set **Root Directory** to `backend`
4. Render auto-detects `render.yaml`
5. You'll get a URL like `https://churn-api-xxxx.onrender.com`

## Updating Models

After retraining, copy the new model files:
```bash
copy ..\models\best_model.joblib models\
copy ..\models\scaler.joblib models\
copy ..\models\best_model_name.txt models\
copy ..\data\processed\feature_names_selected.csv data\
```
