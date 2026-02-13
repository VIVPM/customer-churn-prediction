# Customer Churn Prediction

A machine learning project to predict customer churn using multi-source data analysis. Built with clean, modular Python scripts.

## Overview

This project predicts customer churn by analyzing data from multiple sources:
- **Transaction History**: Customer purchase behavior
- **Customer Service**: Support interactions and resolution status
- **Online Activity**: Login frequency and engagement
- **Churn Status**: Target variable

The pipeline demonstrates:
- Multi-source data merging and integration
- Comprehensive EDA with visualizations
- Z-score based outlier detection
- Correlation-based feature selection
- Multiple ML model training and comparison
- SHAP-based model explainability
- Production-ready prediction pipeline

## Project Structure

```
customer-churn-prediction/
├── data/
│   ├── raw/                 # Place your Excel file here
│   └── processed/           # Generated train/test splits
├── models/                  # Saved trained models
├── reports/
│   └── figures/             # Generated visualizations
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Load and merge Excel sheets
│   ├── eda.py               # Exploratory data analysis
│   ├── preprocessing.py     # Data cleaning and encoding
│   ├── feature_engineering.py # Feature selection
│   ├── train.py             # Model training
│   ├── evaluate.py          # Model evaluation & SHAP
│   ├── predict.py           # Make predictions
│   └── utils.py             # Helper functions
├── main.py                  # Pipeline runner
├── config.py                # Configuration settings
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

This project expects an Excel file with 4 sheets:

| Sheet Name | Description | Key Columns |
|------------|-------------|-------------|
| Transaction_History | Purchase records | CustomerID, TransactionDate, AmountSpent, ProductCategory |
| Customer_Service | Support interactions | CustomerID, InteractionDate, InteractionType, ResolutionStatus |
| Online_Activity | Digital engagement | CustomerID, LastLoginDate, LoginFrequency, ServiceUsage |
| Churn_Status | Target variable | CustomerID, ChurnStatus |

Place your Excel file as `data/raw/Customer_Churn_Data_Large.xlsx`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place your data file in `data/raw/`

## Usage

### Run Complete Pipeline

```bash
python main.py --all
```

This runs: EDA → Preprocessing → Feature Engineering → Training → Evaluation

### Run Individual Steps

```bash
# Exploratory Data Analysis
python main.py --eda

# Data Preprocessing
python main.py --preprocess

# Feature Engineering
python main.py --features

# Model Training
python main.py --train

# Model Evaluation
python main.py --evaluate

# Interactive Prediction
python main.py --predict
```

### Run Individual Modules

```bash
python src/data_loader.py
python src/eda.py
python src/preprocessing.py
python src/feature_engineering.py
python src/train.py
python src/evaluate.py
python src/predict.py
```

## Pipeline Details

### 1. Data Loading (`data_loader.py`)
- Loads 4 Excel sheets
- Merges on CustomerID
- Computes recency features (DaysSinceLastTransaction, etc.)
- Extracts date components (TransactionMonth, TransactionYear)

### 2. EDA (`eda.py`)
- Basic statistics and missing value analysis
- Correlation heatmap
- Distribution plots for numerical features
- Churn rate by categorical features
- Time series analysis of transactions/interactions

### 3. Preprocessing (`preprocessing.py`)
- Missing value imputation (mean/mode/forward-fill)
- Z-score computation for outlier detection
- Outlier removal (|z-score| > 3)
- One-hot encoding of categorical features
- Train/test split with stratification

### 4. Feature Engineering (`feature_engineering.py`)
- Correlation analysis
- Removal of highly correlated features (r > 0.8)
- Feature selection based on domain knowledge

### 5. Training (`train.py`)
- SMOTE for class imbalance
- Models trained:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost
- 5-fold cross-validation
- Model persistence with joblib

### 6. Evaluation (`evaluate.py`)
- Classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- ROC curves comparison
- Confusion matrices
- Feature importance plots
- SHAP analysis for model interpretability

### 7. Prediction (`predict.py`)
- Single customer prediction
- Batch prediction from CSV
- Risk level classification (Low/Medium/High)
- Retention recommendations

## Models

| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear model with class balancing |
| Random Forest | Ensemble of 100 decision trees |
| Gradient Boosting | Sequential boosting with max_depth=5 |
| XGBoost | Optimized gradient boosting |

## Configuration

Edit `config.py` to customize:

```python
# Reference date for recency features
REFERENCE_DATE = "2023-12-08"

# Z-score threshold for outlier detection
ZSCORE_THRESHOLD = 3

# Correlation threshold for feature selection
CORRELATION_THRESHOLD = 0.8

# Model settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
```

## Generated Outputs

After running the pipeline:

### Reports (`reports/`)
- `model_comparison.csv` - Performance metrics for all models

### Figures (`reports/figures/`)
- `churn_distribution.png` - Target variable distribution
- `correlation_heatmap.png` - Feature correlations
- `amount_spent_distribution.png` - Spending distribution
- `churn_by_*.png` - Churn rate by categorical features
- `roc_curves.png` - ROC curves for all models
- `confusion_matrix_*.png` - Confusion matrices
- `feature_importance_*.png` - Feature importance plots
- `shap_summary_*.png` - SHAP analysis

### Models (`models/`)
- `logistic_regression.joblib`
- `random_forest.joblib`
- `gradient_boosting.joblib`
- `xgboost.joblib`
- `cv_comparison.csv` - Cross-validation results

## Key Findings

From EDA:
- Unresolved customer service issues correlate with higher churn
- Lower login frequency indicates higher churn risk
- Certain product categories have higher churn rates
- Recency of last transaction is a strong predictor

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- imbalanced-learn
- shap
- matplotlib
- seaborn
- openpyxl (for Excel files)

## Future Improvements

- [ ] Add hyperparameter tuning with Optuna
- [ ] Implement more feature engineering techniques
- [ ] Add time-series features
- [ ] Create REST API for predictions
- [ ] Add unit tests
- [ ] Containerize with Docker
- [ ] Add MLflow for experiment tracking

## License

MIT License

## Author

Your Name - [GitHub](https://github.com/yourusername)
