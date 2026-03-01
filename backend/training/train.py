# Trains 4 classifiers (SVM, Random Forest, Logistic Regression, Decision Tree)
# using GridSearchCV 5-fold CV, picks the winner, and saves it.
# HF Hub upload is triggered at the end via api.py's _upload_to_hf.

import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from config import DATA_PROCESSED, MODELS_DIR, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE
from backend.training.utils import save_model, load_dataframe, create_directories, print_separator
from backend.api import _upload_to_hf

# Hyperparameter grids — these match what the notebook found to be reasonable ranges
MODEL_PARAMS = {
    'svm': {
        'model': SVC(gamma='auto', class_weight='balanced'),
        'params': {
            'C': [1, 10, 20],
            'kernel': ['rbf', 'linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(class_weight='balanced'),
        'params': {
            'n_estimators': [50, 60, 70, 80, 90, 100]
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(solver='liblinear', class_weight='balanced'),
        'params': {
            'C': [1, 5, 10],
        }
    },
    'decision_tree': {
        'model': DecisionTreeClassifier(class_weight='balanced'),
        'params': {
            'criterion': ['gini', 'entropy']
        }
    }
}


def load_training_data():
    """
    Prefers feature-selected X_train if it exists (run feature_engineering.py first).
    Falls back to X_train.csv if not — still works, just uses more features.
    """
    print_separator("LOADING TRAINING DATA")

    X_train_path = DATA_PROCESSED / 'X_train_selected.csv'
    if not X_train_path.exists():
        X_train_path = DATA_PROCESSED / 'X_train.csv'
        print("Note: Using pre-selection features (run feature engineering first for best results)")

    X_train = load_dataframe(X_train_path)
    y_train = load_dataframe(DATA_PROCESSED / 'y_train.csv').values.ravel()

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"Class distribution: {pd.Series(y_train).value_counts().to_dict()}")
    return X_train, y_train


def scale_features(X_train):
    """StandardScaler fit on train only. Scaler is returned so we can apply it at prediction time."""
    print_separator("FEATURE SCALING")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    print(f"Scaled training data shape: {X_train_scaled.shape}")
    return X_train_scaled, scaler


def run_gridsearch(X_train_scaled, y_train):
    """
    Run 5-fold GridSearchCV for each model. Returns:
    - scores: list of {model, best_score, best_params} dicts
    - grid_searches: {model_name: fitted GridSearchCV object}
    """
    print_separator("GRIDSEARCH HYPERPARAMETER TUNING")

    scores = []
    grid_searches = {}

    for model_name, mp in MODEL_PARAMS.items():
        print(f"\nTraining {model_name}...")
        clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
        clf.fit(X_train_scaled, y_train)

        scores.append({
            'model':       model_name,
            'best_score':  clf.best_score_,
            'best_params': clf.best_params_
        })
        grid_searches[model_name] = clf

        print(f"  Best Score: {clf.best_score_:.4f}")
        print(f"  Best Params: {clf.best_params_}")

    return scores, grid_searches


def select_best_model(scores, X_train_scaled, y_train):
    """
    Print the comparison table, pick the highest CV score, retrain that model
    with the best params on the full training set.
    """
    print_separator("MODEL COMPARISON")

    scores_df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
    print(scores_df.to_string(index=True))

    best_model_info = max(scores, key=lambda x: x['best_score'])
    best_model_name = best_model_info['model']
    best_params     = best_model_info['best_params']

    print(f"\n{'='*40}")
    print(f"Best Model: {best_model_name}")
    print(f"Best CV Score: {best_model_info['best_score']:.4f}")
    print(f"Best Params: {best_params}")
    print(f"{'='*40}")

    best_model = MODEL_PARAMS[best_model_name]['model']
    best_model.set_params(**best_params)
    best_model.fit(X_train_scaled, y_train)

    return best_model, best_model_name, scores_df


def train_models():
    """Full training pipeline: load → scale → grid search → pick best → save → upload."""
    create_directories(MODELS_DIR)

    X_train, y_train = load_training_data()
    X_train_scaled, scaler = scale_features(X_train)
    scores, grid_searches  = run_gridsearch(X_train_scaled, y_train)
    best_model, best_model_name, scores_df = select_best_model(scores, X_train_scaled, y_train)

    print_separator("SAVING MODELS")
    save_model(scaler, MODELS_DIR / 'scaler.joblib')
    save_model(best_model, MODELS_DIR / 'best_model.joblib')
    scores_df.to_csv(MODELS_DIR / 'model_comparison.csv', index=False)

    # Write best model name to a text file so evaluate.py can load the right one
    with open(MODELS_DIR / 'best_model_name.txt', 'w') as f:
        f.write(best_model_name)

    print(f"Saved scaler to: {MODELS_DIR / 'scaler.joblib'}")
    print(f"Saved best model ({best_model_name}) to: {MODELS_DIR / 'best_model.joblib'}")
    print(f"Saved comparison to: {MODELS_DIR / 'model_comparison.csv'}")

    print_separator("UPLOADING TO HUGGING FACE")
    _upload_to_hf(metrics_df=scores_df)

    print_separator("TRAINING COMPLETE")
    return best_model, scaler, scores_df


if __name__ == "__main__":
    train_models()
