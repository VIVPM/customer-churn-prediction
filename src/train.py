"""
Model Training Module

Trains multiple machine learning models for churn prediction:
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

Handles class imbalance using SMOTE and class weights.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Skipping XGBoost model.")

from config import DATA_PROCESSED, MODELS_DIR, RANDOM_STATE
from src.utils import save_model, load_dataframe, create_directories, print_separator


def load_training_data():
    """Load processed training data."""
    # Try to load selected features first, fall back to full features
    try:
        X_train = load_dataframe(DATA_PROCESSED / "X_train_selected.csv")
        print("Loaded selected features.")
    except FileNotFoundError:
        X_train = load_dataframe(DATA_PROCESSED / "X_train.csv")
        print("Loaded full features.")
    
    y_train = load_dataframe(DATA_PROCESSED / "y_train.csv").values.ravel()
    
    return X_train, y_train


def get_models():
    """
    Get dictionary of models to train.
    
    Returns:
        dict: Model name -> model instance
    """
    models = {
        'logistic_regression': LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver='lbfgs'
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            max_depth=5
        )
    }
    
    if XGBOOST_AVAILABLE:
        models['xgboost'] = XGBClassifier(
            n_estimators=100,
            scale_pos_weight=3,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )
    
    return models


def apply_smote(X_train, y_train):
    """
    Apply SMOTE to handle class imbalance.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        tuple: (X_balanced, y_balanced)
    """
    print("Applying SMOTE for class balancing...")
    print(f"Original class distribution: {np.bincount(y_train.astype(int))}")
    
    smote = SMOTE(random_state=RANDOM_STATE)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"Balanced class distribution: {np.bincount(y_balanced.astype(int))}")
    print(f"Samples after SMOTE: {len(X_balanced)}")
    
    return X_balanced, y_balanced


def train_single_model(model, X_train, y_train, model_name):
    """
    Train a single model with cross-validation.
    
    Args:
        model: Model instance
        X_train: Training features
        y_train: Training labels
        model_name: Name of the model
        
    Returns:
        dict: Training results
    """
    print(f"\nTraining {model_name}...")
    
    # Cross-validation scores
    cv_scores_f1 = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    cv_scores_accuracy = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores_roc_auc = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    
    # Fit the model
    model.fit(X_train, y_train)
    
    results = {
        'model': model,
        'cv_f1_mean': cv_scores_f1.mean(),
        'cv_f1_std': cv_scores_f1.std(),
        'cv_accuracy_mean': cv_scores_accuracy.mean(),
        'cv_accuracy_std': cv_scores_accuracy.std(),
        'cv_roc_auc_mean': cv_scores_roc_auc.mean(),
        'cv_roc_auc_std': cv_scores_roc_auc.std()
    }
    
    print(f"  CV Accuracy: {cv_scores_accuracy.mean():.4f} (+/- {cv_scores_accuracy.std():.4f})")
    print(f"  CV F1 Score: {cv_scores_f1.mean():.4f} (+/- {cv_scores_f1.std():.4f})")
    print(f"  CV ROC-AUC:  {cv_scores_roc_auc.mean():.4f} (+/- {cv_scores_roc_auc.std():.4f})")
    
    return results


def train_models():
    """
    Train all models and save them.
    
    Returns:
        dict: Results for all models
    """
    create_directories(MODELS_DIR)
    
    # Load training data
    print_separator("LOADING TRAINING DATA")
    X_train, y_train = load_training_data()
    print(f"Training data shape: {X_train.shape}")
    print(f"Target distribution: {np.bincount(y_train.astype(int))}")
    
    # Apply SMOTE
    print_separator("HANDLING CLASS IMBALANCE")
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)
    
    # Get models
    models = get_models()
    
    # Train all models
    print_separator("TRAINING MODELS")
    results = {}
    
    for name, model in models.items():
        results[name] = train_single_model(
            model, X_train_balanced, y_train_balanced, name
        )
        
        # Save model
        save_model(model, MODELS_DIR / f"{name}.joblib")
    
    # Find best model
    print_separator("MODEL COMPARISON")
    
    comparison_df = pd.DataFrame({
        name: {
            'CV Accuracy': f"{r['cv_accuracy_mean']:.4f} (+/- {r['cv_accuracy_std']:.4f})",
            'CV F1': f"{r['cv_f1_mean']:.4f} (+/- {r['cv_f1_std']:.4f})",
            'CV ROC-AUC': f"{r['cv_roc_auc_mean']:.4f} (+/- {r['cv_roc_auc_std']:.4f})"
        }
        for name, r in results.items()
    }).T
    
    print(comparison_df)
    
    # Save comparison
    comparison_df.to_csv(MODELS_DIR / "cv_comparison.csv")
    print(f"\nSaved CV comparison to: {MODELS_DIR / 'cv_comparison.csv'}")
    
    # Identify best model
    best_name = max(results, key=lambda x: results[x]['cv_f1_mean'])
    print(f"\nBest model (by F1): {best_name}")
    print(f"  F1 Score: {results[best_name]['cv_f1_mean']:.4f}")
    
    print_separator("TRAINING COMPLETE")
    
    return results


if __name__ == "__main__":
    results = train_models()
