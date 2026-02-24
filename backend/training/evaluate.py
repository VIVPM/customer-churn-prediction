"""
Model Evaluation Module
=======================
Evaluates the best trained model using classification report
and confusion matrix, matching the Part 2 notebook approach.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.metrics import classification_report, confusion_matrix

from config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, FIGURES_DIR
from backend.training.utils import load_model, load_dataframe, create_directories, print_separator


def load_test_data():
    """Load preprocessed test data (feature-selected if available)."""
    print_separator("LOADING TEST DATA")

    # Prefer feature-selected data, fall back to raw processed
    X_test_path = DATA_PROCESSED / 'X_test_selected.csv'
    if not X_test_path.exists():
        X_test_path = DATA_PROCESSED / 'X_test.csv'

    X_test = load_dataframe(X_test_path)
    y_test = load_dataframe(DATA_PROCESSED / 'y_test.csv').values.ravel()

    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_test, y_test


def load_trained_model():
    """Load the trained best model and scaler."""
    print_separator("LOADING MODEL AND SCALER")

    best_model = load_model(MODELS_DIR / 'best_model.joblib')
    scaler = load_model(MODELS_DIR / 'scaler.joblib')

    # Load model name
    model_name = "unknown"
    name_file = MODELS_DIR / 'best_model_name.txt'
    if name_file.exists():
        model_name = name_file.read_text().strip()

    print(f"Loaded model: {model_name}")

    return best_model, scaler, model_name


def evaluate_model(best_model, scaler, X_test, y_test, model_name):
    """
    Evaluate the model using classification report and confusion matrix.
    Matches notebook cells 15-16.
    """
    print_separator("MODEL EVALUATION")

    # Scale test data
    X_test_scaled = scaler.transform(X_test)

    # Make predictions
    y_pred = best_model.predict(X_test_scaled)

    # Classification report (matching notebook cell 15)
    print(f"Classification Report for the Best Model ({model_name}):")
    report_str = classification_report(y_test, y_pred)
    print(report_str)

    # Classification report as dict for saving
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    return y_pred, report_df, report_str


def plot_confusion_matrix(y_test, y_pred, model_name):
    """
    Plot confusion matrix heatmap (matching notebook cell 16).
    """
    print_separator("CONFUSION MATRIX")

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'confusion_matrix.png', dpi=150)
    plt.close()

    print(f"Confusion Matrix:\n{cm}")
    print(f"\nSaved to: {FIGURES_DIR / 'confusion_matrix.png'}")

    return cm


def evaluate_models():
    """Run the complete evaluation pipeline."""
    create_directories(REPORTS_DIR)
    create_directories(FIGURES_DIR)

    # Load test data and model
    X_test, y_test = load_test_data()
    best_model, scaler, model_name = load_trained_model()

    # Evaluate
    y_pred, report_df, report_str = evaluate_model(
        best_model, scaler, X_test, y_test, model_name
    )

    # Plot confusion matrix
    cm = plot_confusion_matrix(y_test, y_pred, model_name)

    # Save report
    report_df.to_csv(REPORTS_DIR / 'classification_report.csv')
    print(f"\nSaved classification report to: {REPORTS_DIR / 'classification_report.csv'}")

    # Save summary
    with open(REPORTS_DIR / 'evaluation_summary.txt', 'w') as f:
        f.write(f"Best Model: {model_name}\n")
        f.write(f"{'='*50}\n")
        f.write(f"Classification Report:\n{report_str}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
    print(f"Saved evaluation summary to: {REPORTS_DIR / 'evaluation_summary.txt'}")

    print_separator("EVALUATION COMPLETE")

    return report_df


if __name__ == "__main__":
    evaluate_models()
