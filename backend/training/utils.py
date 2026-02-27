"""
Utility functions for the Customer Churn Prediction project.
"""

import os
import joblib
import pandas as pd


def save_model(model, filepath):
    """Save a trained model to disk."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """Load a trained model from disk."""
    return joblib.load(filepath)


def create_directories(*dirs):
    """Create directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def save_dataframe(df, filepath, index=False):
    """Save a DataFrame to CSV."""
    df.to_csv(filepath, index=index)
    print(f"DataFrame saved to {filepath}")


def load_dataframe(filepath):
    """Load a DataFrame from CSV."""
    return pd.read_csv(filepath)


def print_separator(title=""):
    """Print a visual separator for console output."""
    print("\n" + "=" * 60)
    if title:
        print(title)
        print("=" * 60)
