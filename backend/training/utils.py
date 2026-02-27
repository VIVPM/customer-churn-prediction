# Small helpers shared across the pipeline.
# Nothing clever here — just I/O wrappers so the rest of the code stays clean.

import os
import joblib
import pandas as pd


def save_model(model, filepath):
    """Persist a trained model with joblib."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """Load a joblib model from disk."""
    return joblib.load(filepath)


def create_directories(*dirs):
    """Make sure output directories exist before anything tries to write to them."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def save_dataframe(df, filepath, index=False):
    """Write a DataFrame to CSV."""
    df.to_csv(filepath, index=index)
    print(f"DataFrame saved to {filepath}")


def load_dataframe(filepath):
    """Read a CSV into a DataFrame."""
    return pd.read_csv(filepath)


def print_separator(title=""):
    """Console divider — makes long pipeline output easier to scan."""
    print("\n" + "=" * 60)
    if title:
        print(title)
        print("=" * 60)
