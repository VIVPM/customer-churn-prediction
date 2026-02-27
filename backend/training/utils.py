# load/save helpers — keeps the rest of the code from caring about file formats

import os
import joblib
import pandas as pd


def save_model(model, filepath):
    """Dump model to disk with joblib. Prints path so you know where it went."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """Load a joblib model. No existence check — let joblib raise a clear error."""
    return joblib.load(filepath)


def create_directories(*dirs):
    """Make sure output directories exist before anything tries to write to them."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def save_dataframe(df, filepath, index=False):
    """Save DataFrame to CSV. index=False by default — row numbers are noise."""
    df.to_csv(filepath, index=index)
    print(f"DataFrame saved to {filepath}")


def load_dataframe(filepath):
    """Read a CSV into a DataFrame. Straightforward."""
    return pd.read_csv(filepath)


def print_separator(title=""):
    """Visual break in console output — makes long pipeline logs readable."""
    print("\n" + "=" * 60)
    if title:
        print(title)
        print("=" * 60)
