# Correlation-based feature selection.
# Drops columns that are highly correlated with each other to reduce
# multicollinearity before training. Thresholds and specific columns
# to drop are defined in config.py.

import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    DATA_PROCESSED, CORRELATION_THRESHOLD, HIGH_CORRELATION_COLUMNS
)
from backend.training.utils import load_dataframe, save_dataframe, print_separator


def find_correlated_features(df, threshold):
    """
    Walk the upper triangle of the correlation matrix and collect any column
    that has |r| > threshold with at least one earlier column.
    Returns a set of column names to consider dropping.
    """
    col_set = set()
    corr_matrix = df.corr()

    for i in range(len(df.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                col_set.add(corr_matrix.columns[i])

    return col_set


def remove_correlated_features(X, threshold=None):
    """
    Drop columns listed in HIGH_CORRELATION_COLUMNS (config.py).
    find_correlated_features() is called to log what's correlated,
    but the actual drops use the explicit list from config for reproducibility.
    """
    if threshold is None:
        threshold = CORRELATION_THRESHOLD

    print(f"Finding features with correlation > {threshold}...")

    corr_features = find_correlated_features(X, threshold)
    if corr_features:
        print(f"Highly correlated features found: {corr_features}")
    else:
        print("No highly correlated features found.")

    cols_to_drop = [c for c in HIGH_CORRELATION_COLUMNS if c in X.columns]
    if cols_to_drop:
        print(f"Removing columns: {cols_to_drop}")
        X = X.drop(columns=cols_to_drop, errors='ignore')

    return X


def analyze_feature_correlations(X):
    """Print the top 10 pairwise correlations — useful to sanity-check before dropping."""
    print_separator("FEATURE CORRELATION ANALYSIS")

    corr_matrix = X.corr()
    correlations = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            correlations.append({
                'Feature 1':   corr_matrix.columns[i],
                'Feature 2':   corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })

    corr_df = pd.DataFrame(correlations)
    corr_df['Abs Correlation'] = corr_df['Correlation'].abs()
    corr_df = corr_df.sort_values('Abs Correlation', ascending=False)

    print("\nTop 10 Feature Correlations:")
    print(corr_df.head(10).to_string(index=False))
    return corr_df


def select_features(X_train, X_test):
    """
    Drop correlated features from train, then apply the same column list to test.
    Never fit any selection logic on test data.
    """
    print_separator("FEATURE SELECTION")
    print(f"Features before selection: {X_train.shape[1]}")

    analyze_feature_correlations(X_train)

    X_train_selected = remove_correlated_features(X_train.copy())

    # Use train's column list to filter test — keeps shapes in sync
    cols_to_keep    = X_train_selected.columns.tolist()
    X_test_selected = X_test[cols_to_keep].copy()

    print(f"\nFeatures after selection: {X_train_selected.shape[1]}")
    print(f"Remaining features: {X_train_selected.columns.tolist()}")
    return X_train_selected, X_test_selected


def run_feature_engineering():
    """Load processed splits, run feature selection, save results."""
    print_separator("LOADING PROCESSED DATA")
    X_train = load_dataframe(DATA_PROCESSED / "X_train.csv")
    X_test  = load_dataframe(DATA_PROCESSED / "X_test.csv")

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    X_train_selected, X_test_selected = select_features(X_train, X_test)

    print_separator("SAVING SELECTED FEATURES")
    save_dataframe(X_train_selected, DATA_PROCESSED / "X_train_selected.csv")
    save_dataframe(X_test_selected,  DATA_PROCESSED / "X_test_selected.csv")

    feature_names = pd.DataFrame({'feature': X_train_selected.columns})
    save_dataframe(feature_names, DATA_PROCESSED / "feature_names_selected.csv")

    print_separator("FEATURE ENGINEERING COMPLETE")
    return X_train_selected, X_test_selected


if __name__ == "__main__":
    X_train, X_test = run_feature_engineering()

    print("\n=== Final Feature Summary ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"\nSelected features:")
    for col in X_train.columns:
        print(f"  - {col}")
