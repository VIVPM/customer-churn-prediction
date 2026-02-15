"""
Feature Engineering Module

Handles feature engineering and selection:
- Correlation-based feature selection
- Removal of highly correlated features
- Feature importance analysis

Based on the original notebook implementation (cells 57-59).
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from config import (
    DATA_PROCESSED, CORRELATION_THRESHOLD, HIGH_CORRELATION_COLUMNS
)
from src.utils import load_dataframe, save_dataframe, print_separator


def find_correlated_features(df, threshold):
    """
    Find highly correlated features.
    Based on notebook cell 57.
    
    Args:
        df: DataFrame with features
        threshold: Correlation threshold
        
    Returns:
        set: Column names with correlation above threshold
    """
    col_set = set()
    corr_matrix = df.corr()
    
    for i in range(len(df.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_set.add(colname)
    
    return col_set


def remove_correlated_features(X, threshold=None):
    """
    Remove highly correlated features from the dataset.
    Based on notebook cells 58-59.
    
    Args:
        X: Feature DataFrame
        threshold: Correlation threshold (default from config)
        
    Returns:
        pd.DataFrame: DataFrame with correlated features removed
    """
    if threshold is None:
        threshold = CORRELATION_THRESHOLD
    
    print(f"Finding features with correlation > {threshold}...")
    
    # Find correlated features
    corr_features = find_correlated_features(X, threshold)
    
    if corr_features:
        print(f"Highly correlated features found: {corr_features}")
    else:
        print("No highly correlated features found.")
    
    # Remove predefined high correlation columns
    cols_to_drop = [c for c in HIGH_CORRELATION_COLUMNS if c in X.columns]
    
    if cols_to_drop:
        print(f"Removing columns: {cols_to_drop}")
        X = X.drop(columns=cols_to_drop, errors='ignore')
    
    return X


def analyze_feature_correlations(X):
    """
    Analyze and print feature correlation statistics.
    """
    print_separator("FEATURE CORRELATION ANALYSIS")
    
    corr_matrix = X.corr()
    
    # Find top correlations
    correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            correlations.append({
                'Feature 1': corr_matrix.columns[i],
                'Feature 2': corr_matrix.columns[j],
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
    Apply feature selection to training and test sets.
    
    Args:
        X_train: Training features
        X_test: Test features
        
    Returns:
        tuple: (X_train_selected, X_test_selected)
    """
    print_separator("FEATURE SELECTION")
    
    print(f"Features before selection: {X_train.shape[1]}")
    
    # Analyze correlations
    analyze_feature_correlations(X_train)
    
    # Remove correlated features
    X_train_selected = remove_correlated_features(X_train.copy())
    
    # Apply same selection to test set
    cols_to_keep = X_train_selected.columns.tolist()
    X_test_selected = X_test[cols_to_keep].copy()
    
    print(f"\nFeatures after selection: {X_train_selected.shape[1]}")
    print(f"Remaining features: {X_train_selected.columns.tolist()}")
    
    return X_train_selected, X_test_selected


def run_feature_engineering():
    """
    Run the complete feature engineering pipeline.
    """
    # Load processed data
    print_separator("LOADING PROCESSED DATA")
    X_train = load_dataframe(DATA_PROCESSED / "X_train.csv")
    X_test = load_dataframe(DATA_PROCESSED / "X_test.csv")
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # Apply feature selection
    X_train_selected, X_test_selected = select_features(X_train, X_test)
    
    # Save selected features
    print_separator("SAVING SELECTED FEATURES")
    save_dataframe(X_train_selected, DATA_PROCESSED / "X_train_selected.csv")
    save_dataframe(X_test_selected, DATA_PROCESSED / "X_test_selected.csv")
    
    # Update feature names
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
