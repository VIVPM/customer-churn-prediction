"""
Data Preprocessing Module

Handles data cleaning and preprocessing:
- Missing value imputation
- Z-score computation for outlier detection
- Outlier removal
- One-hot encoding
- Train/test split

Based on the original notebook implementation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from config import (
    DATA_PROCESSED, TARGET_COLUMN, ZSCORE_THRESHOLD,
    ZSCORE_COLUMNS, TEST_SIZE, RANDOM_STATE
)
from src.utils import create_directories, save_dataframe, print_separator
from src.data_loader import load_data


def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    
    - Numeric columns: fill with mean
    - Categorical columns: fill with mode
    - Date columns: forward fill
    
    Based on notebook cell 17.
    """
    df = df.copy()
    
    print("Handling missing values...")
    print(f"Missing values before: {df.isna().sum().sum()}")
    
    # Handle InteractionID (numeric) - fill with mean
    if 'InteractionID' in df.columns and df['InteractionID'].isna().any():
        df['InteractionID'] = df['InteractionID'].fillna(int(df['InteractionID'].mean()))
    
    # Handle InteractionDate - forward fill
    if 'InteractionDate' in df.columns and df['InteractionDate'].isna().any():
        df['InteractionDate'] = df['InteractionDate'].ffill()
    
    # Handle InteractionType (categorical) - fill with mode
    if 'InteractionType' in df.columns and df['InteractionType'].isna().any():
        df['InteractionType'] = df['InteractionType'].fillna(df['InteractionType'].mode().iloc[0])
    
    # Handle ResolutionStatus (categorical) - fill with mode
    if 'ResolutionStatus' in df.columns and df['ResolutionStatus'].isna().any():
        df['ResolutionStatus'] = df['ResolutionStatus'].fillna(df['ResolutionStatus'].mode().iloc[0])
    
    # Handle DaysSinceLastInteraction - forward fill
    if 'DaysSinceLastInteraction' in df.columns and df['DaysSinceLastInteraction'].isna().any():
        df['DaysSinceLastInteraction'] = df['DaysSinceLastInteraction'].ffill()
    
    # Handle any remaining numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Handle any remaining categorical columns with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    
    print(f"Missing values after: {df.isna().sum().sum()}")
    
    return df


def compute_zscores(df):
    """
    Compute z-scores for numerical columns.
    Based on notebook cell 27.
    """
    df = df.copy()
    
    print("Computing z-scores...")
    
    zscore_cols = []
    for col in ZSCORE_COLUMNS:
        if col in df.columns:
            zscore_col = f'zscore{col}'
            df[zscore_col] = (df[col] - df[col].mean()) / df[col].std()
            zscore_cols.append(zscore_col)
            print(f"  Computed z-score for: {col}")
    
    return df, zscore_cols


def remove_outliers(df, zscore_cols):
    """
    Remove outliers using z-score threshold.
    Based on notebook cell 29.
    """
    print(f"\nRemoving outliers (z-score threshold: {ZSCORE_THRESHOLD})...")
    print(f"Rows before: {len(df)}")
    
    # Create a mask for all z-score conditions
    mask = pd.Series([True] * len(df), index=df.index)
    
    for col in zscore_cols:
        if col in df.columns:
            mask = mask & (df[col].abs() < ZSCORE_THRESHOLD)
    
    df_clean = df[mask].copy()
    
    print(f"Rows after: {len(df_clean)}")
    print(f"Rows removed: {len(df) - len(df_clean)}")
    
    # Drop z-score columns
    df_clean = df_clean.drop(columns=zscore_cols, errors='ignore')
    
    return df_clean


def encode_categorical_features(df):
    """
    One-hot encode categorical features.
    Based on notebook cell 53.
    """
    print("\nEncoding categorical features...")
    
    # Drop date columns before encoding
    date_cols = ['TransactionDate', 'InteractionDate', 'LastLoginDate']
    df = df.drop(columns=[c for c in date_cols if c in df.columns], errors='ignore')
    
    # Get list of categorical columns before encoding
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns to encode: {cat_cols}")
    
    # One-hot encode
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    print(f"Columns after encoding: {len(df_encoded.columns)}")
    
    return df_encoded


def split_features_target(df):
    """
    Split dataframe into features (X) and target (y).
    Based on notebook cell 56.
    """
    print("\nSplitting features and target...")
    
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")
    
    return X, y


def preprocess_data():
    """
    Run the complete preprocessing pipeline.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    create_directories(DATA_PROCESSED)
    
    # Load data
    print_separator("LOADING DATA")
    df = load_data()
    
    # Handle missing values
    print_separator("HANDLING MISSING VALUES")
    df = handle_missing_values(df)
    
    # Compute z-scores and remove outliers
    print_separator("OUTLIER DETECTION & REMOVAL")
    df, zscore_cols = compute_zscores(df)
    df = remove_outliers(df, zscore_cols)
    
    # Encode categorical features
    print_separator("ENCODING FEATURES")
    df = encode_categorical_features(df)
    
    # Split features and target
    print_separator("SPLITTING DATA")
    X, y = split_features_target(df)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Save processed data
    print_separator("SAVING PROCESSED DATA")
    save_dataframe(X_train, DATA_PROCESSED / "X_train.csv")
    save_dataframe(X_test, DATA_PROCESSED / "X_test.csv")
    save_dataframe(y_train.to_frame(), DATA_PROCESSED / "y_train.csv")
    save_dataframe(y_test.to_frame(), DATA_PROCESSED / "y_test.csv")
    
    # Save feature names for later use
    feature_names = pd.DataFrame({'feature': X_train.columns})
    save_dataframe(feature_names, DATA_PROCESSED / "feature_names.csv")
    
    print_separator("PREPROCESSING COMPLETE")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()
    
    print("\n=== Final Data Summary ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"\nFeature columns:")
    print(X_train.columns.tolist())
