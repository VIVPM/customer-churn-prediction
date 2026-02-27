# Preprocessing pipeline — cleans the merged DataFrame, removes outliers,
# one-hot encodes categoricals, and splits into train/test.
#
# Run this after data_loader.py. Outputs CSVs land in data/processed/.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    DATA_PROCESSED, TARGET_COLUMN, ZSCORE_THRESHOLD,
    ZSCORE_COLUMNS, TEST_SIZE, RANDOM_STATE
)
from backend.training.utils import create_directories, save_dataframe, print_separator
from backend.training.data_loader import load_data


def handle_missing_values(df):
    """
    Fill NaNs per column type:
    - InteractionID → mean (it's numeric but acts like an ID)
    - Dates → forward fill (assume last known value carries forward)
    - Categoricals → mode
    - Everything else numeric → median
    """
    df = df.copy()

    print("Handling missing values...")
    print(f"Missing values before: {df.isna().sum().sum()}")

    if 'InteractionID' in df.columns and df['InteractionID'].isna().any():
        df['InteractionID'] = df['InteractionID'].fillna(int(df['InteractionID'].mean()))

    if 'InteractionDate' in df.columns and df['InteractionDate'].isna().any():
        df['InteractionDate'] = df['InteractionDate'].ffill()

    if 'InteractionType' in df.columns and df['InteractionType'].isna().any():
        df['InteractionType'] = df['InteractionType'].fillna(df['InteractionType'].mode().iloc[0])

    if 'ResolutionStatus' in df.columns and df['ResolutionStatus'].isna().any():
        df['ResolutionStatus'] = df['ResolutionStatus'].fillna(df['ResolutionStatus'].mode().iloc[0])

    if 'DaysSinceLastInteraction' in df.columns and df['DaysSinceLastInteraction'].isna().any():
        df['DaysSinceLastInteraction'] = df['DaysSinceLastInteraction'].ffill()

    # Catch-all for any remaining numeric/categorical NaNs
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=['object']).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    print(f"Missing values after: {df.isna().sum().sum()}")
    return df


def compute_zscores(df):
    """
    Add z-score columns for the numeric features listed in ZSCORE_COLUMNS.
    Skips columns with zero variance (std=0) to avoid NaN z-scores.
    """
    df = df.copy()
    print("Computing z-scores...")

    zscore_cols = []
    for col in ZSCORE_COLUMNS:
        if col in df.columns:
            col_std = df[col].std()
            if col_std == 0:
                print(f"  Skipping {col} (zero variance)")
                continue
            zscore_col = f'zscore{col}'
            df[zscore_col] = (df[col] - df[col].mean()) / col_std
            zscore_cols.append(zscore_col)
            print(f"  Computed z-score for: {col}")

    return df, zscore_cols


def remove_outliers(df, zscore_cols):
    """
    Drop rows where any z-score column exceeds ZSCORE_THRESHOLD (|z| > 3).
    Drops the z-score helper columns afterwards — they're noise for training.
    """
    print(f"\nRemoving outliers (z-score threshold: {ZSCORE_THRESHOLD})...")
    print(f"Rows before: {len(df)}")

    mask = pd.Series([True] * len(df), index=df.index)
    for col in zscore_cols:
        if col in df.columns:
            mask = mask & (df[col].abs() < ZSCORE_THRESHOLD)

    df_clean = df[mask].copy()
    print(f"Rows after: {len(df_clean)}")
    print(f"Rows removed: {len(df) - len(df_clean)}")

    df_clean = df_clean.drop(columns=zscore_cols, errors='ignore')
    return df_clean


def encode_categorical_features(df):
    """
    Drop raw date columns (already extracted into month/year/days-since features),
    then one-hot encode whatever object-typed columns remain.
    drop_first=True to avoid multicollinearity.
    """
    print("\nEncoding categorical features...")

    date_cols = ['TransactionDate', 'InteractionDate', 'LastLoginDate']
    df = df.drop(columns=[c for c in date_cols if c in df.columns], errors='ignore')

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns to encode: {cat_cols}")

    df_encoded = pd.get_dummies(df, drop_first=True)
    print(f"Columns after encoding: {len(df_encoded.columns)}")
    return df_encoded


def split_features_target(df):
    """Separate features from the target column. Nothing fancy."""
    print("\nSplitting features and target...")

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")
    return X, y


def preprocess_data():
    """
    Full preprocessing pipeline. Saves train/test CSVs to data/processed/.
    Returns (X_train, X_test, y_train, y_test).
    """
    create_directories(DATA_PROCESSED)

    print_separator("LOADING DATA")
    df = load_data()

    print_separator("HANDLING MISSING VALUES")
    df = handle_missing_values(df)

    print_separator("OUTLIER DETECTION & REMOVAL")
    df, zscore_cols = compute_zscores(df)
    df = remove_outliers(df, zscore_cols)

    print_separator("ENCODING FEATURES")
    df = encode_categorical_features(df)

    print_separator("SPLITTING DATA")
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    print_separator("SAVING PROCESSED DATA")
    save_dataframe(X_train,            DATA_PROCESSED / "X_train.csv")
    save_dataframe(X_test,             DATA_PROCESSED / "X_test.csv")
    save_dataframe(y_train.to_frame(), DATA_PROCESSED / "y_train.csv")
    save_dataframe(y_test.to_frame(),  DATA_PROCESSED / "y_test.csv")

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
