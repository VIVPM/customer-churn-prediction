"""
Data Loader Module

Loads and merges data from multiple Excel sheets:
- Transaction_History
- Customer_Service
- Online_Activity
- Churn_Status

Based on the original notebook implementation.
"""

import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DATA_RAW, RAW_DATA_FILE, REFERENCE_DATE,
    SHEET_TRANSACTION_HISTORY, SHEET_CUSTOMER_SERVICE,
    SHEET_ONLINE_ACTIVITY, SHEET_CHURN_STATUS, CUSTOMER_ID_COLUMN
)


def load_transaction_history():
    """Load Transaction_History sheet."""
    filepath = DATA_RAW / RAW_DATA_FILE
    df = pd.read_excel(filepath, sheet_name=SHEET_TRANSACTION_HISTORY)
    print(f"Transaction_History loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def load_customer_service():
    """Load Customer_Service sheet."""
    filepath = DATA_RAW / RAW_DATA_FILE
    df = pd.read_excel(filepath, sheet_name=SHEET_CUSTOMER_SERVICE)
    print(f"Customer_Service loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def load_online_activity():
    """Load Online_Activity sheet."""
    filepath = DATA_RAW / RAW_DATA_FILE
    df = pd.read_excel(filepath, sheet_name=SHEET_ONLINE_ACTIVITY)
    print(f"Online_Activity loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def load_churn_status():
    """Load Churn_Status sheet."""
    filepath = DATA_RAW / RAW_DATA_FILE
    df = pd.read_excel(filepath, sheet_name=SHEET_CHURN_STATUS)
    print(f"Churn_Status loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def load_and_merge_all_data():
    """
    Load all sheets and merge them on CustomerID.
    Also computes recency features and extracts date components.
    
    Returns:
        pd.DataFrame: Merged DataFrame with all features
    """
    # Load all sheets
    df1 = load_transaction_history()
    df2 = load_customer_service()
    df3 = load_online_activity()
    df4 = load_churn_status()
    
    # Convert dates to datetime
    df1['TransactionDate'] = pd.to_datetime(df1['TransactionDate'])
    df2['InteractionDate'] = pd.to_datetime(df2['InteractionDate'])
    df3['LastLoginDate'] = pd.to_datetime(df3['LastLoginDate'])
    
    # Merge dataframes
    merged_df = pd.merge(df1, df2, on=CUSTOMER_ID_COLUMN, how='outer')
    merged_df = pd.merge(merged_df, df3, on=CUSTOMER_ID_COLUMN, how='outer')
    
    # Compute recency features
    current_date = pd.to_datetime(REFERENCE_DATE)
    merged_df['DaysSinceLastTransaction'] = (current_date - merged_df['TransactionDate']).dt.days
    merged_df['DaysSinceLastInteraction'] = (current_date - merged_df['InteractionDate']).dt.days
    merged_df['DaysSinceLastLogin'] = (current_date - merged_df['LastLoginDate']).dt.days
    
    # Extract month/year from dates
    merged_df['TransactionMonth'] = merged_df['TransactionDate'].dt.month
    merged_df['TransactionYear'] = merged_df['TransactionDate'].dt.year
    merged_df['InteractionMonth'] = merged_df['InteractionDate'].dt.month
    merged_df['InteractionYear'] = merged_df['InteractionDate'].dt.year
    merged_df['LastLoginMonth'] = merged_df['LastLoginDate'].dt.month
    
    # Merge churn status
    merged_df = pd.merge(merged_df, df4, on=CUSTOMER_ID_COLUMN, how='outer')
    
    print(f"\nMerged DataFrame: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
    
    return merged_df


def load_data():
    """
    Main function to load all data.
    Alias for load_and_merge_all_data().
    """
    return load_and_merge_all_data()


if __name__ == "__main__":
    df = load_data()
    print("\n=== Data Sample ===")
    print(df.head())
    print("\n=== Data Info ===")
    print(df.info())
    print("\n=== Data Types ===")
    print(df.dtypes)
