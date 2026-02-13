"""
Configuration file for Customer Churn Prediction project.
Contains all paths, constants, and settings.
"""

from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
ROOT_DIR = Path(__file__).parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# =============================================================================
# DATA FILE SETTINGS
# =============================================================================
RAW_DATA_FILE = "Customer_Churn_Data_Large.xlsx"

# Sheet names in the Excel file
SHEET_TRANSACTION_HISTORY = "Transaction_History"
SHEET_CUSTOMER_SERVICE = "Customer_Service"
SHEET_ONLINE_ACTIVITY = "Online_Activity"
SHEET_CHURN_STATUS = "Churn_Status"

# =============================================================================
# FEATURE SETTINGS
# =============================================================================
TARGET_COLUMN = "ChurnStatus"
CUSTOMER_ID_COLUMN = "CustomerID"

# Reference date for computing recency features
REFERENCE_DATE = "2023-12-08"

# Z-score threshold for outlier detection
ZSCORE_THRESHOLD = 3

# Correlation threshold for feature selection
CORRELATION_THRESHOLD = 0.8

# =============================================================================
# MODEL SETTINGS
# =============================================================================
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Columns to compute z-scores for
ZSCORE_COLUMNS = [
    'CustomerID', 'TransactionID', 'AmountSpent', 'InteractionID',
    'LoginFrequency', 'DaysSinceLastTransaction', 'DaysSinceLastInteraction',
    'DaysSinceLastLogin', 'TransactionMonth'
]

# Columns to drop after feature selection (high correlation)
HIGH_CORRELATION_COLUMNS = [
    'DaysSinceLastInteraction', 'DaysSinceLastLogin',
    'DaysSinceLastTransaction', 'TransactionMonth'
]
