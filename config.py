from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

# File names
RAW_DATA_FILE = "telco_churn.csv"

# Model settings
TARGET_COLUMN = "Churn"
TEST_SIZE = 0.2
RANDOM_STATE = 42