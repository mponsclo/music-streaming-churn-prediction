"""
Central path configuration for the KKBox churn prediction project.
All data file references go through this module so the messy
raw directory names (with spaces) are isolated to one place.
"""

from pathlib import Path

# Project root (two levels up from config/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Raw data paths (v2 refresh files = canonical dataset)
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_PATH = DATA_DIR / "data 2" / "churn_comp_refresh" / "train_v2.csv"
MEMBERS_PATH = DATA_DIR / "members_v3.csv"
TRANSACTIONS_PATH = DATA_DIR / "data 3" / "churn_comp_refresh" / "transactions_v2.csv"
USER_LOGS_PATH = DATA_DIR / "data 4" / "churn_comp_refresh" / "user_logs_v2.csv"

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
FEATURE_TABLE_PATH = OUTPUT_DIR / "feature_table.parquet"
