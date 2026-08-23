import os
from pathlib import Path

# Base Directory Resolution
BASE_DIR = Path(__file__).resolve().parent.parent

# Database Connection Settings
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'data_esi_sba_2023')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'root')

# Prediction Settings
SEQ_LEN = 720
WELLS = ["LNB712-CE", "LNB234-CE"]
TABLES = ['dg_values', 'cm_values', 'pt_values', 'pl_values', 'dp_values', 'of_values']

# Preprocessing Constants
MAX_GAP = 40
VERY_LARGE_GAP = 300
FFT_WINDOW = 64

# Model Paths (Portable)
MODELS_DIR = BASE_DIR / "models"
MODEL_PATHS = {
    '6h': str(MODELS_DIR / "pm_transformer_6h.keras"),
    '12h': str(MODELS_DIR / "pm_transformer_12h.keras"),
    '24h': str(MODELS_DIR / "pm_transformer_24h.keras")
}
SCALER_PATH = str(MODELS_DIR / "scaler.pkl")
MEDIANS_PATH = str(MODELS_DIR / "medians.json")

# Time config
DEFAULT_REFRESH_INTERVAL = 120