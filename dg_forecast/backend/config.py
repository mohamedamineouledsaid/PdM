"""
SCADA Industrial Gas Flow Forecasting Dashboard - Configuration Module
"""

import os

# Database Connection Parameters
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT", 5432))
DB_NAME = os.environ.get("DB_NAME", "data_esi_sba_2023")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "root")
DB_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# SCADA Tables & Well Configurations
TABLES = ['dg_values', 'cm_values', 'pt_values', 'pl_values', 'dp_values', 'of_values']
AVAILABLE_WELLS = ["LNB712-CE"]
DEFAULT_WELL = "LNB712-CE"

# Maintenance / Intervention File Path
INTERVENTION_FILE_PATH = r"C:\Users\BADRO INFO\OneDrive\سطح المكتب\MAINTENANCE PUITS\LNB712-CE.csv"

# Model Horizons Configuration
HORIZON_MAP = {
    "1 Hour": {
        "horizon_min": 60,
        "model_path": os.path.join("models", "dg_forecast_1h.keras"),
        "fallback_path": "model_h3.keras",
        "label": "1 Hour (60 min)"
    },
    "3 Hours": {
        "horizon_min": 180,
        "model_path": os.path.join("models", "dg_forecast_3h.keras"),
        "fallback_path": "model_h3.keras",
        "label": "3 Hours (180 min)"
    },
    "6 Hours": {
        "horizon_min": 360,
        "model_path": os.path.join("models", "dg_forecast_6h.keras"),
        "fallback_path": "model_h6.keras",
        "label": "6 Hours (360 min)"
    }
}

DEFAULT_HORIZON = "6 Hours"

# Sequence Window Length (in minutes)
SEQ_LEN = 360
MIN_LOOKBACK_MINUTES = 1440  # 24 hours lookback to compute 1440-min rolling features cleanly

# Auto-refresh Timer Interval in Seconds
AUTO_REFRESH_INTERVAL_SEC = 120

# Feature Column Definitions (79 Features - Exactly matching notebook)
FEATURE_COLS = [
    # RAW
    'DG',
    'CM',
    'PT',
    'PL',
    'DP',
    'OF',

    # LAGS
    'DG_lag_30',
    'DG_lag_60',
    'DG_lag_180',
    'DG_lag_360',
    'DG_lag_720',

    'PT_lag_60',
    'PT_lag_180',
    'PT_lag_360',

    'DP_lag_60',
    'DP_lag_180',
    'DP_lag_360',

    # DIFFERENCES
    'DG_diff_1',
    'DG_diff_5',
    'DG_diff_30',

    'PT_diff_5',
    'PT_diff_30',

    'DP_diff_5',
    'DP_diff_30',

    # ROLLING MEAN
    'DG_mean_60',
    'DG_mean_180',
    'DG_mean_720',

    'PT_mean_60',
    'PT_mean_180',

    'DP_mean_60',
    'DP_mean_180',

    # ROLLING STD
    'DG_std_60',
    'DG_std_180',

    'PT_std_60',
    'DP_std_60',

    # EMA
    'DG_ema_60',
    'DG_ema_180',

    'PT_ema_60',
    'DP_ema_60',

    'DG_ema_gap',
    'PT_ema_gap',
    'DP_ema_gap',

    # TREND
    'DG_trend_60',
    'DG_trend_180',
    'DG_trend_720',

    'PT_trend_60',
    'PT_trend_180',

    'DP_trend_60',
    'DP_trend_180',

    # Z SCORE
    'DG_zscore',
    'PT_zscore',
    'DP_zscore',

    # FFT
    'DG_fft_energy',
    'PT_fft_energy',
    'DP_fft_energy',

    # PRESSURE
    'PT_PL_diff',
    'DP_PT_gap',
    'pressure_ratio',
    'pressure_drop',
    'pressure_instability',

    # FLOW
    'flow_efficiency',
    'flow_per_pressure',
    'flow_change_60',

    # DEGRADATION
    'dg_roll_max_1440',
    'dg_loss_from_peak',
    'dg_degradation_rate',
    'dg_rolling_slope_360',

    # VALVE
    'valve_drift',
    'valve_change_rate',

    # TIME
    'hour_sin',
    'hour_cos',
    'dow_sin',
    'dow_cos',

    # INTERVENTION HISTORY
    'is_intervention',
    'hours_since_intervention',
    'dg_decay_since_intervention',
    'pt_since_intervention',
    'dg_loss_since_intervention',
    'hours_since_intervention_log',
]
