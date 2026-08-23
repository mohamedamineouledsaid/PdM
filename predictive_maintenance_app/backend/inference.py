import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
import joblib

from .config import MODEL_PATHS, SCALER_PATH, MEDIANS_PATH, SEQ_LEN
from .database import fetch_training_data_for_scaler
from .preprocessing import resample_and_pivot, impute_sensor_data, reshape_sensor_data, engineer_features, align_features
from utils.logger import log_event

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Custom Layer needed for loading the model
class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)

        self.seq_len = seq_len
        self.d_model = d_model

        position = np.arange(seq_len)[:, None]
        div_term = np.exp(
            np.arange(0, d_model, 2) *
            -(np.log(10000.0) / d_model)
        )

        pe = np.zeros((seq_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = tf.constant(pe[None, ...], dtype=tf.float32)

    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "seq_len": self.seq_len,
            "d_model": self.d_model,
        })
        return config

class ReduceSumLayer(tf.keras.layers.Layer):
    def call(self, x, **kwargs):
        return tf.reduce_sum(x, axis=1)
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
        
    def get_config(self):
        return super().get_config()

FEATURE_COLS = [
    'DG', 'PT', 'PL', 'DP', 'CM', 'OF',
    'PT_diff_1', 'DP_diff_1', 'DG_diff_1',
    'PT_pct', 'DP_pct', 'DG_pct',
    'PT_PL_diff', 'DP_PT_gap',
    'DP_vol_20', 'PT_vol_20', 'DG_vol_20',
    'DP_ema_20', 'DP_ema_60', 'PT_ema_20', 'PT_ema_60', 'DG_ema_20', 'DG_ema_60',
    'DP_shock_10', 'PT_shock_10', 'DG_shock_10',
    'PT_mean_60', 'PT_std_60', 'DP_mean_60', 'DP_std_60',
    'PT_trend_60', 'DP_trend_60',
    'DP_zscore', 'PT_zscore',
    'PT_lag_1', 'PT_lag_6', 'DP_lag_1', 'DP_lag_6', 'CM_lag_1', 'CM_lag_6',
    'DG_fft_energy', 'DP_fft_energy', 'PT_fft_energy',
    'flow_efficiency', 'pressure_ratio', 'efficiency_loss',
    'pressure_instability', 'pressure_drop', 'valve_drift',
    'hour_sin', 'hour_cos'
]

# Optimal Threshold obtained during model evaluation
BEST_THRESHOLD = 0.70

class PredictiveMaintenanceModel:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.medians = None
        self.scaler_path = SCALER_PATH
        self.medians_path = MEDIANS_PATH
        self._load_models()
        self._load_or_fit_scaler()

    def _load_models(self):
        tf.keras.backend.clear_session()
        for horizon, path in MODEL_PATHS.items():
            try:
                self.models[horizon] = tf.keras.models.load_model(
                    path,
                    custom_objects={
                        'PositionalEncoding': PositionalEncoding, 
                        'ReduceSumLayer': ReduceSumLayer,
                        'tf': tf
                    },
                    safe_mode=True, # Improved Security
                    compile=False
                )
                log_event(f"Model {horizon} loaded successfully.", "SUCCESS")
            except Exception as e:
                log_event(f"Error loading model {horizon}: {e}", "ERROR")
            
    def _load_or_fit_scaler(self):
        if os.path.exists(self.scaler_path) and os.path.exists(self.medians_path):
            try:
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = joblib.load(f)
                with open(self.medians_path, 'r') as f:
                    self.medians = json.load(f)
                log_event("Scaler and medians loaded from disk.", "SUCCESS")
            except Exception as e:
                log_event(f"Error loading scaler/medians: {e}", "ERROR")
        else:
            log_event("Scaler/medians not found. Attempting to fit on historical data...", "INFO")
            self.scaler = RobustScaler()
            # Fetch a large chunk of historical data (e.g. 100k rows) from a representative well
            df_raw = fetch_training_data_for_scaler("LNB234-CE", limit=50000)
            if not df_raw.empty:
                df_resampled = resample_and_pivot(df_raw)
                df_imputed = impute_sensor_data(df_resampled)
                df_wide = reshape_sensor_data(df_imputed)
                df_features = engineer_features(df_wide)
                df_aligned = align_features(df_features, FEATURE_COLS) # Calculate initial medians implicitly
                
                # Extract and save medians directly from dataframe to avoid data leakage during real-time inference
                self.medians = df_aligned[FEATURE_COLS].median().to_dict()
                self.scaler.fit(df_aligned[FEATURE_COLS])
                
                # Save artifacts
                try:
                    os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
                    with open(self.scaler_path, 'wb') as f:
                        joblib.dump(self.scaler, f)
                    with open(self.medians_path, 'w') as f:
                        json.dump(self.medians, f)
                    log_event("Scaler and medians fitted and saved.", "SUCCESS")
                except Exception as e:
                    log_event(f"Failed to save scaler/medians: {e}", "ERROR")
            else:
                log_event("Warning: Could not fetch data to fit scaler. Predictions will be incorrect.", "WARNING")

    def process_and_predict(self, df_raw):
        if not self.models or self.scaler is None:
            return None, None, "Models or Scaler not loaded"
            
        if len(df_raw) < SEQ_LEN:
            return None, None, f"Not enough data. Need at least {SEQ_LEN} rows, got {len(df_raw)}."

        try:
            df_resampled = resample_and_pivot(df_raw)
            df_imputed = impute_sensor_data(df_resampled)
            df_wide = reshape_sensor_data(df_imputed)
            df_features = engineer_features(df_wide)
            if df_features.empty:
                return None, None, "Feature engineering resulted in empty dataframe."
                
            # Align features securely with pre-calculated training medians
            df_aligned = align_features(df_features, FEATURE_COLS, training_medians=self.medians)
            
            if len(df_aligned) < SEQ_LEN:
                return None, None, f"Not enough data after feature engineering. Need {SEQ_LEN}, got {len(df_aligned)}."
                
            scaled_data = self.scaler.transform(df_aligned[FEATURE_COLS])
            
            sequence = scaled_data[-SEQ_LEN:]
            X = np.expand_dims(sequence, axis=0).astype(np.float32)
            
            # Predict for all loaded models
            probs = {}
            for horizon, model in self.models.items():
                probs[horizon] = float(model.predict(X, verbose=0)[0][0])
            
            sequence_timestamp = df_aligned['time'].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
            return probs, sequence_timestamp, None
            
        except Exception as e:
            log_event(f"Inference error: {e}", "ERROR")
            return None, None, str(e)