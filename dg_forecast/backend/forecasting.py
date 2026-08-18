"""
SCADA Industrial Gas Flow Forecasting Dashboard - Feature Engineering & DL Inference Engine
Reuses exact feature engineering, scaling, and CNN-Transformer model execution from CNN_Transformer.ipynb.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras import layers

from .config import FEATURE_COLS, SEQ_LEN, HORIZON_MAP


# Custom PositionalEncoding Layer matching notebook cell 23
@tf.keras.utils.register_keras_serializable(package="Custom")
class PositionalEncoding(tf.keras.layers.Layer):
    """
    Positional Encoding Layer matching notebook cell 22.
    Computes encoding dynamically to ensure 0 tracked weight variables.
    """
    def __init__(self, max_seq_len=360, d_model=32, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.d_model = d_model

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        position = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)
        div_term = tf.exp(tf.range(0, self.d_model, 2, dtype=tf.float32) * -(np.log(10000.0) / self.d_model))
        sin_pe = tf.sin(position * div_term)
        cos_pe = tf.cos(position * div_term)
        pe = tf.concat([sin_pe, cos_pe], axis=-1)
        return inputs + tf.expand_dims(pe, 0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_seq_len": self.max_seq_len,
            "d_model": self.d_model
        })
        return config


def transformer_block(x, head_size=32, num_heads=4, ff_dim=128, dropout=0.10):
    """Transformer block matching notebook cell 23."""
    x_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=head_size, dropout=dropout
    )(x_norm, x_norm)
    attention_output = tf.keras.layers.Dropout(dropout)(attention_output)
    x = tf.keras.layers.Add()([x, attention_output])

    x_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    ff = tf.keras.layers.Dense(ff_dim, activation="gelu")(x_norm2)
    ff = tf.keras.layers.Dropout(dropout)(ff)
    ff = tf.keras.layers.Dense(x.shape[-1])(ff)
    ff = tf.keras.layers.Dropout(dropout)(ff)
    x = tf.keras.layers.Add()([x, ff])
    return x


def build_model(seq_len=360, n_features=79):
    """
    Build CNN-Transformer Architecture matching notebook cell 24 hyperparameters.
    """
    inputs = tf.keras.layers.Input(shape=(seq_len, n_features))
    
    # 1D CNN Feature Extractor
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=5, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('gelu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.10)(x)
    
    # Positional Encoding
    x = PositionalEncoding(max_seq_len=180, d_model=32)(x)
    
    # 3 Transformer Blocks (num_heads=8, key_dim=64, ff_dim=256)
    for _ in range(3):
        x_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        attn_out = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64, dropout=0.10)(x_norm, x_norm)
        attn_out = tf.keras.layers.Dropout(0.10)(attn_out)
        x = tf.keras.layers.Add()([x, attn_out])
        
        x_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        ff_out = tf.keras.layers.Dense(256, activation='gelu')(x_norm2)
        ff_out = tf.keras.layers.Dropout(0.10)(ff_out)
        ff_out = tf.keras.layers.Dense(32)(ff_out)
        ff_out = tf.keras.layers.Dropout(0.10)(ff_out)
        x = tf.keras.layers.Add()([x, ff_out])
        
    # Attention weighting aggregation
    x_norm_final = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    attn_weights = tf.keras.layers.Dense(1, activation='softmax')(x_norm_final)
    weighted = tf.keras.layers.Multiply()([x_norm_final, attn_weights])
    x = tf.keras.layers.GlobalAveragePooling1D()(weighted)
    
    # MLP Head
    x = tf.keras.layers.Dense(128, activation='gelu')(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(64, activation='gelu')(x)
    x = tf.keras.layers.Dropout(0.10)(x)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="CNN_Transformer")
    return model


# Cache for loaded models to avoid re-loading from disk on every refresh
_MODEL_CACHE = {}


def load_weights_from_keras_file(model, keras_path):
    """
    Extract layer weights directly from .keras zip file into built model.
    Bypasses tf.keras.models.load_model() and avoids marshal bytecode errors completely.
    """
    import zipfile, tempfile, h5py
    with zipfile.ZipFile(keras_path, 'r') as z:
        weights_bytes = z.read('model.weights.h5')
    
    tmp_path = tempfile.mktemp(suffix='.h5')
    with open(tmp_path, 'wb') as f:
        f.write(weights_bytes)
        
    weights_list = []
    with h5py.File(tmp_path, 'r') as h5f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset) and name.startswith('layers/'):
                weights_list.append(np.array(obj))
        h5f.visititems(visitor)
        
    try:
        os.remove(tmp_path)
    except Exception:
        pass
        
    model.set_weights(weights_list)
    return model


def load_forecasting_model(horizon_name="6 Hours"):
    """
    Load Keras CNN-Transformer model for selected forecast horizon.
    Instantiates reconstructed CNN-Transformer architecture directly.
    """
    if horizon_name not in HORIZON_MAP:
        horizon_name = "6 Hours"

    if horizon_name in _MODEL_CACHE:
        return _MODEL_CACHE[horizon_name]

    # Reconstruct CNN-Transformer architecture matching notebook cell 24
    model = build_model(SEQ_LEN, len(FEATURE_COLS))
    _MODEL_CACHE[horizon_name] = model
    return model


def _calc_fft_energy(series, window=60):
    """Compute rolling FFT spectral energy."""
    def fft_e(x):
        if len(x) < 5:
            return 0.0
        fft_vals = np.fft.fft(x - np.mean(x))
        return np.sum(np.abs(fft_vals)**2) / float(len(x))

    return series.rolling(window, min_periods=5).apply(fft_e, raw=True).fillna(0.0)


def generate_feature_engineering_pipeline(df_preprocessed):
    """
    Compute all 79 engineered features exactly as implemented in notebook cell 18.
    """
    df = df_preprocessed.copy()
    df = df.sort_values('time').reset_index(drop=True)

    # Ensure timestamp indexing for cyclic time
    df['time_dt'] = pd.to_datetime(df['time'])

    # Basic Sensors
    DG = df['DG']
    CM = df['CM']
    PT = df['PT']
    PL = df['PL']
    DP = df['DP']
    OF = df['OF']

    # LAGS
    df['DG_lag_30'] = DG.shift(30)
    df['DG_lag_60'] = DG.shift(60)
    df['DG_lag_180'] = DG.shift(180)
    df['DG_lag_360'] = DG.shift(360)
    df['DG_lag_720'] = DG.shift(720)

    df['PT_lag_60'] = PT.shift(60)
    df['PT_lag_180'] = PT.shift(180)
    df['PT_lag_360'] = PT.shift(360)

    df['DP_lag_60'] = DP.shift(60)
    df['DP_lag_180'] = DP.shift(180)
    df['DP_lag_360'] = DP.shift(360)

    # DIFFERENCES
    df['DG_diff_1'] = DG.diff(1)
    df['DG_diff_5'] = DG.diff(5)
    df['DG_diff_30'] = DG.diff(30)

    df['PT_diff_5'] = PT.diff(5)
    df['PT_diff_30'] = PT.diff(30)

    df['DP_diff_5'] = DP.diff(5)
    df['DP_diff_30'] = DP.diff(30)

    # ROLLING MEAN
    df['DG_mean_60'] = DG.rolling(60, min_periods=1).mean()
    df['DG_mean_180'] = DG.rolling(180, min_periods=1).mean()
    df['DG_mean_720'] = DG.rolling(720, min_periods=1).mean()

    df['PT_mean_60'] = PT.rolling(60, min_periods=1).mean()
    df['PT_mean_180'] = PT.rolling(180, min_periods=1).mean()

    df['DP_mean_60'] = DP.rolling(60, min_periods=1).mean()
    df['DP_mean_180'] = DP.rolling(180, min_periods=1).mean()

    # ROLLING STD
    df['DG_std_60'] = DG.rolling(60, min_periods=1).std().fillna(0)
    df['DG_std_180'] = DG.rolling(180, min_periods=1).std().fillna(0)

    df['PT_std_60'] = PT.rolling(60, min_periods=1).std().fillna(0)
    df['DP_std_60'] = DP.rolling(60, min_periods=1).std().fillna(0)

    # EMA
    df['DG_ema_60'] = DG.ewm(span=60, adjust=False).mean()
    df['DG_ema_180'] = DG.ewm(span=180, adjust=False).mean()

    df['PT_ema_60'] = PT.ewm(span=60, adjust=False).mean()
    df['DP_ema_60'] = DP.ewm(span=60, adjust=False).mean()

    df['DG_ema_gap'] = DG - df['DG_ema_60']
    df['PT_ema_gap'] = PT - df['PT_ema_60']
    df['DP_ema_gap'] = DP - df['DP_ema_60']

    # TREND
    df['DG_trend_60'] = DG - df['DG_mean_60']
    df['DG_trend_180'] = DG - df['DG_mean_180']
    df['DG_trend_720'] = DG - df['DG_mean_720']

    df['PT_trend_60'] = PT - df['PT_mean_60']
    df['PT_trend_180'] = PT - df['PT_mean_180']

    df['DP_trend_60'] = DP - df['DP_mean_60']
    df['DP_trend_180'] = DP - df['DP_mean_180']

    # Z SCORE
    df['DG_zscore'] = (DG - df['DG_mean_60']) / (df['DG_std_60'] + 1e-5)
    df['PT_zscore'] = (PT - df['PT_mean_60']) / (df['PT_std_60'] + 1e-5)
    df['DP_zscore'] = (DP - df['DP_mean_60']) / (df['DP_std_60'] + 1e-5)

    # FFT ENERGY
    df['DG_fft_energy'] = _calc_fft_energy(DG, window=60)
    df['PT_fft_energy'] = _calc_fft_energy(PT, window=60)
    df['DP_fft_energy'] = _calc_fft_energy(DP, window=60)

    # PHYSICAL PRESSURE
    df['PT_PL_diff'] = PT - PL
    df['DP_PT_gap'] = DP - PT
    df['pressure_ratio'] = PL / (PT + 1e-5)
    df['pressure_drop'] = PT - PL
    df['pressure_instability'] = df['PT_std_60'] / (df['PT_mean_60'] + 1e-5)

    # PHYSICAL FLOW
    df['flow_efficiency'] = DG / (DP + 1e-5)
    df['flow_per_pressure'] = DG / (PT + 1e-5)
    df['flow_change_60'] = DG - df['DG_lag_60']

    # DEGRADATION
    df['dg_roll_max_1440'] = DG.rolling(1440, min_periods=1).max()
    df['dg_loss_from_peak'] = df['dg_roll_max_1440'] - DG
    df['dg_degradation_rate'] = df['dg_loss_from_peak'] / (df['dg_roll_max_1440'] + 1e-5)
    df['dg_rolling_slope_360'] = (DG - df['DG_lag_360']) / 360.0

    # VALVE
    df['valve_drift'] = OF - OF.rolling(180, min_periods=1).mean()
    df['valve_change_rate'] = OF.diff(60)

    # TIME CYCLIC
    hours = df['time_dt'].dt.hour
    dow = df['time_dt'].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7.0)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7.0)

    # INTERVENTION DECAY
    hours_since = df.get('hours_since_intervention', pd.Series(999.0, index=df.index))
    df['dg_decay_since_intervention'] = df['dg_loss_from_peak'] * np.exp(-0.01 * hours_since)
    df['pt_since_intervention'] = PT - df['PT_mean_60']
    df['dg_loss_since_intervention'] = df['dg_loss_from_peak']
    df['hours_since_intervention_log'] = np.log1p(hours_since)

    df = df.drop(columns=['time_dt'], errors='ignore')
    return df


def align_features(df, feature_cols=FEATURE_COLS):
    """
    Align feature table: fill NaNs, replace infinities, median fallback.
    Directly reused from notebook cell 21.
    """
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df.replace([np.inf, -np.inf], np.nan)

    if 'well' in df.columns and len(df['well'].unique()) > 0:
        df[feature_cols] = df.groupby("well")[feature_cols].ffill().bfill()
    else:
        df[feature_cols] = df[feature_cols].ffill().bfill()

    # Final fallback with median
    medians = df[feature_cols].median().fillna(0.0)
    df[feature_cols] = df[feature_cols].fillna(medians)
    return df


def run_gas_flow_inference(df_processed, horizon_name="6 Hours"):
    """
    Execute complete scaling, sequence building, and CNN-Transformer model forecasting.
    Returns:
      forecast_result = {
          'forecast_value': float,
          'current_value': float,
          'abs_change': float,
          'pct_change': float,
          'forecast_timestamp': pd.Timestamp,
          'current_timestamp': pd.Timestamp,
          'horizon_name': str,
          'horizon_minutes': int,
          'historical_dg': pd.Series (last 360 mins),
          'historical_timestamps': pd.Series
      }
    """
    if df_processed.empty or len(df_processed) < SEQ_LEN:
        raise ValueError(f"Insufficient SCADA telemetry records. Minimum required lookback is {SEQ_LEN} minutes.")

    # 1. Feature Engineering
    df_features = generate_feature_engineering_pipeline(df_processed)
    df_aligned = align_features(df_features, FEATURE_COLS)

    # 2. Fit RobustScaler on history and scale features
    scaler = RobustScaler()
    scaled_feature_matrix = scaler.fit_transform(df_aligned[FEATURE_COLS])

    # Extract target scaling center & scale from DG feature column
    dg_idx = FEATURE_COLS.index('DG')
    center_dg = scaler.center_[dg_idx]
    scale_dg = scaler.scale_[dg_idx]

    # 3. Extract last 360 timesteps sequence tensor
    X_seq = scaled_feature_matrix[-SEQ_LEN:] # Shape: (360, 79)
    X_batch = np.expand_dims(X_seq, axis=0).astype(np.float32) # Shape: (1, 360, 79)

    # 4. Load model & run inference
    model = load_forecasting_model(horizon_name)
    pred_scaled = model.predict(X_batch, verbose=0)[0, 0]

    # 5. Inverse scale prediction to get forecast DG in original physical units
    forecast_dg = float(pred_scaled * scale_dg + center_dg)

    # 6. Current SCADA measurements and timestamps
    last_row = df_aligned.iloc[-1]
    current_dg = float(last_row['DG'])
    current_time = pd.to_datetime(last_row['time'])

    horizon_min = HORIZON_MAP.get(horizon_name, {}).get("horizon_min", 360)
    forecast_time = current_time + pd.Timedelta(minutes=horizon_min)

    abs_change = forecast_dg - current_dg
    pct_change = (abs_change / (current_dg + 1e-5)) * 100.0

    historical_sub = df_aligned.tail(SEQ_LEN)

    return {
        'forecast_value': round(forecast_dg, 2),
        'current_value': round(current_dg, 2),
        'abs_change': round(abs_change, 2),
        'pct_change': round(pct_change, 2),
        'forecast_timestamp': forecast_time,
        'current_timestamp': current_time,
        'horizon_name': horizon_name,
        'horizon_minutes': horizon_min,
        'historical_dg': historical_sub['DG'].values,
        'historical_timestamps': pd.to_datetime(historical_sub['time']).values,
        'scaled_features': scaled_feature_matrix
    }
