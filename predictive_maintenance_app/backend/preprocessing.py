import numpy as np
import pandas as pd

def resample_and_pivot(df_raw):
    df = df_raw.copy()
    if df.empty:
        return pd.DataFrame()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')
    df_pivot = df.pivot_table(
        index='time',
        columns=['well', 'sensor'],
        values='value', 
        aggfunc='mean'
    )
    df_resampled = df_pivot.resample('1min').mean()
    return df_resampled

def impute_sensor_data(df, max_gap=40, very_large_gap=300):
    high_freq_sensors = ['CM', 'DG', 'OF']
    low_freq_sensors = ['PT', 'DP', 'PL']
    
    df_imputed = df.copy()
    
    if isinstance(df_imputed.columns, pd.MultiIndex):
        wells = df_imputed.columns.levels[0]
    else:
        return df_imputed

    for well in wells:
        df_well = df_imputed[well].copy()
        
        # HIGH FREQUENCY
        for sensor in high_freq_sensors:
            if sensor not in df_well.columns:
                continue
            s = df_well[sensor].copy()
            is_nan = s.isna()
            group = (is_nan != is_nan.shift()).cumsum()
            gap_sizes = is_nan.groupby(group).transform('sum')
            small_gap_mask = gap_sizes <= max_gap
            
            s_interp = s.interpolate(method='time', limit=max_gap, limit_direction='forward', limit_area='inside')
            s_filled = s.where(~is_nan, s_interp)
            s_filled[is_nan & (~small_gap_mask)] = np.nan
            
            interp_mask = is_nan & small_gap_mask
            rolling_mean = s_filled.rolling(window=5, min_periods=1, center=False).mean()
            s_final = s_filled.copy()
            s_final[interp_mask] = rolling_mean[interp_mask]
            df_well[sensor] = s_final
            
        # LOW FREQUENCY
        for sensor in low_freq_sensors:
            if sensor not in df_well.columns:
                continue
            s = df_well[sensor].copy()
            is_nan = s.isna()
            group = (is_nan != is_nan.shift()).cumsum()
            gap_sizes = is_nan.groupby(group).transform('sum')
            
            small_gap_mask = gap_sizes <= max_gap
            large_gap_mask = gap_sizes > very_large_gap
            
            s_filled = s.ffill(limit=max_gap)
            s_filled[large_gap_mask & is_nan] = np.nan
            df_well[sensor] = s_filled
            
        for sensor in df_well.columns:
            df_imputed[(well, sensor)] = df_well[sensor]
            
    return df_imputed

def reshape_sensor_data(df_imputed):
    if isinstance(df_imputed.columns, pd.MultiIndex):
        df_imputed.columns = [f"{well}_{sensor}" for well, sensor in df_imputed.columns]
    
    df_flat = df_imputed.reset_index()
    df_melted = df_flat.melt(id_vars='time', var_name='well_sensor', value_name='value')
    df_melted[['well', 'sensor']] = df_melted['well_sensor'].str.split('_', expand=True)
    df_wide = df_melted.pivot_table(index=['time', 'well'], columns='sensor', values='value').reset_index()
    df_wide.columns.name = None
    return df_wide

def add_fft_feature(series, window=64):
    values = series.values
    fft_energy = np.full(len(values), np.nan, dtype=np.float32)
    for i in range(window, len(values)):
        segment = values[i-window:i]
        if np.isnan(segment).mean() > 0.3:
            continue
        segment = np.nan_to_num(segment)
        fft_vals = np.abs(np.fft.fft(segment))
        fft_energy[i] = np.mean(fft_vals)
    return pd.Series(fft_energy, index=series.index)

def engineer_features(df, well_col='well', time_col='time', fft_window=64):
    feature_list = []
    for well in df[well_col].unique():
        df_w = df[df[well_col] == well].sort_values(time_col).set_index(time_col)
        if len(df_w) < 100:
            # We still need to process even if it's less than 100 in realtime, but let's be careful.
            # Realtime might pass 12 hours of data (720 rows) so it should be fine.
            pass
            
        df_feat = df_w[['CM', 'DG', 'PT', 'PL', 'DP', 'OF']].copy()
        
        for col in ['CM', 'PT', 'DP', 'DG']:
            df_feat[f'{col}_diff_1'] = df_feat[col].diff(1)
            df_feat[f'{col}_pct'] = df_feat[col].pct_change()
            
        df_feat['PT_PL_diff'] = df_feat['PT'] - df_feat['PL']
        df_feat['DP_PT_gap'] = df_feat['DP'] - df_feat['PT']
        
        for col in ['DP', 'PT', 'DG']:
            df_feat[f'{col}_vol_20'] = df_feat[col].rolling(20, min_periods=5).std()
            df_feat[f'{col}_ema_20'] = df_feat[col].ewm(span=20, adjust=False).mean()
            df_feat[f'{col}_ema_60'] = df_feat[col].ewm(span=60, adjust=False).mean()
            df_feat[f'{col}_shock_10'] = df_feat[col].diff(10).abs()
            
        for col in ['CM', 'PT', 'DP']:
            df_feat[f'{col}_mean_60'] = df_feat[col].rolling(60, min_periods=10).mean()
            df_feat[f'{col}_std_60'] = df_feat[col].rolling(60, min_periods=10).std()
            df_feat[f'{col}_trend_60'] = df_feat[col].diff(60)
            
        for col in ['DP', 'PT']:
            mean = df_feat[col].rolling(60, min_periods=10).mean()
            std = df_feat[col].rolling(60, min_periods=10).std()
            df_feat[f'{col}_zscore'] = (df_feat[col] - mean) / (std + 1e-6)
            
        for col in ['CM', 'PT', 'DP']:
            df_feat[f'{col}_lag_1'] = df_feat[col].shift(1)
            df_feat[f'{col}_lag_6'] = df_feat[col].shift(6)
            
        df_feat['DG_fft_energy'] = add_fft_feature(df_feat['DG'], fft_window)
        df_feat['DP_fft_energy'] = add_fft_feature(df_feat['DP'], fft_window)
        df_feat['PT_fft_energy'] = add_fft_feature(df_feat['PT'], fft_window)
        
        df_feat['flow_efficiency'] = df_feat['DG'] / (df_feat['DP'] + 1e-6)
        df_feat['pressure_ratio'] = df_feat['PT'] / (df_feat['PL'] + 1e-6)
        df_feat['efficiency_loss'] = (df_feat['CM'] / (df_feat['DG'] + 1e-6)).diff(10)
        
        df_feat['pressure_instability'] = (
            df_feat['PT'].rolling(30, min_periods=5).std() /
            (df_feat['PT'].rolling(30, min_periods=5).mean() + 1e-6)
        )
        
        df_feat['pressure_drop'] = df_feat['PT'].diff(5)
        df_feat['valve_drift'] = df_feat['OF'].diff(30)
        
        df_feat['hour'] = df_feat.index.hour
        df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24)
        df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24)
        
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
        df_feat = df_feat.dropna(how='all')
        df_feat = df_feat.ffill(limit=10)
        df_feat[well_col] = well
        feature_list.append(df_feat)
        
    if len(feature_list) == 0:
        return pd.DataFrame()
        
    df_features = pd.concat(feature_list).reset_index()
    return df_features

def align_features(df, feature_cols, training_medians=None):
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
            
    df = df.replace([np.inf, -np.inf], np.nan)
    
    if 'well' in df.columns:
        df[feature_cols] = df.groupby("well")[feature_cols].ffill().bfill()
    else:
        df[feature_cols] = df[feature_cols].ffill().bfill()
        
    # Use static training medians to avoid data leakage
    if training_medians is not None:
        for col in feature_cols:
            if col in training_medians:
                df[col] = df[col].fillna(training_medians[col])
            else:
                df[col] = df[col].fillna(0)
    else:
        # Fallback to zeros if no medians provided (should not happen in production)
        df[feature_cols] = df[feature_cols].fillna(0)
        
    return df
