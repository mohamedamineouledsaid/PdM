"""
SCADA Industrial Gas Flow Forecasting Dashboard - Data Preprocessing Module
Reuses exact preprocessing & imputation logic from CNN_Transformer.ipynb.
"""

import pandas as pd
import numpy as np
import os
from .config import INTERVENTION_FILE_PATH


def resample_raw_telemetry(df_raw, freq='1min'):
    """
    Pivot raw long-format telemetry into uniform 1-minute time series grid.
    """
    df = df_raw.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')

    # Long to wide multi-index table
    df_pivot = df.pivot_table(
        index='time',
        columns=['well', 'sensor'],
        values='value',
        aggfunc='mean'
    )

    # Uniform temporal resampling
    df_resampled = df_pivot.resample(freq).mean()
    return df_resampled


def impute_sensor_data(df, max_gap=40, very_large_gap=300):
    """
    Sensor-aware imputation for high and low frequency sensors.
    NO DATA LEAKAGE - uses only past information for imputation.
    Directly reused from notebook cell 9.
    """
    high_freq_sensors = ['CM', 'DG', 'OF']
    low_freq_sensors = ['PT', 'DP', 'PL']

    df_imputed = df.copy()

    for well in df_imputed.columns.levels[0]:
        df_well = df_imputed[well].copy()

        # HIGH FREQUENCY SENSORS
        for sensor in high_freq_sensors:
            if sensor not in df_well.columns:
                continue

            s = df_well[sensor].copy()
            is_nan = s.isna()

            group = (is_nan != is_nan.shift()).cumsum()
            gap_sizes = is_nan.groupby(group).transform('sum')

            small_gap_mask = gap_sizes <= max_gap

            # Time interpolation
            s_interp = s.interpolate(
                method='time',
                limit=max_gap,
                limit_direction='forward',
            )

            s_filled = s.where(~is_nan, s_interp)
            s_filled[is_nan & (~small_gap_mask)] = np.nan

            # Smooth only interpolated values
            interp_mask = is_nan & small_gap_mask
            rolling_mean = s_filled.rolling(
                window=5,
                min_periods=1,
                center=False
            ).mean()

            s_final = s_filled.copy()
            s_final[interp_mask] = rolling_mean[interp_mask]

            df_well[sensor] = s_final

        # LOW FREQUENCY SENSORS
        for sensor in low_freq_sensors:
            if sensor not in df_well.columns:
                continue

            s = df_well[sensor].copy()
            is_nan = s.isna()

            group = (is_nan != is_nan.shift()).cumsum()
            gap_sizes = is_nan.groupby(group).transform('sum')

            large_gap_mask = gap_sizes > very_large_gap

            s_filled = s.ffill(limit=max_gap)
            s_filled[large_gap_mask & is_nan] = np.nan

            df_well[sensor] = s_filled

        # Write back
        for sensor in df_well.columns:
            df_imputed[(well, sensor)] = df_well[sensor]

    return df_imputed


def reshape_sensor_data(df_imputed):
    """
    Reshape imputed sensor data from MultiIndex columns to flat wide DataFrame.
    Directly reused from notebook cell 10.
    """
    if isinstance(df_imputed.columns, pd.MultiIndex):
        df_imputed.columns = [f"{well}_{sensor}" for well, sensor in df_imputed.columns]

    df_flat = df_imputed.reset_index()

    # Wide to long
    df_melted = df_flat.melt(
        id_vars='time',
        var_name='well_sensor',
        value_name='value'
    )

    # Split well and sensor
    df_melted[['well', 'sensor']] = df_melted['well_sensor'].str.split('_', expand=True)

    # Long to wide
    df_wide = df_melted.pivot_table(
        index=['time', 'well'],
        columns='sensor',
        values='value'
    ).reset_index()

    df_wide.columns.name = None
    return df_wide


def remove_critical_missing_rows(df_wide, required_cols=None):
    """
    Remove rows missing primary required sensors.
    Directly reused from notebook cell 11.
    """
    if required_cols is None:
        required_cols = ['DG', 'CM', 'PT', 'DP']

    before_dropna = len(df_wide)
    df_final = df_wide.dropna(subset=required_cols).copy()
    after_dropna = len(df_final)

    stats = {
        'before': before_dropna,
        'after': after_dropna,
        'removed': before_dropna - after_dropna
    }
    return df_final, stats


def load_intervention_data(filepath=INTERVENTION_FILE_PATH):
    """
    Load and parse well intervention CSV file.
    Directly reused from notebook cells 12-14.
    """
    if not os.path.exists(filepath):
        return pd.DataFrame(columns=['well', 'activity', 'sub_activity', 'start_time', 'end_time', 'expected_gain'])

    try:
        df_intervention = pd.read_csv(filepath, sep=",")
        df_intervention.columns = df_intervention.columns.str.strip()

        interv = df_intervention.copy()
        interv["start_time"] = pd.to_datetime(
            interv["DATE"].astype(str) + " " + interv["Heure de fermeture"].astype(str),
            dayfirst=True,
            utc=True
        )
        interv["end_time"] = pd.to_datetime(
            interv["DATE"].astype(str) + " " + interv["Heure d'ouverture"].astype(str),
            dayfirst=True,
            utc=True
        )

        interv = interv.rename(columns={
            "Puits": "well",
            "ACTIVITE": "activity",
            "SOUS ACTIVITE": "sub_activity"
        })

        if "Débit après trvx" in interv.columns and "Débit initail" in interv.columns:
            interv["expected_gain"] = interv["Débit après trvx"] - interv["Débit initail"]
        else:
            interv["expected_gain"] = 0

        return interv[['well', 'activity', 'sub_activity', 'start_time', 'end_time', 'expected_gain']]
    except Exception as e:
        print(f"Warning: Could not parse intervention file: {e}")
        return pd.DataFrame(columns=['well', 'activity', 'sub_activity', 'start_time', 'end_time', 'expected_gain'])


def add_intervention_features(sensor_df, intervention_df=None):
    """
    Add maintenance/intervention features to telemetry.
    Directly reused from notebook cell 15.
    """
    if intervention_df is None:
        intervention_df = load_intervention_data()

    df = sensor_df.copy()

    df["is_intervention"] = 0
    df["hours_since_intervention"] = np.nan
    df["hours_to_intervention"] = np.nan

    if intervention_df.empty:
        df["hours_since_intervention"] = 999.0
        return df

    # Ensure UTC timezone alignment for comparison
    df['time_dt'] = pd.to_datetime(df['time'], utc=True)

    for well in df["well"].unique():
        interventions = intervention_df[intervention_df["well"] == well]

        if len(interventions) == 0:
            continue

        # Mark intervention period
        for _, row in interventions.iterrows():
            start = row["start_time"]
            end = row["end_time"]

            mask = (
                (df["well"] == well)
                & (df["time_dt"] >= start)
                & (df["time_dt"] <= end)
            )
            df.loc[mask, "is_intervention"] = 1

        intervention_times = sorted(interventions["start_time"].tolist())
        well_mask = (df["well"] == well)

        for idx in df.loc[well_mask].index:
            t = df.loc[idx, "time_dt"]
            past = [x for x in intervention_times if x <= t]

            if len(past):
                df.loc[idx, "hours_since_intervention"] = (
                    (t - max(past)).total_seconds() / 3600.0
                )

    df = df.drop(columns=['time_dt'])
    df['hours_since_intervention'] = df['hours_since_intervention'].fillna(999.0)
    return df


def execute_full_preprocessing_pipeline(df_raw, well_name=None):
    """
    Run complete preprocessing pipeline on raw telemetry.
    """
    if df_raw.empty:
        return pd.DataFrame()

    df_resampled = resample_raw_telemetry(df_raw)
    df_imputed = impute_sensor_data(df_resampled, max_gap=40, very_large_gap=300)
    df_wide = reshape_sensor_data(df_imputed)
    df_clean, _ = remove_critical_missing_rows(df_wide, ['DG', 'CM', 'PT', 'DP'])
    
    if well_name:
        df_clean = df_clean[df_clean['well'] == well_name].copy()

    df_processed = add_intervention_features(df_clean)
    return df_processed
