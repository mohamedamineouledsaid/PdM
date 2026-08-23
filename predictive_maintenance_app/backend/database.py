import pandas as pd
import time
from sqlalchemy import create_engine, text
from .config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME, TABLES, WELLS
from utils.logger import log_event

def get_engine(retries=3, delay=2):
    for attempt in range(retries):
        try:
            engine = create_engine(
                f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}',
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True
            )
            return engine
        except Exception as e:
            if attempt < retries - 1:
                log_event(f"Database connection attempt {attempt+1} failed. Retrying...", "WARNING")
                time.sleep(delay)
            else:
                log_event(f"Database connection error after {retries} retries: {e}", "ERROR")
                return None

def fetch_recent_data(well_name, hours=24, offset=0):
    """
    Fetch the most recent `hours` of data from all sensor tables for a given well.
    To ensure we have enough data to generate features (like rolling 60, lags, etc.)
    we should fetch at least 24 hours of data.
    Since we need exactly 720 minutes (12 hours) of valid features for the sequence,
    fetching 24-48 hours gives enough buffer for the feature engineering's NaN generation.
    """
    engine = get_engine()
    if engine is None:
        return pd.DataFrame()

    # Get the latest timestamp to simulate time progression correctly across all sensors
    try:
        query_max = text(f"SELECT MAX(time) as max_time FROM {TABLES[0]} WHERE well = :well_name")
        max_time_df = pd.read_sql(query_max, engine, params={"well_name": well_name})
        max_time = max_time_df['max_time'].iloc[0]
        if pd.isna(max_time):
            return pd.DataFrame()
            
        max_time = pd.to_datetime(max_time, utc=True)
        current_time = max_time - pd.Timedelta(minutes=offset)
        start_time = current_time - pd.Timedelta(hours=hours)
        
        current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        log_event(f"Error determining max time for well {well_name}: {e}", "ERROR")
        return pd.DataFrame()

    all_data = []
    for table in TABLES:
        # We need the sensor value column name (excluding 'time' and 'well')
        # We will query and extract
        try:
            # Parameterized query to prevent SQL Injection
            query = text(f"""
                SELECT *
                FROM {table}
                WHERE well = :well_name
                AND time > :start_time
                AND time <= :current_time
                ORDER BY time DESC
            """)
            df = pd.read_sql(query, engine, params={
                "well_name": well_name,
                "start_time": start_time_str,
                "current_time": current_time_str
            })
            if df.empty:
                continue
            
            value_cols = [c for c in df.columns if c not in ['time', 'well']]
            if len(value_cols) != 1:
                continue
            
            value_col = value_cols[0]
            df = df.rename(columns={value_col: 'value'})
            df['sensor'] = table.split('_')[0].upper()
            df['time'] = pd.to_datetime(df['time'], utc=True)
            all_data.append(df)
            
        except Exception as e:
            log_event(f"Error querying {table} for well {well_name}: {e}", "ERROR")
            continue

    if not all_data:
        return pd.DataFrame()

    df_raw = pd.concat(all_data, ignore_index=True)
    df_raw = df_raw.sort_values(['well', 'time']).reset_index(drop=True)
    df_raw = df_raw.drop_duplicates(subset=['time', 'well', 'sensor'])
    return df_raw

def fetch_training_data_for_scaler(well_name, limit=100000):
    """
    Fetches a large chunk of historical data to fit the RobustScaler if it's missing.
    """
    engine = get_engine()
    if engine is None:
        return pd.DataFrame()

    all_data = []
    for table in TABLES:
        try:
            # Added deterministic ordering and parameterization
            query = text(f"""
                SELECT *
                FROM {table}
                WHERE well = :well_name
                ORDER BY time ASC
                LIMIT :limit
            """)
            df = pd.read_sql(query, engine, params={
                "well_name": well_name,
                "limit": limit
            })
            if df.empty:
                continue
            value_cols = [c for c in df.columns if c not in ['time', 'well']]
            if len(value_cols) == 1:
                df = df.rename(columns={value_cols[0]: 'value'})
                df['sensor'] = table.split('_')[0].upper()
                df['time'] = pd.to_datetime(df['time'], utc=True)
                all_data.append(df)
        except Exception as e:
            log_event(f"Error querying {table} for training data: {e}", "ERROR")

    if not all_data:
        return pd.DataFrame()
    df_raw = pd.concat(all_data, ignore_index=True)
    df_raw = df_raw.sort_values(['well', 'time']).reset_index(drop=True)
    df_raw = df_raw.drop_duplicates(subset=['time', 'well', 'sensor'])
    return df_raw
