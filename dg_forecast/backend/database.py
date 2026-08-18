"""
SCADA Industrial Gas Flow Forecasting Dashboard - Database Access Module
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect, text
from .config import DB_URI, TABLES, DEFAULT_WELL


def get_db_engine():
    """Create and return an SQLAlchemy database engine."""
    return create_engine(DB_URI, pool_pre_ping=True, pool_recycle=3600)


def check_db_connection():
    """
    Check if PostgreSQL database is reachable.
    Returns (status: bool, message: str)
    """
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True, "Connected to PostgreSQL"
    except Exception as e:
        return False, f"Database Connection Error: {str(e)}"


def get_available_tables():
    """Inspect and return available tables in database."""
    try:
        engine = get_db_engine()
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        engine.dispose()
        return tables
    except Exception as e:
        print(f"Error inspecting database tables: {e}")
        return []


def load_scada_telemetry(well=DEFAULT_WELL, tables=None, limit_records=None):
    """
    Load raw SCADA sensor telemetry for specified well from PostgreSQL tables.
    Returns combined long-format DataFrame df_raw.
    """
    if tables is None:
        tables = TABLES

    engine = get_db_engine()
    inspector = inspect(engine)
    available = inspector.get_table_names()

    all_data = []

    try:
        for table in tables:
            if table not in available:
                print(f"⚠️ Table '{table}' not found in database.")
                continue

            cols = inspector.get_columns(table)
            col_names = [col['name'] for col in cols]
            value_cols = [c for c in col_names if c not in ['time', 'well']]

            if len(value_cols) != 1:
                continue

            value_col = value_cols[0]

            if limit_records:
                query = f"""
                    SELECT time, well, {value_col}
                    FROM (
                        SELECT time, well, {value_col}
                        FROM {table}
                        WHERE well = '{well}'
                        ORDER BY time DESC
                        LIMIT {limit_records}
                    ) sub
                    ORDER BY time ASC
                """
            else:
                query = f"""
                    SELECT time, well, {value_col}
                    FROM {table}
                    WHERE well = '{well}'
                    ORDER BY time ASC
                """

            df = pd.read_sql(query, engine)
            if df.empty:
                continue

            df = df.rename(columns={value_col: 'value'})
            df['sensor'] = table.split('_')[0].upper()
            df['time'] = pd.to_datetime(df['time'])

            all_data.append(df)

        if not all_data:
            return pd.DataFrame()

        df_raw = pd.concat(all_data, ignore_index=True)
        df_raw = df_raw.sort_values(['well', 'time']).reset_index(drop=True)
        df_raw = df_raw.drop_duplicates(subset=['time', 'well', 'sensor'])

        return df_raw

    finally:
        engine.dispose()
