"""
Scratch Test Script: Verify Backend Execution
"""
import sys
import os

from backend.database import check_db_connection, load_scada_telemetry
from backend.preprocessing import execute_full_preprocessing_pipeline
from backend.forecasting import run_gas_flow_inference

print("1. Testing Database Connection...")
status, msg = check_db_connection()
print(f"Status: {status}, Msg: {msg}")

print("\n2. Loading SCADA Telemetry...")
df_raw = load_scada_telemetry("LNB627-CE", limit_records=2000)
print(f"Loaded raw records: {len(df_raw)}")

print("\n3. Running Preprocessing Pipeline...")
df_processed = execute_full_preprocessing_pipeline(df_raw, well_name="LNB627-CE")
print(f"Processed shape: {df_processed.shape}")

print("\n4. Running Model Inference for 6 Hours horizon...")
res_6h = run_gas_flow_inference(df_processed, horizon_name="6 Hours")
print("Forecast Result 6H:", res_6h['forecast_value'], "Current:", res_6h['current_value'], "Delta %:", res_6h['pct_change'])

print("\n5. Running Model Inference for 3 Hours horizon...")
res_3h = run_gas_flow_inference(df_processed, horizon_name="3 Hours")
print("Forecast Result 3H:", res_3h['forecast_value'], "Current:", res_3h['current_value'], "Delta %:", res_3h['pct_change'])

print("\nPipeline execution completed successfully!")
