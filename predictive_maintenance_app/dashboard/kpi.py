import streamlit as st
import pandas as pd

def render_kpi_cards(df_raw):
    st.markdown("### Real-Time KPI Telemetry")
    if df_raw.empty:
        st.warning("No data available for the selected well.")
        return
        
    # Get the latest value for each sensor and the previous value to compute trend
    df_raw_sorted = df_raw.sort_values('time')
    
    latest_data = df_raw_sorted.groupby('sensor').tail(1).set_index('sensor')
    
    # Try to get data from ~5 minutes ago for trend calculation
    # Since resample is not applied here, we just take the row index - 5 (assuming 1min frequency roughly)
    # or just the previous row per sensor
    prev_data = df_raw_sorted.groupby('sensor').nth(-5).set_index('sensor')
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    sensors = [
        ('DG', 'Gas Flow Rate', 'kSm³/d', col1),
        ('PT', 'Wellhead Pressure', 'bar', col2),
        ('PL', 'Line Pressure', 'bar', col3),
        ('DP', 'Diff. Pressure', 'bar', col4),
        ('OF', 'Valve Opening', '%', col5),
        ('CM', 'Cum. Production', 'm³', col6)
    ]
    
    for sensor_code, sensor_name, unit, col in sensors:
        with col:
            val = latest_data.loc[sensor_code, 'value'] if sensor_code in latest_data.index else None
            prev_val = prev_data.loc[sensor_code, 'value'] if sensor_code in prev_data.index else None
            
            val_str = "N/A"
            trend_html = ""
            
            if val is not None:
                val_str = f"{val:.2f}"
                
                if prev_val is not None:
                    diff = val - prev_val
                    if abs(diff) < 0.01:
                        trend_html = "<span class='kpi-trend trend-flat'>→ 0.00</span>"
                    elif diff > 0:
                        trend_html = f"<span class='kpi-trend trend-up'>↑ {diff:.2f}</span>"
                    else:
                        trend_html = f"<span class='kpi-trend trend-down'>↓ {abs(diff):.2f}</span>"
                        
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">{sensor_name}</div>
                    <div class="kpi-value">{val_str} <span style="font-size:1rem;color:#94a3b8;">{unit}</span></div>
                    <div>{trend_html}</div>
                </div>
            """, unsafe_allow_html=True)
