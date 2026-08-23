import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta

def render_charts(df_raw, time_window_hours):
    st.markdown("### Sensor Trends")
    if df_raw.empty:
        return
        
    df_raw = df_raw.sort_values('time')
    
    # Filter by selected time window
    latest_time = df_raw['time'].max()
    start_time = latest_time - timedelta(hours=time_window_hours)
    df_filtered = df_raw[df_raw['time'] >= start_time]
    
    # Two columns for charts
    col1, col2 = st.columns(2)
    
    # Desired order to show important ones first
    target_sensors = ['DG', 'PT', 'PL', 'DP', 'OF', 'CM']
    sensors_present = df_filtered['sensor'].unique()
    sensors = [s for s in target_sensors if s in sensors_present]
    
    for i, sensor in enumerate(sensors):
        df_sensor = df_filtered[df_filtered['sensor'] == sensor]
        
        # Color logic based on sensor type for aesthetics
        color = '#60a5fa'
        if sensor == 'PT': color = '#f43f5e'
        elif sensor == 'DP': color = '#facc15'
        elif sensor == 'PL': color = '#a78bfa'
        elif sensor == 'OF': color = '#10b981'
        elif sensor == 'CM': color = '#fb923c'
        
        # Unit logic
        unit = ""
        if sensor == 'PT': unit = " (Bar)"
        elif sensor == 'DP': unit = " (mBar)"
        elif sensor == 'PL': unit = " (Bar)"
        elif sensor == 'OF': unit = " (%)"
        elif sensor == 'CM': unit = " (mm)"
        elif sensor == 'DG': unit = " (m³/h)"
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_sensor['time'], 
            y=df_sensor['value'],
            mode='lines',
            line=dict(width=2, color=color),
            fill='tozeroy',
            fillcolor=f'rgba({int(color[1:3],16)}, {int(color[3:5],16)}, {int(color[5:7],16)}, 0.1)'
        ))
        
        fig.update_layout(
            title=dict(text=f"<b>{sensor} Trend</b>", font=dict(color='#f8fafc', size=16)),
            height=340,
            margin=dict(l=60, r=20, t=50, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'),
            hovermode="x unified",
            xaxis=dict(
                title="Timestamp",
                showgrid=True, 
                gridcolor='rgba(255,255,255,0.05)', 
                linecolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                title=dict(text=f"<b>{sensor} {unit}</b>", font=dict(color='#e5e7eb', size=13)),
                showgrid=True, 
                gridcolor='rgba(255,255,255,0.05)', 
                linecolor='rgba(255,255,255,0.1)'
            )
        )
        
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            st.plotly_chart(fig, use_container_width=True)
