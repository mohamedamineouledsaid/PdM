"""
SCADA Industrial Gas Flow Forecasting Dashboard - Industrial Visualizations
Grafana, AVEVA PI Vision & Siemens WinCC Inspired Visualizations
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Industrial SCADA Theme Color Tokens
SCADA_BG = "#0b0f19"        # Deep Control Room Dark
PAPER_BG = "#131b2e"        # Card Container Dark Slate
GRID_COLOR = "#23314d"      # Subtle Metallic Grid Line
TEXT_COLOR = "#f1f5f9"      # Bright Off-White
TEXT_MUTED = "#64748b"      # Muted Slate

SENSOR_COLORS = {
    'DG': "#00e5ff",  # Electric Cyan (Gas Flow Rate)
    'CM': "#3b82f6",  # Royal Blue (Cumulative Production)
    'PT': "#00ff66",  # Lime Green (Wellhead Pressure)
    'PL': "#f59e0b",  # Industrial Amber (Line Pressure)
    'DP': "#ff3366",  # Neon Rose (Differential Pressure)
    'OF': "#a855f7",  # Vivid Purple (Valve Opening)
    'FORECAST': "#00ff66" # Bright Green
}


def generate_sparkline_svg(series, color="#00e5ff", width=100, height=26):
    """Generate SVG polyline sparkline for SCADA KPI cards."""
    try:
        arr = np.array(series, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) < 2:
            return ""
        
        arr = arr[-30:]
        min_v, max_v = arr.min(), arr.max()
        rng = (max_v - min_v) if max_v != min_v else 1.0

        points = []
        n = len(arr)
        for i, val in enumerate(arr):
            x = (i / (n - 1)) * (width - 4) + 2
            y = height - 2 - ((val - min_v) / rng) * (height - 6)
            points.append(f"{x:.1f},{y:.1f}")

        poly_str = " ".join(points)
        return f'''<svg width="{width}" height="{height}" style="overflow: visible;"><polyline fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" points="{poly_str}" /><circle cx="{points[-1].split(',')[0]}" cy="{points[-1].split(',')[1]}" r="3" fill="{color}" /></svg>'''
    except Exception:
        return ""


def generate_radial_confidence_gauge_svg(confidence_pct=95.4, size=110):
    """
    Generate circular donut radial progress gauge SVG for Model Confidence.
    """
    radius = 42
    circumference = 2 * np.pi * radius
    stroke_dasharray = circumference
    stroke_dashoffset = circumference - (confidence_pct / 100.0) * circumference

    return f'''<div style="text-align: center;"><svg width="{size}" height="{size}" viewBox="0 0 100 100"><circle cx="50" cy="50" r="{radius}" fill="none" stroke="#1e293b" stroke-width="9" /><circle cx="50" cy="50" r="{radius}" fill="none" stroke="url(#cyanLimeGradient)" stroke-width="9" stroke-dasharray="{stroke_dasharray:.1f}" stroke-dashoffset="{stroke_dashoffset:.1f}" stroke-linecap="round" transform="rotate(-90 50 50)" /><defs><linearGradient id="cyanLimeGradient" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#00e5ff" /><stop offset="100%" stop-color="#00ff66" /></linearGradient></defs><text x="50" y="47" text-anchor="middle" font-family="'JetBrains Mono', monospace" font-size="16" font-weight="800" fill="#00ff66">{confidence_pct:.1f}%</text><text x="50" y="62" text-anchor="middle" font-family="'JetBrains Mono', monospace" font-size="7" font-weight="700" fill="#64748b">HIGH CONF</text></svg></div>'''


def apply_scada_layout_theme(fig, title_text=""):
    """Apply WinCC / AVEVA industrial control room dark layout theme."""
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(family="JetBrains Mono, monospace, sans-serif", size=13, color=TEXT_COLOR),
            x=0.01,
            y=0.97
        ),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=SCADA_BG,
        margin=dict(l=45, r=45, t=50, b=40),
        font=dict(family="Inter, sans-serif", color=TEXT_COLOR),
        legend=dict(
            bgcolor="rgba(11, 15, 25, 0.85)",
            bordercolor=GRID_COLOR,
            borderwidth=1,
            font=dict(color=TEXT_COLOR, size=10, family="JetBrains Mono, monospace")
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=False,
            showline=True,
            linecolor=GRID_COLOR,
            tickfont=dict(color=TEXT_MUTED, size=10)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=False,
            showline=True,
            linecolor=GRID_COLOR,
            tickfont=dict(color=TEXT_MUTED, size=10)
        ),
        hovermode="x unified"
    )
    return fig


def create_multi_sensor_scada_chart(df_processed, lookback_minutes=720):
    """
    Create Grafana/Experion-style 6-sensor SCADA subplot grid.
    """
    if df_processed.empty:
        fig = go.Figure()
        return apply_scada_layout_theme(fig, "NO DATA AVAILABLE")

    df_sub = df_processed.tail(lookback_minutes).copy()
    time_series = pd.to_datetime(df_sub['time'])

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "GAS FLOW RATE (DG - M3/h)",
            "CUMULATIVE PRODUCTION (CM - M3)",
            "WELLHEAD PRESSURE (PT - bar)",
            "LINE PRESSURE (PL - bar)",
            "DIFFERENTIAL PRESSURE (DP - bar)",
            "VALVE OPENING (OF - %)"
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.07
    )

    # 1. DG
    if 'DG' in df_sub.columns:
        fig.add_trace(
            go.Scatter(
                x=time_series, y=df_sub['DG'],
                name="DG",
                line=dict(color=SENSOR_COLORS['DG'], width=2),
                hovertemplate="%{y:.2f} M3/h"
            ),
            row=1, col=1
        )

    # 2. CM
    if 'CM' in df_sub.columns:
        fig.add_trace(
            go.Scatter(
                x=time_series, y=df_sub['CM'],
                name="CM",
                line=dict(color=SENSOR_COLORS['CM'], width=2),
                hovertemplate="%{y:.0f} M3"
            ),
            row=1, col=2
        )

    # 3. PT
    if 'PT' in df_sub.columns:
        fig.add_trace(
            go.Scatter(
                x=time_series, y=df_sub['PT'],
                name="PT",
                line=dict(color=SENSOR_COLORS['PT'], width=2),
                hovertemplate="%{y:.2f} bar"
            ),
            row=2, col=1
        )

    # 4. PL
    if 'PL' in df_sub.columns:
        fig.add_trace(
            go.Scatter(
                x=time_series, y=df_sub['PL'],
                name="PL",
                line=dict(color=SENSOR_COLORS['PL'], width=2),
                hovertemplate="%{y:.2f} bar"
            ),
            row=2, col=2
        )

    # 5. DP
    if 'DP' in df_sub.columns:
        fig.add_trace(
            go.Scatter(
                x=time_series, y=df_sub['DP'],
                name="DP",
                line=dict(color=SENSOR_COLORS['DP'], width=2),
                hovertemplate="%{y:.2f} bar"
            ),
            row=3, col=1
        )

    # 6. OF
    if 'OF' in df_sub.columns:
        fig.add_trace(
            go.Scatter(
                x=time_series, y=df_sub['OF'],
                name="OF",
                line=dict(color=SENSOR_COLORS['OF'], width=2),
                hovertemplate="%{y:.1f} %"
            ),
            row=3, col=2
        )

    fig = apply_scada_layout_theme(fig, "HISTORICAL TRENDS — SCADA MULTI-SENSOR TELEMETRY (LAST 12 HOURS)")
    fig.update_layout(height=600, showlegend=False)

    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=10, color=TEXT_COLOR, family="JetBrains Mono, monospace")

    return fig


def create_forecast_timeline_chart(forecast_result):
    """
    Create AVEVA PI Vision / WinCC SCADA gas flow forecasting graph featuring:
      - Shaded Historical Region & Forecast Region
      - Shaded Confidence Cone (Uncertainty Band ±3.5%)
      - Solid Cyan Measured Line
      - Bright Green/Purple Dashed Forecast Line
      - Prominent Current Diamond Marker
      - Highlighted Glowing Forecast Target Marker
      - Vertical Separator Line at t0
    """
    hist_timestamps = pd.to_datetime(forecast_result['historical_timestamps'])
    hist_dg = forecast_result['historical_dg']

    curr_time = pd.to_datetime(forecast_result['current_timestamp'])
    curr_val = forecast_result['current_value']

    fc_time = pd.to_datetime(forecast_result['forecast_timestamp'])
    fc_val = forecast_result['forecast_value']

    pct_change = forecast_result['pct_change']
    horizon_name = forecast_result['horizon_name']

    # Trajectory color coding (Green: Increase, Yellow: Stable, Red: Decline)
    if pct_change >= 1.0:
        fc_color = "#00ff66"  # Bright Green
    elif pct_change <= -1.0:
        fc_color = "#ff3366"  # Red
    else:
        fc_color = "#a855f7"  # Vivid Purple / Forecast Accent

    y_min = min(hist_dg.min(), fc_val) * 0.95
    y_max = max(hist_dg.max(), fc_val) * 1.05

    fig = go.Figure()

    # 1. Shaded Historical Region
    fig.add_vrect(
        x0=hist_timestamps[0],
        x1=curr_time,
        fillcolor="rgba(0, 229, 255, 0.03)",
        layer="below",
        line_width=0
    )

    # 2. Shaded Forecast Region
    fig.add_vrect(
        x0=curr_time,
        x1=fc_time,
        fillcolor="rgba(0, 255, 102, 0.05)",
        layer="below",
        line_width=0
    )

    # 3. Confidence Cone (Uncertainty Band ±3.5%)
    upper_bound = fc_val * 1.035
    lower_bound = fc_val * 0.965
    fig.add_trace(
        go.Scatter(
            x=[curr_time, fc_time, fc_time, curr_time],
            y=[curr_val, upper_bound, lower_bound, curr_val],
            fill='toself',
            fillcolor='rgba(0, 255, 102, 0.10)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Cone (±3.5%)',
            showlegend=True,
            hoverinfo="skip"
        )
    )

    # 4. Historical Measured Line
    fig.add_trace(
        go.Scatter(
            x=hist_timestamps,
            y=hist_dg,
            mode="lines",
            name="Measured",
            line=dict(color="#00e5ff", width=2.5),
            hovertemplate="%{x|%H:%M}: %{y:.2f} M3/h"
        )
    )

    # 5. Current Observation Marker (t0)
    fig.add_trace(
        go.Scatter(
            x=[curr_time],
            y=[curr_val],
            mode="markers+text",
            name="Current",
            marker=dict(color="#f59e0b", size=13, symbol="diamond", line=dict(color="#ffffff", width=1.5)),
            text=[f"t0: {curr_val:.1f}"],
            textposition="top left",
            textfont=dict(color="#f59e0b", size=11, family="JetBrains Mono, monospace"),
            hovertemplate="Current DG (t0): %{y:.2f} M3/h"
        )
    )

    # 6. Forecast Trajectory Line
    fig.add_trace(
        go.Scatter(
            x=[curr_time, fc_time],
            y=[curr_val, fc_val],
            mode="lines",
            name="Forecast",
            line=dict(color=fc_color, width=3.5, dash="dot"),
            showlegend=True
        )
    )

    # 7. Highlighted Forecast Target Marker
    fig.add_trace(
        go.Scatter(
            x=[fc_time],
            y=[fc_val],
            mode="markers+text",
            name="Predicted DG",
            marker=dict(color=fc_color, size=16, symbol="circle", line=dict(color="#ffffff", width=2.5)),
            text=[f"Predicted: {fc_val:.1f} M3/h"],
            textposition="top right",
            textfont=dict(color=fc_color, size=12, family="JetBrains Mono, monospace"),
            hovertemplate=f"Predicted DG (+{horizon_name}): %{{y:.2f}} M3/h"
        )
    )

    # 8. Vertical Boundary Line at t0
    fig.add_trace(
        go.Scatter(
            x=[curr_time, curr_time],
            y=[y_min, y_max],
            mode="lines",
            name="t0 Boundary",
            line=dict(color="#64748b", width=1.5, dash="dash"),
            showlegend=False,
            hoverinfo="skip"
        )
    )

    fig = apply_scada_layout_theme(
        fig,
        f"FORECAST TIMELINE — {horizon_name.upper()} HORIZON (TARGET: {fc_val:.2f} M3/h)"
    )
    fig.update_layout(height=440)

    return fig
