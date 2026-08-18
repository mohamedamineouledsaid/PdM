"""
SCADA Industrial Gas Flow Forecasting Dashboard - Utility & AI Interpretation Module
"""

import pandas as pd
import datetime
import os


def get_ai_production_interpretation(current_dg, forecast_dg, pct_change):
    """
    Generate industrial production interpretation based on forecast change.
    Categories:
      - Stable production (-1% <= pct <= +1%)
      - Slight production increase (+1% < pct < +5%)
      - Significant production recovery (pct >= +5%)
      - Slight production decline (-5% < pct < -1%)
      - Moderate production decline (-15% < pct <= -5%)
      - Severe production loss (pct <= -15%)
    """
    abs_diff = forecast_dg - current_dg

    if -1.0 <= pct_change <= 1.0:
        status = "Stable Production"
        category = "STABLE"
        color = "#10b981"  # Emerald Green
        icon = "🟢"
        explanation = (
            f"The gas flow rate is forecasted to remain stable at **{forecast_dg:.2f} M3/h** "
            f"(Delta: {abs_diff:+.2f} M3/h, {pct_change:+.2f}%). Wellhead thermal-pressure dynamics "
            f"and choke opening indicate steady-state operating conditions over the forecast window."
        )
    elif 1.0 < pct_change < 5.0:
        status = "Slight Production Increase"
        category = "INCREASE"
        color = "#06b6d4"  # Cyan
        icon = "📈"
        explanation = (
            f"Gas production is predicted to undergo a minor upward trend to **{forecast_dg:.2f} M3/h** "
            f"({abs_diff:+.2f} M3/h, {pct_change:+.2f}%). Favorable pressure differential (DP/PT ratio) "
            f"indicates improved reservoir flow efficiency."
        )
    elif pct_change >= 5.0:
        status = "Significant Production Recovery"
        category = "RECOVERY"
        color = "#3b82f6"  # Bright Blue
        icon = "🚀"
        explanation = (
            f"Strong positive flow recovery expected, reaching **{forecast_dg:.2f} M3/h** "
            f"({abs_diff:+.2f} M3/h, {pct_change:+.2f}%). Post-intervention stabilization or line pressure "
            f"de-bottlenecking is significantly increasing deliverability."
        )
    elif -5.0 < pct_change < -1.0:
        status = "Slight Production Decline"
        category = "SLIGHT_DECLINE"
        color = "#f59e0b"  # Amber / Yellow
        icon = "📉"
        explanation = (
            f"Minor flow reduction anticipated to **{forecast_dg:.2f} M3/h** "
            f"({abs_diff:+.2f} M3/h, {pct_change:+.2f}%). Natural reservoir pressure decline or minor line backpressure "
            f"is causing minor flow attenuation."
        )
    elif -15.0 < pct_change <= -5.0:
        status = "Moderate Production Decline"
        category = "MODERATE_DECLINE"
        color = "#f97316"  # Orange
        icon = "⚠️"
        explanation = (
            f"Noticeable flow decline expected, dropping to **{forecast_dg:.2f} M3/h** "
            f"({abs_diff:+.2f} M3/h, {pct_change:+.2f}%). Wellhead pressure depletion or choke turbulence "
            f"requires close monitoring by field operators."
        )
    else: # pct_change <= -15.0
        status = "Severe Production Loss"
        category = "SEVERE_LOSS"
        color = "#ef4444"  # Red
        icon = "🚨"
        explanation = (
            f"CRITICAL WARNING: Significant flow rate drop forecasted down to **{forecast_dg:.2f} M3/h** "
            f"({abs_diff:+.2f} M3/h, {pct_change:+.2f}%). Potential liquid loading, hydration, or valve restriction "
            f"detected. Immediate engineering review recommended."
        )

    return {
        "status": status,
        "category": category,
        "color": color,
        "icon": icon,
        "explanation": explanation
    }


_FORECAST_HISTORY_FILE = os.path.join("backend", "forecast_history.csv")


def append_forecast_history(timestamp, well, horizon, current_dg, predicted_dg):
    """
    Append new prediction entry to persistent forecast history log table.
    """
    diff = predicted_dg - current_dg
    pct = (diff / (current_dg + 1e-5)) * 100.0

    new_entry = pd.DataFrame([{
        "Timestamp": pd.to_datetime(timestamp).strftime("%Y-%m-%d %H:%M:%S"),
        "Well": well,
        "Forecast Horizon": horizon,
        "Current DG (M3/h)": round(current_dg, 2),
        "Predicted DG (M3/h)": round(predicted_dg, 2),
        "Absolute Difference": round(diff, 2),
        "Percentage Change (%)": f"{pct:+.2f}%"
    }])

    if os.path.exists(_FORECAST_HISTORY_FILE):
        try:
            history_df = pd.read_csv(_FORECAST_HISTORY_FILE)
            history_df = pd.concat([new_entry, history_df], ignore_index=True)
            # Retain up to 200 recent entries
            history_df = history_df.head(200)
            history_df.to_csv(_FORECAST_HISTORY_FILE, index=False)
        except Exception:
            new_entry.to_csv(_FORECAST_HISTORY_FILE, index=False)
    else:
        new_entry.to_csv(_FORECAST_HISTORY_FILE, index=False)


def get_forecast_history():
    """
    Retrieve forecast history table.
    """
    if os.path.exists(_FORECAST_HISTORY_FILE):
        try:
            return pd.read_csv(_FORECAST_HISTORY_FILE)
        except Exception:
            pass

    return pd.DataFrame(columns=[
        "Timestamp", "Well", "Forecast Horizon",
        "Current DG (M3/h)", "Predicted DG (M3/h)",
        "Absolute Difference", "Percentage Change (%)"
    ])
