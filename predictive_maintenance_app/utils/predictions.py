import pandas as pd
import streamlit as st
from backend.inference import BEST_THRESHOLD
from utils.logger import log_event

def calculate_risk_and_confidence(prob):
    """
    Standardized logic for determining risk levels, UI colors, and confidence scores.
    This replaces duplicated logic across main_dashboard.py and ai_panel.py.
    """
    if prob < 0.40:
        risk_label = "LOW RISK"
        table_risk = "Low"
        color_hex = "#10b981" # Green
        border_rgba = "rgba(16, 185, 129, 0.3)"
        icon = "🟢"
    elif prob < 0.70:
        risk_label = "MEDIUM RISK"
        table_risk = "Medium"
        color_hex = "#f59e0b" # Yellow
        border_rgba = "rgba(245, 158, 11, 0.3)"
        icon = "🟡"
    else:
        risk_label = "HIGH RISK"
        table_risk = "High"
        color_hex = "#ef4444" # Red
        border_rgba = "rgba(239, 68, 68, 0.4)"
        icon = "🔴"
        
    confidence = 100.0 - (prob * 10 if prob < 0.5 else (1.0 - prob) * 10)
    
    # Calculate confidence score distance for AI panel (kept for backward compatibility with AI panel visuals)
    confidence_score_distance = abs(prob - BEST_THRESHOLD) * 200
    if confidence_score_distance > 99: confidence_score_distance = 99.9
    if confidence_score_distance < 10: confidence_score_distance = 75.0 + confidence_score_distance

    return {
        "risk_label": risk_label,
        "table_risk": table_risk,
        "color_hex": color_hex,
        "border_rgba": border_rgba,
        "icon": icon,
        "confidence": confidence,
        "confidence_score_distance": confidence_score_distance
    }

def log_predictions_to_history(timestamp, probs, error=None):
    """
    Centralized function to log predictions.
    Enforces standard schema, prevents duplicate timestamps, 
    and bounds memory growth to 100 records.
    """
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = pd.DataFrame(columns=[
            "Timestamp", "Prediction Horizon", "Predicted Risk", 
            "Risk Level", "Confidence Score", "Status"
        ])
    
    if error or not probs:
        return
        
    df = st.session_state.prediction_history
    
    # Avoid duplicate logic: do not add if timestamp already exists
    if not df.empty and (df['Timestamp'] == timestamp).any():
        return
        
    new_rows = []
    
    for horizon, prob in probs.items():
        metrics = calculate_risk_and_confidence(prob)
            
        new_rows.append({
            "Timestamp": timestamp,
            "Prediction Horizon": horizon,
            "Predicted Risk": round(prob * 100, 1),
            "Risk Level": metrics["table_risk"],
            "Confidence Score": round(metrics['confidence'], 1),
            "Status": "Success"
        })
        
        # Log backend event
        log_event(
            f"Prediction generated: {metrics['table_risk']} ({prob*100:.1f}%) for {horizon} horizon", 
            level="SUCCESS" if metrics['table_risk']=="Low" else "WARNING" if metrics['table_risk']=="Medium" else "ERROR"
        )
        
    new_df = pd.DataFrame(new_rows)
    
    # Prepend new rows and keep only the latest 100
    df = pd.concat([new_df, df], ignore_index=True)
    if len(df) > 100:
        df = df.head(100)
        
    st.session_state.prediction_history = df
