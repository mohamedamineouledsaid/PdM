import streamlit as st
from datetime import datetime
from backend.inference import BEST_THRESHOLD
from utils.logger import log_event
from utils.predictions import calculate_risk_and_confidence

def render_ai_panel(prob, error, well, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
    st.markdown("### AI Diagnostic Engine")
    
    st.markdown("<div class='ai-panel'>", unsafe_allow_html=True)
    
    if error:
        st.error(f"Prediction Error: {error}")
        log_event(f"AI Prediction Error for {well}: {error}", level="ERROR")
        st.markdown("</div>", unsafe_allow_html=True)
        return
        
    # Use centralized logic for risk, colors, and confidence
    metrics = calculate_risk_and_confidence(prob)
    risk_level = metrics['risk_label'].replace(" RISK", "") # AI panel displays just NORMAL/WARNING/CRITICAL
    if risk_level == "LOW": risk_level = "NORMAL"
    elif risk_level == "MEDIUM": risk_level = "WARNING"
    
    color_hex = metrics['color_hex']
    icon = metrics['icon']
    confidence_score = metrics['confidence_score_distance']
    
    # Recommendations based on table risk
    if metrics['table_risk'] == "Low":
        recommendations = [
            "Continue monitoring.",
            "Normal operations confirmed."
        ]
    elif metrics['table_risk'] == "Medium":
        recommendations = [
            "Inspect the well.",
            "Monitor pressure behavior.",
            "Check for potential valve drift."
        ]
    else:
        recommendations = [
            "Immediate inspection recommended.",
            "Schedule maintenance.",
            "Verify pressure system."
        ]
    
    col1, col2, col3 = st.columns([1, 2, 2])
    
    with col1:
        st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <div style="font-size: 5rem; line-height: 1; animation: {'pulse 1.5s infinite' if risk_level == 'CRITICAL' else 'none'};">{icon}</div>
                <h2 style="color: {color_hex}; margin-top: 10px;">{risk_level}</h2>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
            <h4 style="color: #94a3b8; border-bottom: 1px solid #334155; padding-bottom: 10px;">Model Output</h4>
            <div style="margin-top: 15px;">
                <p style="font-size: 1.1rem; color: #cbd5e1; margin-bottom: 5px;">Prediction Probability: 
                    <strong style="font-size: 1.8rem; color: #60a5fa; float: right;">{prob*100:.1f}%</strong>
                </p>
                <div style="background: #1e293b; height: 10px; border-radius: 5px; width: 100%; margin-bottom: 15px;">
                    <div style="background: {color_hex}; height: 10px; border-radius: 5px; width: {prob*100}%;"></div>
                </div>
                <p style="font-size: 1.1rem; color: #cbd5e1; margin-bottom: 5px;">Decision Threshold: 
                    <strong style="font-size: 1.2rem; color: #f8fafc; float: right;">{BEST_THRESHOLD*100:.0f}%</strong>
                </p>
                <p style="font-size: 1.1rem; color: #cbd5e1; margin-bottom: 5px;">Confidence Score: 
                    <strong style="font-size: 1.2rem; color: #f8fafc; float: right;">{confidence_score:.1f}%</strong>
                </p>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        rec_html = "".join([f"<li style='margin-bottom: 8px;'>{r}</li>" for r in recommendations])
        st.markdown(f"""
            <h4 style="color: #94a3b8; border-bottom: 1px solid #334155; padding-bottom: 10px;">Maintenance Recommendation</h4>
            <div style="background: rgba({int(color_hex[1:3], 16)}, {int(color_hex[3:5], 16)}, {int(color_hex[5:7], 16)}, 0.1); 
                        border-left: 4px solid {color_hex}; 
                        padding: 15px; 
                        border-radius: 4px; 
                        margin-top: 15px;
                        height: 100%;">
                <ul style="font-size: 1.1rem; color: #e2e8f0; padding-left: 20px;">
                    {rec_html}
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    # History saving and event logging is now handled centrally in app.py
