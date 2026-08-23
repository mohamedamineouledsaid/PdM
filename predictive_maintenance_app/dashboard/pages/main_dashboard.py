import streamlit as st
from dashboard.header import render_header
from dashboard.kpi import render_kpi_cards
from dashboard.charts import render_charts
from dashboard.history_table import render_history_table
from backend.inference import BEST_THRESHOLD
from utils.predictions import calculate_risk_and_confidence

def render_prediction_cards(probs):
    st.markdown("### AI Prediction Overview")
    
    col1, col2, col3 = st.columns(3)
    
    horizons = [('6h', '6 Hours', col1), ('12h', '12 Hours', col2), ('24h', '24 Hours', col3)]
    
    for key, title, col in horizons:
        prob = probs.get(key)
        
        if prob is None:
            with col:
                st.warning(f"No prediction available for {title}")
            continue
            
        # Determine risk level and color using centralized logic
        metrics = calculate_risk_and_confidence(prob)
        risk = metrics['risk_label']
        color = metrics['color_hex']
        border = metrics['border_rgba']
        confidence = metrics['confidence']
        
        card_html = f"""<div style="
background: linear-gradient(145deg, #1e293b, #0f172a);
border: 1px solid {border};
border-left: 4px solid {color};
border-radius: 8px;
padding: 20px;
box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
height: 100%;
">
<h4 style="color: #94a3b8; margin-top: 0; margin-bottom: 10px; font-weight: 500;">Maintenance Prediction</h4>
<div style="font-size: 1.1rem; color: #cbd5e1; margin-bottom: 15px;">Horizon: <strong style="color: #f8fafc;">{title}</strong></div>

<div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 10px;">
<span style="font-size: 1rem; color: #94a3b8;">Failure Risk:</span>
<span style="font-size: 2rem; font-weight: bold; color: {color};">{prob*100:.1f}%</span>
</div>

<div style="background: rgba(0,0,0,0.2); border-radius: 4px; height: 8px; width: 100%; margin-bottom: 15px; overflow: hidden;">
<div style="background: {color}; height: 100%; width: {prob*100}%;"></div>
</div>

<div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.05);">
<span style="font-size: 0.9rem; color: #94a3b8;">Confidence:</span>
<span style="font-size: 0.9rem; font-weight: 600; color: #cbd5e1;">{confidence:.1f}%</span>
</div>

<div style="text-align: center; font-weight: bold; letter-spacing: 1px; color: {color}; font-size: 1.1rem;">
{risk}
</div>
</div>"""
        
        with col:
            st.markdown(card_html, unsafe_allow_html=True)
            
    st.markdown("<br>", unsafe_allow_html=True)

def main_dashboard(selected_well, df_raw, probs, error, time_window, current_data_timestamp=None):
    if current_data_timestamp is None and not df_raw.empty:
        current_data_timestamp = df_raw['time'].max().strftime("%Y-%m-%d %H:%M:%S")
        
    # Render Header
    render_header(selected_well, current_data_timestamp)
    
    if not df_raw.empty:
        # Layout components
        render_kpi_cards(df_raw)
        
        if error:
            st.error(f"Prediction Error: {error}")
        elif probs:
            render_prediction_cards(probs)
            
        render_charts(df_raw, time_window)
        
        # Render Prediction History at the bottom
        render_history_table()
    else:
        st.error(f"No recent data found for {selected_well}. Please check the database connection.")
