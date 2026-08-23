import streamlit as st
from datetime import datetime

def render_header(selected_well, current_data_timestamp=None):
    if current_data_timestamp:
        try:
            dt = datetime.strptime(current_data_timestamp, "%Y-%m-%d %H:%M:%S")
            date_str = dt.strftime('%Y-%m-%d')
            time_str = dt.strftime('%H:%M:%S')
        except Exception:
            date_str = current_data_timestamp
            time_str = ""
    else:
        date_str = "N/A"
        time_str = ""
        
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("Predictive Maintenance System")
        st.markdown(f"<h3 style='color: #94a3b8;'>Hassi R'Mel Gas Wells - {selected_well}</h3>", unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
            <div style="text-align: right; padding-top: 20px;">
                <div style="font-size: 1.2rem; color: #f8fafc;">{date_str}</div>
                <div style="font-size: 2rem; font-weight: bold; color: #60a5fa; font-family: monospace;">{time_str}</div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("---")
