import streamlit as st
import pandas as pd

def render_history():
    st.markdown("### Prediction History")
    
    if "prediction_history" not in st.session_state or not st.session_state.prediction_history:
        st.info("No prediction history available yet.")
        return
        
    df_hist = pd.DataFrame(st.session_state.prediction_history)
    
    # Style the dataframe for SCADA UI
    def color_status(val):
        color = '#10b981' if val == 'NORMAL' else '#f59e0b' if val == 'WARNING' else '#ef4444'
        return f'color: {color}; font-weight: bold;'
        
    styled_df = df_hist.style.applymap(color_status, subset=['Decision'])
    
    # Custom styling for dataframe
    st.dataframe(styled_df, use_container_width=True, height=250)
