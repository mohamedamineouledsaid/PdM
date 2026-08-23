import streamlit as st
import time

# Set Page Config MUST be the very first Streamlit command
st.set_page_config(
    page_title="Sonatrach PMS",
    page_icon="assets/sonatrach_favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

from utils.helpers import inject_custom_css
from utils.logger import log_event
from dashboard.sidebar import render_sidebar
from dashboard.pages.main_dashboard import main_dashboard
from dashboard.pages.placeholders import render_reports, render_settings
from utils.predictions import log_predictions_to_history

from backend.database import fetch_recent_data
from backend.inference import PredictiveMaintenanceModel

@st.cache_resource(show_spinner="Initializing AI Models and Scaler...")
def load_ai_model():
    return PredictiveMaintenanceModel()

def init_session_state():
    if 'model' not in st.session_state:
        st.session_state.model = load_ai_model()
        log_event("AI Models initialized successfully", "SUCCESS")
            
    if 'prediction_history' not in st.session_state:
        import pandas as pd
        st.session_state.prediction_history = pd.DataFrame(columns=[
            "Timestamp", "Prediction Horizon", "Predicted Risk", 
            "Risk Level", "Confidence Score", "Status"
        ])
            
    if 'system_logs' not in st.session_state:
        st.session_state.system_logs = []
        log_event("SCADA AI Operations Platform started", "INFO")
        
    if 'refresh_interval' not in st.session_state:
        from backend.config import DEFAULT_REFRESH_INTERVAL
        st.session_state.refresh_interval = DEFAULT_REFRESH_INTERVAL
        
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
        
    if 'time_window' not in st.session_state:
        st.session_state.time_window = 24
        
    if 'sim_offset' not in st.session_state:
        st.session_state.sim_offset = 1000

def main():
    # Inject Custom CSS for Glassmorphism & SCADA aesthetic
    inject_custom_css()
    
    # Initialize Session
    init_session_state()
    
    # Render Sidebar and get selections
    selected_well = render_sidebar()

    # Fetch Data for predictive modules
    with st.spinner("Connecting to PostgreSQL and fetching telemetry..."):
        fetch_hours = max(48, st.session_state.time_window) 
        df_raw = fetch_recent_data(selected_well, hours=fetch_hours, offset=st.session_state.sim_offset)
        
    if not df_raw.empty:
        # Run prediction for all models
        probs = None
        error = None
        seq_time = None
        with st.spinner("AI Diagnostics in progress..."):
            probs, seq_time, error = st.session_state.model.process_and_predict(df_raw)
            
            # Save predictions to history (centralized)
            log_predictions_to_history(seq_time, probs, error)
            
        # Render the unified dashboard
        main_dashboard(selected_well, df_raw, probs, error, st.session_state.time_window, seq_time)
    else:
        st.error(f"No recent data found for {selected_well}. Please check the database connection.")
        log_event(f"Failed to fetch data for {selected_well}", "ERROR")
        
    # Auto-refresh loop without blocking
    current_time = time.time()
    
    # Process simulation step if time elapsed
    if current_time - st.session_state.last_refresh >= st.session_state.refresh_interval:
        st.session_state.last_refresh = current_time
        if st.session_state.sim_offset > 0:
            st.session_state.sim_offset -= 1
            
    # Schedule the next rerun using the official streamlit_autorefresh package
    if st.session_state.refresh_interval < 9999999:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=st.session_state.refresh_interval * 1000, key="data_refresh")

if __name__ == "__main__":
    main()
