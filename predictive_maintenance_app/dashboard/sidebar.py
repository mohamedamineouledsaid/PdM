import streamlit as st

from backend.config import WELLS
from backend.database import get_engine


import os
import base64

def get_db_status():
    engine = get_engine()
    if engine:
        try:
            with engine.connect():
                return True
        except Exception:
            return False
    return False

def get_base64_of_bin_file(bin_file):
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

def render_sidebar():

    # SONATRACH HEADER - SCADA STYLE
    logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "sonatrach_logo.png")
    logo_base64 = get_base64_of_bin_file(logo_path)
    
    if logo_base64:
        img_html = f'<img src="data:image/png;base64,{logo_base64}" style="width: 200px; display: block; margin: 0 auto; margin-bottom: 5px;">'
    else:
        # Fallback if image not found
        img_html = f'<div style="width: 200px; height: 60px; border: 1px solid #E2E8F0; margin: 0 auto; display: flex; align-items: center; justify-content: center; margin-bottom: 5px; color: #94a3b8; border-radius: 4px;">SONATRACH LOGO</div>'

    st.sidebar.markdown(
        f"""
        <div style="text-align: center; margin-top: -30px; margin-bottom: 15px;">
            {img_html}
            <h2 style="margin-bottom: 0px; margin-top: 5px; color: #1E293B; font-size: 1.4rem; font-weight: 700; letter-spacing: 0.5px;">
                Predictive Maintenance System
            </h2>
            <p style="margin-top: 2px; margin-bottom: 0px; color: #64748B; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px;">
                Hassi R'Mel Gas Wells
            </p>
        </div>
        <hr style="margin-top: 0px; margin-bottom: 15px; border: 0; border-top: 1px solid #e2e8f0;">
        """,
        unsafe_allow_html=True,
    )

    # WELL SELECTION
    st.sidebar.markdown("### Monitored Well")

    selected_well = st.sidebar.selectbox(
        "Select Well",
        WELLS,
        label_visibility="collapsed",
    )

    refresh_options = {
        "1 Minute": 60,
        "2 Minutes": 120,
        "5 Minutes": 300,
        "10 Minutes": 600,
        "Never (Disabled)": 9999999
    }
    
    current_interval = st.session_state.get('refresh_interval', 120)
    refresh_keys = list(refresh_options.keys())
    refresh_vals = list(refresh_options.values())
    default_index = refresh_vals.index(current_interval) if current_interval in refresh_vals else 1

    refresh_label = st.sidebar.selectbox(
        "Auto-Refresh Interval",
        refresh_keys,
        index=default_index,
    )
    st.session_state.refresh_interval = refresh_options[refresh_label]

    window_options = {
        "Last 6 Hours": 6,
        "Last 12 Hours": 12,
        "Last 24 Hours": 24,
    }

    window_label = st.sidebar.selectbox(
        "History Window",
        list(window_options.keys()),
        index=0,
    )
    st.session_state.time_window = window_options[window_label]

    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    if st.sidebar.button("🔄 Manual Refresh", use_container_width=True):
        st.rerun()

    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    # SYSTEM STATUS
    st.sidebar.markdown("### System Status")

    db_ok = get_db_status()
    model_ok = ("model" in st.session_state and bool(st.session_state.model.models) and st.session_state.model.scaler is not None)

    def render_status_indicator(is_ok):
        indicator_style = "" if is_ok else "background-color: #ef4444; box-shadow: 0 0 6px rgba(239, 68, 68, 0.5);"
        return f'<div class="scada-status-indicator" style="{indicator_style}"></div>'

    st.sidebar.markdown(
        f"""
        <div class="scada-status-widget">
            <div class="scada-status-row">
                <span class="scada-status-text">PostgreSQL Connected</span>
                {render_status_indicator(db_ok)}
            </div>
            <div class="scada-status-row">
                <span class="scada-status-text">AI Model Active</span>
                {render_status_indicator(model_ok)}
            </div>
            <div class="scada-status-row">
                <span class="scada-status-text">Data Stream Live</span>
                {render_status_indicator(db_ok)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    if st.sidebar.button(
        "📄 Export Daily Report",
        use_container_width=True,
    ):
        st.sidebar.info("Report generation not implemented yet.")

    st.sidebar.markdown(
        """
        <div style="text-align:center; color:#64748B; font-size:12px; margin-top: 20px;">
            Sonatrach Predictive Maintenance Platform<br>
            Version 2.0
        </div>
        """,
        unsafe_allow_html=True,
    )

    return selected_well