import streamlit as st

def render_reports():
    st.markdown("### 📄 Reports & Analytics")
    st.info("Report generation and advanced analytics are currently under development.")
    st.markdown("""
        Future features will include:
        - Daily PDF Export of SCADA alarms
        - Weekly predictive maintenance summaries
        - Sensor drift analysis
    """)

def render_settings():
    st.markdown("### 🛠️ Platform Settings")
    st.info("Platform settings are managed by the SCADA Administrator.")
    st.markdown("""
        - **Database Configuration**: Validated
        - **AI Engine Endpoints**: Active
        - **Notification Webhooks**: Disabled
    """)
