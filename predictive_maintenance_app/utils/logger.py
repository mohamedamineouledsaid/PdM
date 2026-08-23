import logging
import streamlit as st
from datetime import datetime

# Configure standard Python logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("PMS_App")

def log_event(message, level="INFO"):
    """
    Log an event to the session state so it can be displayed in the UI,
    and to the standard Python logging system for production monitoring.
    """
    # System logging
    if level == "INFO":
        logger.info(message)
    elif level == "WARNING":
        logger.warning(message)
    elif level == "ERROR":
        logger.error(message)
    elif level == "SUCCESS":
        logger.info(f"SUCCESS: {message}")
    else:
        logger.debug(message)

    # UI logging
    if "system_logs" not in st.session_state:
        st.session_state.system_logs = []
        
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "level": level,
        "message": message
    }
    
    # Add to beginning and keep last 100
    st.session_state.system_logs.insert(0, log_entry)
    if len(st.session_state.system_logs) > 100:
        st.session_state.system_logs.pop()

def get_logs():
    if "system_logs" not in st.session_state:
        return []
    return st.session_state.system_logs
