import streamlit as st
from utils.logger import get_logs

def render_logs():
    st.markdown("### System Logs")
    
    logs = get_logs()
    
    if not logs:
        st.markdown("<div class='log-container'>No logs available...</div>", unsafe_allow_html=True)
        return
        
    log_html = "<div class='log-container'>"
    
    for log in logs:
        time_str = f"[{log['timestamp']}]"
        level_class = f"log-{log['level'].lower()}"
        log_html += f"<div class='log-entry'><span style='color: #64748b;'>{time_str}</span> <span class='{level_class}'>[{log['level']}]</span> {log['message']}</div>"
        
    log_html += "</div>"
    
    st.markdown(log_html, unsafe_allow_html=True)
