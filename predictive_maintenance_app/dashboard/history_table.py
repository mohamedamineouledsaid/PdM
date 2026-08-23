import streamlit as st
import pandas as pd

def render_history_table():
    st.markdown("<br><hr style='border-top: 1px solid rgba(255,255,255,0.1);'><br>", unsafe_allow_html=True)
    st.markdown("### Prediction History")
    
    df = st.session_state.get('prediction_history', pd.DataFrame())
    
    if df.empty:
        st.info("No predictions recorded yet. Waiting for AI inference...")
        return
        
    # Filters Layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        horizons = ["All"] + sorted(list(df['Prediction Horizon'].unique()))
        selected_horizon = st.selectbox("Filter Horizon", horizons)
        
    with col2:
        risks = ["All", "Low", "Medium", "High"]
        selected_risk = st.selectbox("Filter Risk Level", risks)
        
    with col3:
        sort_order = st.selectbox("Sort By Date", ["Newest First", "Oldest First"])
        
    # Apply filters
    filtered_df = df.copy()
    if selected_horizon != "All":
        filtered_df = filtered_df[filtered_df['Prediction Horizon'] == selected_horizon]
    if selected_risk != "All":
        filtered_df = filtered_df[filtered_df['Risk Level'] == selected_risk]
        
    # Apply sorting
    if sort_order == "Newest First":
        filtered_df = filtered_df.sort_values(by="Timestamp", ascending=False)
    else:
        filtered_df = filtered_df.sort_values(by="Timestamp", ascending=True)
        
    # Format badges using emojis to retain Streamlit dataframe interactivity
    display_df = filtered_df.copy()
    
    def get_risk_emoji(val):
        if val == 'High': return '🔴 High'
        if val == 'Medium': return '🟡 Medium'
        return '🟢 Low'
        
    def get_status_emoji(val):
        if val == 'Success': return '✅ Success'
        return f'⚠️ {val}'
        
    if 'Risk Level' in display_df.columns:
        display_df['Risk Level'] = display_df['Risk Level'].apply(get_risk_emoji)
    if 'Status' in display_df.columns:
        display_df['Status'] = display_df['Status'].apply(get_status_emoji)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Styled Dataframe presentation
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Predicted Risk": st.column_config.ProgressColumn(
                "Predicted Risk",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "Confidence Score": st.column_config.ProgressColumn(
                "Confidence",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "Risk Level": st.column_config.TextColumn("Risk Level"),
            "Status": st.column_config.TextColumn("Status"),
            "Timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm:ss"),
        }
    )
    
    # CSV Export
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Prediction History (CSV)",
        data=csv,
        file_name='prediction_history.csv',
        mime='text/csv',
    )
