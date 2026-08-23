import streamlit as st

def inject_custom_css():
    st.markdown("""
        <style>
        /* Dark theme & Glassmorphism */
        .stApp {
            background-color: #0b101e;
            color: #e2e8f0;
            font-family: 'Inter', 'Roboto', sans-serif;
        }
        /* Dark SCADA Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #111827 !important;
            color: #e5e7eb !important;
            border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
        }
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2 {
            color: #e5e7eb !important;
        }
        
        /* Sidebar Subheaders */
        section[data-testid="stSidebar"] h3 {
            color: #f8fafc !important;
            font-size: 1.1rem !important;
            font-weight: 700 !important;
            margin-top: 24px !important;
            margin-bottom: 12px !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 8px;
        }

        /* Transform Radio Buttons into Professional Sidebar Links */
        section[data-testid="stSidebar"] div[role="radiogroup"] > label {
            padding: 10px 15px !important;
            border-radius: 8px !important;
            background: transparent !important;
            margin-bottom: 5px !important;
            transition: all 0.2s ease-in-out !important;
            cursor: pointer !important;
            width: 100% !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] > label p {
            color: #e5e7eb !important;
            font-weight: 500 !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
            background: rgba(255, 255, 255, 0.05) !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] > label[aria-checked="true"],
        section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {
            background: rgba(249, 115, 22, 0.1) !important;
            border-left: 4px solid #F97316 !important;
            border-radius: 4px 8px 8px 4px !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] > label[aria-checked="true"] p,
        section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] p {
            color: #F97316 !important;
            font-weight: 600 !important;
        }
        
        /* Hide the actual radio circle */
        section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
            display: none !important;
        }
        /* Some Streamlit versions use an SVG inside a container for the radio circle */
        section[data-testid="stSidebar"] div[role="radiogroup"] svg {
            display: none !important;
        }
        
        /* Sidebar Selectbox Customization - Background & Borders */
        section[data-testid="stSidebar"] .stSelectbox > div > div {
            border-radius: 8px !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            background-color: #1e293b !important;
        }
        
        /* AGGRESSIVE TEXT COLOR FIX FOR SELECTBOX & BUTTONS */
        section[data-testid="stSidebar"] .stSelectbox * {
            color: #f8fafc !important;
        }
        
        /* Restore the label color explicitly */
        section[data-testid="stSidebar"] .stSelectbox label * {
            color: #94a3b8 !important;
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
        }
        
        /* Fix dropdown arrow color */
        section[data-testid="stSidebar"] .stSelectbox svg {
            fill: #94a3b8 !important;
            color: #94a3b8 !important;
        }
        
        /* Fix Sidebar Buttons */
        section[data-testid="stSidebar"] .stButton button {
            background-color: transparent !important;
            border: 1px solid #3b82f6 !important;
            border-radius: 8px !important;
            transition: all 0.2s ease !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
        }
        section[data-testid="stSidebar"] .stButton button * {
            color: #3b82f6 !important;
            font-weight: 600 !important;
        }
        section[data-testid="stSidebar"] .stButton button:hover {
            background-color: rgba(59, 130, 246, 0.1) !important;
            border-color: #60a5fa !important;
        }
        
        /* AGGRESSIVE TEXT COLOR FIX FOR DROPDOWN MENUS (React Portals) */
        .stSelectbox div[role="listbox"] *,
        div[data-baseweb="popover"] *,
        div[data-testid="stVirtualDropdown"] *,
        ul[role="listbox"] *,
        li[role="option"] *,
        li[role="option"] {
            color: #f8fafc !important;
            background-color: #1e293b !important;
        }
        /* Hover state for dropdown options */
        li[role="option"]:hover,
        li[role="option"]:hover * {
            background-color: #334155 !important;
        }
        
        /* Custom SCADA Status Widget Unified */
        .scada-status-widget {
            background-color: #1e293b;
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .scada-status-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .scada-status-text {
            color: #e5e7eb;
            font-weight: 500;
            font-size: 0.9rem;
        }
        .scada-status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background-color: #22C55E;
            box-shadow: 0 0 8px rgba(34, 197, 94, 0.6);
        }
        
        .kpi-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
            margin-bottom: 20px;
            text-align: center;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            height: 100%;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.6);
            border-color: rgba(99, 102, 241, 0.5);
        }
        .kpi-value {
            font-size: 2.2rem;
            font-weight: bold;
            color: #60a5fa;
            margin: 10px 0;
            font-family: 'Courier New', Courier, monospace;
        }
        .kpi-title {
            font-size: 1rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }
        .kpi-trend {
            font-size: 1rem;
            font-weight: bold;
        }
        .trend-up { color: #10b981; }
        .trend-down { color: #ef4444; }
        .trend-flat { color: #94a3b8; }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-green { background-color: #10b981; box-shadow: 0 0 10px #10b981; }
        .status-yellow { background-color: #f59e0b; box-shadow: 0 0 10px #f59e0b; }
        .status-red { background-color: #ef4444; box-shadow: 0 0 15px #ef4444; animation: pulse 1.5s infinite; }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 10px #ef4444; }
            50% { box-shadow: 0 0 25px #ef4444; }
            100% { box-shadow: 0 0 10px #ef4444; }
        }
        
        .ai-panel {
            background: linear-gradient(145deg, #1e293b, #0f172a);
            border-radius: 12px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            margin-top: 20px;
            margin-bottom: 20px;
            box-shadow: 0 10px 20px -5px rgba(0, 0, 0, 0.8);
        }
        
        .log-container {
            background: #0f172a;
            border-radius: 8px;
            padding: 15px;
            font-family: 'Courier New', Courier, monospace;
            font-size: 0.9rem;
            color: #a3b8cc;
            height: 300px;
            overflow-y: auto;
            border: 1px solid #334155;
        }
        .log-entry {
            margin-bottom: 5px;
            border-bottom: 1px solid #1e293b;
            padding-bottom: 5px;
        }
        .log-info { color: #38bdf8; }
        .log-warning { color: #facc15; }
        .log-error { color: #f87171; }
        .log-success { color: #4ade80; }
        
        /* Headers formatting */
        h1, h2, h3 {
            color: #f8fafc;
        }
        </style>
    """, unsafe_allow_html=True)
