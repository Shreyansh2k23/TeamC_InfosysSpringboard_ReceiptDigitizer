"""
Global styling for the Receipt Vault Analyzer application.
This module provides consistent, professional UI/UX across all pages.
"""

import streamlit as st  # type: ignore


def apply_global_styles():
    """Apply professional global styles to the entire application"""
    st.markdown("""
    <style>
        /* ==================== GLOBAL STYLES ==================== */
        
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
        
        /* Root Variables */
        :root {
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --success-gradient: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            --warning-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            --info-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }
        
        /* Main App Background */
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            font-family: 'Inter', sans-serif;
        }
        
        /* Streamlit Container */
        .block-container {
            padding: 2rem 3rem;
            max-width: 1400px;
        }
        
        /* ==================== HEADERS ==================== */
        
        h1, h2, h3 {
            font-family: 'Inter', sans-serif;
            font-weight: 800;
        }
        
        h1 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding-bottom: 1rem;
            border-bottom: 3px solid #667eea;
            margin-bottom: 2rem;
        }
        
        h2 {
            color: #667eea;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        
        h3 {
            color: #764ba2;
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
        }
        
        /* ==================== CARDS & CONTAINERS ==================== */
        
        /* Metric Cards */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        [data-testid="stMetricLabel"] {
            font-weight: 600;
            color: #555;
            font-size: 1rem;
        }
        
        div[data-testid="metric-container"] {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.08);
            border-left: 4px solid #667eea;
            transition: all 0.3s ease;
        }
        
        div[data-testid="metric-container"]:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.15);
        }
        
        /* ==================== BUTTONS ==================== */
        
        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.3s ease;
            border: none;
            font-family: 'Inter', sans-serif;
        }
        
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }
        
        .stButton > button[kind="primary"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }
        
        .stButton > button[kind="secondary"] {
            background: white;
            color: #667eea;
            border: 2px solid #667eea;
        }
        
        .stButton > button[kind="secondary"]:hover {
            background: #667eea;
            color: white;
            transform: translateY(-2px);
        }
        
        .stDownloadButton > button {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            border-radius: 10px;
            font-weight: 600;
            box-shadow: 0 5px 15px rgba(17, 153, 142, 0.3);
        }
        
        .stDownloadButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(17, 153, 142, 0.4);
        }
        
        /* ==================== INPUTS ==================== */
        
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > div {
            border-radius: 10px;
            border: 2px solid #e0e0e0;
            font-family: 'Inter', sans-serif;
            transition: all 0.3s ease;
        }
        
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        /* ==================== DATA EDITOR / TABLES ==================== */
        
        [data-testid="stDataFrame"] {
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        }
        
        /* ==================== EXPANDER ==================== */
        
        .streamlit-expanderHeader {
            background: white;
            border-radius: 10px;
            font-weight: 600;
            color: #667eea;
            border: 2px solid #e0e0e0;
        }
        
        .streamlit-expanderHeader:hover {
            border-color: #667eea;
        }
        
        /* ==================== DIVIDER ==================== */
        
        hr {
            margin: 2rem 0;
            border: none;
            height: 2px;
            background: linear-gradient(90deg, transparent, #667eea, transparent);
        }
        
        /* ==================== SIDEBAR ==================== */
        
        [data-testid="stSidebar"] {
            background: white;
            padding: 0;
            border-right: 3px solid #e0e0e0;
        }
        
        [data-testid="stSidebar"] > div:first-child {
            background: white;
        }
        
        /* Sidebar Text - Reset to dark, but allow white text in custom cards */
        [data-testid="stSidebar"] *:not(.custom-card-text) {
            color: #333 !important;
        }
        
        /* Allow white text for specific cases in sidebar */
        .sidebar-white-text {
            color: white !important;
        }
        
        /* Sidebar Headers */
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        /* Sidebar Inputs */
        [data-testid="stSidebar"] input {
            background: #f8f9fa !important;
            border: 2px solid #e0e0e0 !important;
            color: #333 !important;
            border-radius: 10px;
        }
        
        [data-testid="stSidebar"] input::placeholder {
            color: #999 !important;
        }
        
        [data-testid="stSidebar"] input:focus {
            background: white !important;
            border-color: #667eea !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        /* Sidebar Selectbox */
        [data-testid="stSidebar"] [data-baseweb="select"] {
            background: #f8f9fa;
            border-radius: 10px;
            border: 2px solid #e0e0e0;
        }
        
        [data-testid="stSidebar"] [data-baseweb="select"]:hover {
            border-color: #667eea;
        }
        
        /* Sidebar Radio Buttons */
        [data-testid="stSidebar"] [data-testid="stRadio"] > div {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            border: 2px solid #e0e0e0;
        }
        
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            background: white;
            padding: 0.8rem 1rem;
            border-radius: 8px;
            margin: 0.3rem 0;
            transition: all 0.3s ease;
            border: 1px solid #e0e0e0;
        }
        
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: #f0f0ff;
            border-color: #667eea;
            transform: translateX(5px);
        }
        
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            border-color: #667eea;
        }
        
        /* Sidebar Buttons */
        [data-testid="stSidebar"] button {
            background: white !important;
            color: #667eea !important;
            border: 2px solid #667eea !important;
            font-weight: 600 !important;
            border-radius: 10px;
        }
        
        [data-testid="stSidebar"] button:hover {
            background: #667eea !important;
            color: white !important;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }
        
        /* Sidebar Divider */
        [data-testid="stSidebar"] hr {
            background: linear-gradient(90deg, transparent, #667eea, transparent);
            margin: 1.5rem 0;
            height: 2px;
        }
        
        /* Sidebar Progress Bar */
        [data-testid="stSidebar"] [data-testid="stProgress"] > div > div {
            background: #e0e0e0;
        }
        
        [data-testid="stSidebar"] [data-testid="stProgress"] > div > div > div {
            background: linear-gradient(90deg, #667eea, #764ba2);
        }
        
        /* ==================== ALERTS ==================== */
        
        .stAlert {
            border-radius: 10px;
            border-left: 4px solid;
            font-family: 'Inter', sans-serif;
        }
        
        /* Success Alert */
        [data-baseweb="notification"][kind="success"] {
            background: linear-gradient(135deg, rgba(17, 153, 142, 0.1), rgba(56, 239, 125, 0.1));
            border-left-color: #11998e;
        }
        
        /* Info Alert */
        [data-baseweb="notification"][kind="info"] {
            background: linear-gradient(135deg, rgba(79, 172, 254, 0.1), rgba(0, 242, 254, 0.1));
            border-left-color: #4facfe;
        }
        
        /* Warning Alert */
        [data-baseweb="notification"][kind="warning"] {
            background: linear-gradient(135deg, rgba(240, 147, 251, 0.1), rgba(245, 87, 108, 0.1));
            border-left-color: #f093fb;
        }
        
        /* Error Alert */
        [data-baseweb="notification"][kind="error"] {
            background: linear-gradient(135deg, rgba(245, 87, 108, 0.1), rgba(240, 147, 251, 0.1));
            border-left-color: #f5576c;
        }
        
        /* ==================== TABS ==================== */
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            background: white;
            padding: 1rem;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            padding: 0.8rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        /* ==================== FILE UPLOADER ==================== */
        
        [data-testid="stFileUploader"] {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            border: 2px dashed #667eea;
            box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: #764ba2;
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.1);
        }
        
        /* ==================== ANIMATIONS ==================== */
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .main > div {
            animation: fadeIn 0.5s ease-out;
        }
        
        /* ==================== SCROLLBAR ==================== */
        
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
        
        /* ==================== RESPONSIVE ==================== */
        
        @media (max-width: 768px) {
            .block-container {
                padding: 1rem;
            }
            
            h1 {
                font-size: 1.8rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)


def apply_page_header(title: str, subtitle: str = "", icon: str = "📊"):
    """Apply a professional page header with icon and subtitle"""
    st.markdown(f"""
    <div style="background: white; padding: 2rem; border-radius: 20px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.08); margin-bottom: 2rem;
                border-left: 5px solid #667eea;">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div style="font-size: 3rem;">{icon}</div>
            <div>
                <h1 style="margin: 0; padding: 0; border: none; font-size: 2.5rem;">{title}</h1>
                {f'<p style="color: #666; margin: 0.5rem 0 0 0; font-size: 1.1rem;">{subtitle}</p>' if subtitle else ''}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)