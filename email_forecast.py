# app.py — Forecasting Tool with Requirement Calc (built for JA)
# --------------------------------------------------------
# Features:
# - Support for Email, Voice, and Chat LOBs
# - Upload weekly or monthly volume data (CSV/XLSX) or use sample data
# - Forecast using ETS (Holt-Winters) or ARIMA
# - Requirements calculator using AHT + Shrinkage + Paid Hours + Backlog + SLA buffer
# - JustAnswer teal-to-sky-blue gradient theme
#
# Install:
#   pip install streamlit pandas numpy plotly openpyxl statsmodels
#   pip install prophet xgboost  # Optional: for Prophet and XGBoost models
# Run:
#   streamlit run app.py

import math
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from io import BytesIO

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Optional imports for Prophet and XGBoost
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

# ----------------------------
# JustAnswer Teal-to-Sky-Blue Gradient Theme
# ----------------------------
DEFAULT_PRIMARY = "#14B8A6"   # bright teal
DEFAULT_ACCENT = "#38BDF8"    # sky-blue
DEFAULT_BG = "#F0FDFA"        # light teal-tinted background
DEFAULT_PANEL = "#FFFFFF"     # white
DEFAULT_TEXT = "#0F172A"      # dark slate
DEFAULT_MUTED = "#475569"     # muted slate
DEFAULT_WARN = "#F59E0B"      # amber
DEFAULT_DANGER = "#EF4444"    # red

st.set_page_config(
    page_title="Forecasting Tool with Requirement Calc (built for JA)",
    page_icon="📊",
    layout="wide",
)


def inject_css(primary, accent, bg, panel, text, muted, warn, danger):
    st.markdown(
        f"""
        <style>
        :root {{
          --primary: {primary};
          --accent: {accent};
          --bg: {bg};
          --panel: {panel};
          --text: {text};
          --muted: {muted};
          --warn: {warn};
          --danger: {danger};
        }}

        .stApp {{
          background: linear-gradient(135deg, rgba(20,184,166,0.08) 0%, rgba(56,189,248,0.12) 50%, rgba(20,184,166,0.08) 100%);
          background-attachment: fixed;
          color: var(--text);
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", "Roboto", "Helvetica Neue", Arial, sans-serif;
          line-height: 1.6;
          min-height: 100vh;
        }}

        /* JustAnswer gradient overlay */
        .stApp::before {{
          content: '';
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          height: 300px;
          background: linear-gradient(135deg, 
                      rgba(20,184,166,0.12) 0%, 
                      rgba(56,189,248,0.15) 50%,
                      rgba(20,184,166,0.12) 100%);
          z-index: -1;
          pointer-events: none;
          opacity: 0.6;
        }}

        /* Smooth scroll behavior */
        html {{
          scroll-behavior: smooth;
        }}

        /* Loading animations */
        @keyframes fadeIn {{
          from {{ opacity: 0; transform: translateY(10px); }}
          to {{ opacity: 1; transform: translateY(0); }}
        }}

        .card, .filter-card, div[data-testid="metric-container"] {{
          animation: fadeIn 0.5s ease-out;
        }}

        /* Modern web app container */
        .main .block-container {{
          background: transparent;
        }}

        /* Section dividers */
        .section-divider {{
          height: 1px;
          background: linear-gradient(90deg, transparent, rgba(20,184,166,0.2), transparent);
          margin: 2.5rem 0;
          border: none;
        }}

        html, body, [class*="css"] {{
          color: var(--text) !important;
          font-size: 15px;
          font-weight: 400;
        }}

        /* Main container improvements */
        .main .block-container {{
          padding-top: 1.5rem;
          padding-bottom: 3rem;
          max-width: 1400px;
        }}

        /* Top navigation bar style */
        header[data-testid="stHeader"] {{
          background: linear-gradient(135deg, rgba(20,184,166,0.12), rgba(56,189,248,0.15));
          border-bottom: 2px solid rgba(20,184,166,0.2);
          box-shadow: 0 2px 12px rgba(20,184,166,0.1);
          backdrop-filter: blur(10px);
        }}

        /* Hide default Streamlit menu for cleaner look */
        #MainMenu {{
          visibility: hidden;
        }}

        footer {{
          visibility: hidden;
        }}

        /* Enhanced section headers */
        h3::before {{
          content: '';
          display: inline-block;
          width: 4px;
          height: 24px;
          background: linear-gradient(135deg, {primary}, {accent});
          border-radius: 2px;
          margin-right: 12px;
          vertical-align: middle;
        }}

        /* Cleaner spacing */
        .element-container:has([data-testid="stMarkdownContainer"]) {{
          margin-bottom: 1.5rem;
        }}

        /* Better section headers */
        h3, h4 {{
          margin-top: 2rem;
          margin-bottom: 1rem;
        }}

        /* Improved list spacing in markdown */
        .stMarkdown ul, .stMarkdown ol {{
          margin-top: 0.5rem;
          margin-bottom: 0.5rem;
        }}

        .stMarkdown li {{
          margin-bottom: 0.375rem;
        }}

        /* Headers */
        h1, h2, h3, h4, h5, h6 {{
          font-weight: 700;
          letter-spacing: -0.025em;
          line-height: 1.2;
        }}

        h1 {{
          font-size: 2.5rem;
        }}

        h2 {{
          font-size: 2rem;
        }}

        h3 {{
          font-size: 1.5rem;
        }}

        h4 {{
          font-size: 1.25rem;
        }}

        section[data-testid="stSidebar"] {{
          background: linear-gradient(180deg, #FFFFFF 0%, rgba(240,253,250,0.5) 100%);
          border-right: 2px solid rgba(20,184,166,0.15);
          box-shadow: 2px 0 12px rgba(20,184,166,0.05);
        }}

        /* Success/Info/Warning/Error messages */
        .stSuccess {{
          border-radius: 12px;
          border-left: 4px solid #10B981;
          background: rgba(16,185,129,0.1);
          padding: 1rem;
        }}

        .stInfo {{
          border-radius: 12px;
          border-left: 4px solid {accent};
          background: rgba(56,189,248,0.1);
          padding: 1rem;
        }}

        .stWarning {{
          border-radius: 12px;
          border-left: 4px solid {warn};
          background: rgba(245,158,11,0.1);
          padding: 1rem;
        }}

        .stError {{
          border-radius: 12px;
          border-left: 4px solid {danger};
          background: rgba(239,68,68,0.1);
          padding: 1rem;
        }}

        /* Spinner improvements */
        .stSpinner > div {{
          border-color: {primary} transparent transparent transparent;
        }}

        .card {{
          background: #FFFFFF;
          border: 1px solid rgba(20,184,166,0.2);
          border-radius: 20px;
          padding: 36px 40px;
          box-shadow: 0 2px 8px rgba(20,184,166,0.08), 0 8px 24px rgba(20,184,166,0.06);
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          position: relative;
          overflow: hidden;
        }}

        .card::before {{
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          height: 3px;
          background: linear-gradient(90deg, {primary}, {accent});
        }}

        .card:hover {{
          border-color: rgba(20,184,166,0.35);
          box-shadow: 0 4px 12px rgba(20,184,166,0.12), 0 12px 32px rgba(20,184,166,0.1);
          transform: translateY(-2px);
        }}

        .card-forecast {{
          background: linear-gradient(135deg, rgba(20,184,166,0.06) 0%, rgba(56,189,248,0.1) 100%);
          border-color: rgba(20,184,166,0.25);
          box-shadow: 0 2px 4px rgba(0,0,0,0.04), 0 8px 16px rgba(20,184,166,0.08);
        }}

        .card-req {{
          background: linear-gradient(135deg, rgba(20,184,166,0.06) 0%, rgba(56,189,248,0.1) 100%);
          border-color: rgba(20,184,166,0.25);
          box-shadow: 0 2px 4px rgba(0,0,0,0.04), 0 8px 16px rgba(20,184,166,0.08);
        }}

        /* Guide section styling */
        .guide-section {{
          background: #FFFFFF;
          border: 1px solid rgba(20,184,166,0.15);
          border-radius: 16px;
          padding: 28px 32px;
          margin-bottom: 24px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }}

        .guide-section h4 {{
          color: {primary};
          margin-bottom: 16px;
          padding-bottom: 12px;
          border-bottom: 2px solid rgba(20,184,166,0.15);
        }}

        .guide-section h5 {{
          color: var(--text);
          margin-top: 24px;
          margin-bottom: 12px;
          font-weight: 600;
        }}

        .guide-item {{
          padding: 12px 0;
          border-bottom: 1px solid rgba(20,184,166,0.08);
        }}

        .guide-item:last-child {{
          border-bottom: none;
        }}

        .guide-item strong {{
          color: {primary};
          font-weight: 600;
        }}

        .badge {{
          display: inline-block;
          padding: 6px 12px;
          border-radius: 999px;
          font-size: 12px;
          font-weight: 600;
          border: 1px solid rgba(20,184,166,0.35);
          background: linear-gradient(135deg, rgba(20,184,166,0.12), rgba(56,189,248,0.18));
          color: {primary};
          margin-left: 8px;
        }}

        .title {{
          font-size: 36px;
          font-weight: 800;
          letter-spacing: -0.02em;
          margin-bottom: 8px;
          background: linear-gradient(135deg, {primary}, {accent});
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }}
        .subtitle {{
          font-size: 15px;
          color: var(--muted);
          margin-top: -8px;
          margin-bottom: 12px;
        }}

        div[data-testid="metric-container"] {{
          background: linear-gradient(135deg, #FFFFFF 0%, rgba(240,253,250,0.5) 100%);
          border: 1.5px solid rgba(20,184,166,0.2);
          padding: 28px 32px;
          border-radius: 16px;
          box-shadow: 0 2px 6px rgba(20,184,166,0.08);
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          position: relative;
          overflow: hidden;
        }}

        div[data-testid="metric-container"]::before {{
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          height: 3px;
          background: linear-gradient(90deg, {primary}, {accent});
        }}

        div[data-testid="metric-container"]:hover {{
          border-color: rgba(20,184,166,0.3);
          box-shadow: 0 4px 12px rgba(20,184,166,0.12);
          transform: translateY(-2px);
        }}

        div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
          font-size: 1.75rem;
          font-weight: 700;
          color: var(--text);
        }}

        div[data-testid="metric-container"] [data-testid="stMetricLabel"] {{
          font-size: 0.875rem;
          font-weight: 600;
          color: var(--muted);
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }}

        .stButton>button {{
          border-radius: 12px;
          border: none;
          background: linear-gradient(135deg, {primary}, {accent});
          color: white;
          font-weight: 600;
          font-size: 0.9375rem;
          padding: 0.75rem 1.5rem;
          box-shadow: 0 2px 8px rgba(20,184,166,0.25), 0 4px 16px rgba(20,184,166,0.15);
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          letter-spacing: 0.025em;
        }}
        .stButton>button:hover {{
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(20,184,166,0.3), 0 8px 24px rgba(20,184,166,0.2);
        }}
        .stButton>button:active {{
          transform: translateY(0);
        }}

        /* Download buttons */
        .stDownloadButton>button {{
          border-radius: 12px;
          border: 1px solid rgba(20,184,166,0.3);
          background: #FFFFFF;
          color: {primary};
          font-weight: 600;
          font-size: 0.875rem;
          padding: 0.625rem 1.25rem;
          box-shadow: 0 1px 3px rgba(0,0,0,0.05);
          transition: all 0.3s ease;
        }}
        .stDownloadButton>button:hover {{
          background: linear-gradient(135deg, rgba(20,184,166,0.05), rgba(56,189,248,0.08));
          border-color: {primary};
          transform: translateY(-1px);
          box-shadow: 0 2px 8px rgba(20,184,166,0.15);
        }}

        .stTextInput input, .stNumberInput input, .stSelectbox div, .stMultiSelect div {{
          background-color: #FFFFFF !important;
          border: 1.5px solid rgba(20,184,166,0.2) !important;
          border-radius: 12px !important;
          color: var(--text) !important;
          box-shadow: 0 1px 3px rgba(0,0,0,0.05);
          transition: all 0.2s ease;
          font-size: 0.9375rem;
          padding: 0.625rem 0.875rem !important;
        }}
        .stTextInput input:focus, .stNumberInput input:focus, .stSelectbox div:focus-within {{
          border-color: {primary} !important;
          box-shadow: 0 0 0 3px rgba(20,184,166,0.1), 0 2px 8px rgba(20,184,166,0.15) !important;
          outline: none;
        }}
        .stTextInput input:hover, .stNumberInput input:hover {{
          border-color: rgba(20,184,166,0.4) !important;
        }}

        /* Selectbox improvements */
        .stSelectbox [data-baseweb="select"] {{
          border-radius: 12px !important;
        }}

        /* Checkbox and radio improvements */
        .stCheckbox label, .stRadio label {{
          font-weight: 500;
          color: var(--text);
        }}

        .stDataFrame, .stTable {{
          border-radius: 16px;
          overflow: hidden;
          border: 1.5px solid rgba(20,184,166,0.2);
          box-shadow: 0 2px 8px rgba(20,184,166,0.08);
          background: #FFFFFF;
          transition: all 0.3s ease;
        }}

        .stDataFrame:hover, .stTable:hover {{
          box-shadow: 0 4px 12px rgba(20,184,166,0.12);
        }}
        .stDataFrame [role="grid"] {{
          font-size: 0.875rem;
          color: var(--text);
        }}
        .stDataFrame [role="columnheader"] {{
          background: linear-gradient(135deg, rgba(20,184,166,0.08), rgba(56,189,248,0.12)) !important;
          font-weight: 600;
          color: var(--text);
          border-bottom: 2px solid rgba(20,184,166,0.2);
        }}
        .stDataFrame [role="row"]:hover {{
          background: rgba(20,184,166,0.05) !important;
        }}

        a {{
          color: var(--primary) !important;
        }}

        div[data-testid="stTabs"] > div[role="tablist"] {{
          gap: 0.5rem;
          background: linear-gradient(135deg, rgba(20,184,166,0.05) 0%, rgba(56,189,248,0.08) 100%);
          padding: 0.75rem;
          border-radius: 16px;
          border: 1px solid rgba(20,184,166,0.15);
          box-shadow: inset 0 1px 3px rgba(20,184,166,0.05);
        }}

        div[data-testid="stTabs"] > div[role="tablist"] button[role="tab"] {{
          color: {text} !important;
          font-weight: 600;
          background: #FFFFFF;
          border-radius: 12px;
          padding: 0.625rem 1.5rem;
          border: 1.5px solid rgba(20,184,166,0.2);
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}

        div[data-testid="stTabs"] > div[role="tablist"] button[role="tab"]:hover {{
          background: linear-gradient(135deg, rgba(20,184,166,0.08), rgba(56,189,248,0.1));
          border-color: rgba(20,184,166,0.35);
          transform: translateY(-1px);
          box-shadow: 0 2px 6px rgba(20,184,166,0.15);
        }}

        div[data-testid="stTabs"] > div[role="tablist"] button[role="tab"][aria-selected="true"] {{
          background: linear-gradient(135deg, {primary}, {accent});
          border-color: {primary};
          color: #FFFFFF !important;
          box-shadow: 0 2px 8px rgba(20,184,166,0.25);
        }}

        div[data-testid="stTabs"] > div[role="tablist"] button[role="tab"]:focus-visible {{
          outline: 2px solid {primary};
          outline-offset: 2px;
        }}

        .js-plotly-plot {{
          background: #FFFFFF !important;
          border-radius: 16px;
          box-shadow: 0 2px 8px rgba(20,184,166,0.08);
          padding: 1.5rem;
          border: 1px solid rgba(20,184,166,0.15);
          transition: all 0.3s ease;
        }}

        .js-plotly-plot:hover {{
          box-shadow: 0 4px 16px rgba(20,184,166,0.12);
        }}

        /* Section spacing */
        .element-container {{
          margin-bottom: 2rem;
        }}

        /* Improved list styling */
        ul, ol {{
          padding-left: 1.5rem;
        }}

        li {{
          margin-bottom: 0.5rem;
          line-height: 1.6;
        }}

        /* Better divider */
        hr {{
          border: none;
          border-top: 1px solid rgba(20,184,166,0.15);
          margin: 2rem 0;
        }}

        /* Caption styling */
        .stCaption {{
          color: var(--muted);
          font-size: 0.875rem;
          font-weight: 500;
        }}

        /* Radio button improvements */
        .stRadio > div {{
          gap: 0.75rem;
        }}

        .stRadio > div > label {{
          padding: 0.75rem 1rem;
          border-radius: 12px;
          border: 1.5px solid rgba(20,184,166,0.2);
          transition: all 0.2s ease;
        }}

        .stRadio > div > label:hover {{
          border-color: {primary};
          background: rgba(20,184,166,0.05);
        }}

        .stRadio > div > label[data-baseweb="radio"]:has(input:checked) {{
          border-color: {primary};
          background: linear-gradient(135deg, rgba(20,184,166,0.1), rgba(56,189,248,0.1));
        }}

        /* File uploader improvements */
        .stFileUploader {{
          border-radius: 12px;
          border: 1.5px dashed rgba(20,184,166,0.3);
          padding: 1.5rem;
          transition: all 0.3s ease;
        }}

        .stFileUploader:hover {{
          border-color: {primary};
          background: rgba(20,184,166,0.02);
        }}

        /* Enhanced filter cards - pill-style segmented toggle container */
        .filter-card {{
          background: rgba(20,184,166,0.08);
          border: 1.5px solid rgba(20,184,166,0.25);
          border-radius: 16px;
          padding: 16px 20px;
          position: relative;
        }}
        .filter-label {{
          font-size: 13px;
          font-weight: 600;
          color: {muted};
          margin: 0 0 12px 0;
          text-transform: none;
          letter-spacing: 0.2px;
        }}
        
        /* Segmented Toggle Buttons (Radio-style) */
        .segmented-toggle {{
          display: flex;
          background: rgba(255,255,255,0.8);
          border-radius: 12px;
          padding: 4px;
          gap: 4px;
          border: 1px solid rgba(20,184,166,0.2);
        }}
        
        .segmented-toggle label {{
          flex: 1;
          padding: 10px 16px;
          text-align: center;
          border-radius: 8px;
          font-size: 14px;
          font-weight: 600;
          color: {muted};
          cursor: pointer;
          transition: all 0.2s ease;
          background: transparent;
          border: none;
          margin: 0;
        }}
        
        .segmented-toggle input[type="radio"] {{
          display: none;
        }}
        
        .segmented-toggle label:hover {{
          background: rgba(20,184,166,0.08);
          color: {primary};
        }}
        
        .segmented-toggle input[type="radio"]:checked + label {{
          background: linear-gradient(135deg, {primary}, {accent});
          color: white;
          box-shadow: 0 2px 6px rgba(20,184,166,0.3);
        }}
        
        /* Override Streamlit radio button styling for segmented toggles */
        .filter-card .stRadio {{
          margin-top: 0;
        }}
        
        .filter-card .stRadio > div {{
          display: flex !important;
          background: rgba(255,255,255,0.95) !important;
          border-radius: 12px !important;
          padding: 4px !important;
          gap: 4px !important;
          border: 1.5px solid rgba(20,184,166,0.25) !important;
          flex-direction: row !important;
          box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);
        }}
        
        /* Highlighted model selection */
        .model-selected {{
          background: linear-gradient(135deg, {primary}, {accent}) !important;
          color: white !important;
          padding: 12px 16px !important;
          border-radius: 8px !important;
          text-align: center !important;
          font-weight: 700 !important;
          box-shadow: 0 2px 8px rgba(20,184,166,0.3) !important;
        }}
        
        /* Style selectbox inside filter card */
        .filter-card ~ .stSelectbox {{
          margin-top: 0;
        }}
        
        .filter-card ~ .stSelectbox > div > div {{
          background: rgba(255,255,255,0.95) !important;
          border: 1.5px solid rgba(20,184,166,0.25) !important;
          border-radius: 8px !important;
        }}
        
        .filter-card .stRadio > div > label {{
          flex: 1 !important;
          padding: 10px 16px !important;
          text-align: center !important;
          border-radius: 8px !important;
          font-size: 14px !important;
          font-weight: 600 !important;
          color: {muted} !important;
          cursor: pointer !important;
          transition: all 0.2s ease !important;
          background: transparent !important;
          border: 1.5px solid transparent !important;
          margin: 0 !important;
          display: flex !important;
          align-items: center !important;
          justify-content: center !important;
          min-height: 40px;
        }}
        
        .filter-card .stRadio > div > label:hover {{
          background: rgba(20,184,166,0.1) !important;
          color: {primary} !important;
          border-color: rgba(20,184,166,0.3) !important;
        }}
        
        .filter-card .stRadio > div > label[data-baseweb="radio"]:has(input:checked),
        .filter-card .stRadio > div > label:has(input[checked]),
        .filter-card .stRadio > div > label:has(input[type="radio"]:checked) {{
          background: linear-gradient(135deg, {primary}, {accent}) !important;
          color: white !important;
          border-color: {primary} !important;
          box-shadow: 0 2px 8px rgba(20,184,166,0.35) !important;
        }}
        
        /* Ensure radio buttons are hidden but functional */
        .filter-card .stRadio input[type="radio"] {{
          margin-right: 0 !important;
          margin-left: 0 !important;
          position: absolute;
          opacity: 0;
          width: 0;
          height: 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css(
    DEFAULT_PRIMARY,
    DEFAULT_ACCENT,
    DEFAULT_BG,
    DEFAULT_PANEL,
    DEFAULT_TEXT,
    DEFAULT_MUTED,
    DEFAULT_WARN,
    DEFAULT_DANGER,
)

# ----------------------------
# Helpers
# ----------------------------
def to_datetime_safe(x: pd.Series) -> pd.Series:
    return pd.to_datetime(x, errors="coerce", infer_datetime_format=True)


def smape(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom == 0, 1e-9, denom)
    return float(100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def mape(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.where(y_true == 0, 1e-9, y_true)
    return float(100.0 * np.mean(np.abs((y_true - y_pred) / denom)))


def wape(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    denom = denom if denom != 0 else 1e-9
    return float(100.0 * np.sum(np.abs(y_true - y_pred)) / denom)


def mae(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def make_line_chart(df: pd.DataFrame, x_col: str, y_cols: List[str], title: str):
    fig = go.Figure()
    for c in y_cols:
        fig.add_trace(go.Scatter(x=df[x_col], y=df[c], mode="lines+markers", name=c))
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title="Count",
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#1E293B"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        margin=dict(l=30, r=30, t=70, b=40),
    )
    return fig


def read_file(upload) -> pd.DataFrame:
    name = upload.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(upload)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(upload)
    raise ValueError("Unsupported file type. Upload CSV or Excel.")


def sample_data(lob_type: str, freq: str) -> pd.DataFrame:
    """Generate sample data for Email, Voice, or Chat"""
    if freq == "Daily":
        rng = pd.date_range("2024-01-01", periods=365, freq="D")
        np.random.seed(42)
        if lob_type == "Email":
            base = 300 + 50 * np.sin(np.linspace(0, 4 * np.pi, len(rng)))  # Weekly seasonality
            weekly_pattern = 30 * np.sin(2 * np.pi * np.arange(len(rng)) / 7)  # Day of week
            noise = np.random.normal(0, 25, size=len(rng))
            vol = np.maximum(50, (base + weekly_pattern + noise)).round().astype(int)
        elif lob_type == "Voice":
            base = 250 + 40 * np.sin(np.linspace(0, 4 * np.pi, len(rng)))
            weekly_pattern = 25 * np.sin(2 * np.pi * np.arange(len(rng)) / 7)
            noise = np.random.normal(0, 20, size=len(rng))
            vol = np.maximum(40, (base + weekly_pattern + noise)).round().astype(int)
        else:  # Chat
            base = 450 + 60 * np.sin(np.linspace(0, 4 * np.pi, len(rng)))
            weekly_pattern = 40 * np.sin(2 * np.pi * np.arange(len(rng)) / 7)
            noise = np.random.normal(0, 35, size=len(rng))
            vol = np.maximum(60, (base + weekly_pattern + noise)).round().astype(int)
    elif freq == "Weekly":
        rng = pd.date_range("2025-01-05", periods=52, freq="W-SUN")
        np.random.seed(42)
        if lob_type == "Email":
            base = 2200 + 250 * np.sin(np.linspace(0, 2 * np.pi, len(rng)))
            noise = np.random.normal(0, 140, size=len(rng))
            vol = np.maximum(200, (base + noise)).round().astype(int)
        elif lob_type == "Voice":
            base = 1800 + 200 * np.sin(np.linspace(0, 2 * np.pi, len(rng)))
            noise = np.random.normal(0, 120, size=len(rng))
            vol = np.maximum(150, (base + noise)).round().astype(int)
        else:  # Chat
            base = 3200 + 300 * np.sin(np.linspace(0, 2 * np.pi, len(rng)))
            noise = np.random.normal(0, 180, size=len(rng))
            vol = np.maximum(250, (base + noise)).round().astype(int)
    else:  # Monthly
        rng = pd.date_range("2023-01-01", periods=36, freq="MS")
        np.random.seed(7)
        if lob_type == "Email":
            trend = np.linspace(16000, 22000, len(rng))
            season = 1200 * np.sin(np.linspace(0, 2 * np.pi, len(rng)))
            noise = np.random.normal(0, 800, size=len(rng))
            vol = np.maximum(500, (trend + season + noise)).round().astype(int)
        elif lob_type == "Voice":
            trend = np.linspace(12000, 18000, len(rng))
            season = 900 * np.sin(np.linspace(0, 2 * np.pi, len(rng)))
            noise = np.random.normal(0, 600, size=len(rng))
            vol = np.maximum(400, (trend + season + noise)).round().astype(int)
        else:  # Chat
            trend = np.linspace(24000, 32000, len(rng))
            season = 1800 * np.sin(np.linspace(0, 2 * np.pi, len(rng)))
            noise = np.random.normal(0, 1200, size=len(rng))
            vol = np.maximum(800, (trend + season + noise)).round().astype(int)
    
    volume_col = f"{lob_type} Volume"
    return pd.DataFrame({"Date": rng, volume_col: vol})


def aggregate_series(df: pd.DataFrame, date_col: str, vol_col: str, freq: str) -> pd.DataFrame:
    x = df.copy()
    x[date_col] = to_datetime_safe(x[date_col])
    x = x.dropna(subset=[date_col, vol_col])
    x[vol_col] = pd.to_numeric(x[vol_col], errors="coerce")
    x = x.dropna(subset=[vol_col])
    x = x.sort_values(date_col)

    if freq == "Daily":
        s = x.set_index(date_col)[vol_col].resample("D").sum()
    elif freq == "Weekly":
        s = x.set_index(date_col)[vol_col].resample("W-SUN").sum()
    else:  # Monthly
        s = x.set_index(date_col)[vol_col].resample("MS").sum()

    out = s.reset_index().rename(columns={date_col: "Period", vol_col: "Volume"})
    return out


def detect_anomalies(series: pd.Series, method: str = "zscore", threshold: float = 3.0) -> pd.Series:
    """
    Detect anomalies in a time series using Z-score method.
    
    Args:
        series: Time series data
        method: 'zscore' (only method supported)
        threshold: Z-score threshold (default 3.0)
    
    Returns:
        Boolean series where True indicates an anomaly
    """
    mean = series.mean()
    std = series.std()
    if std == 0:
        return pd.Series([False] * len(series), index=series.index)
    z_scores = np.abs((series - mean) / std)
    anomalies = z_scores > threshold
    return anomalies


def aggregate_daily_to_weekly(daily_df: pd.DataFrame, date_col: str = "Period", vol_col: str = "Volume") -> pd.DataFrame:
    """Aggregate daily data to weekly (Sunday-Saturday weeks)"""
    df = daily_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    weekly = df[vol_col].resample("W-SUN").sum().reset_index()
    weekly = weekly.rename(columns={date_col: "Period", vol_col: "Volume"})
    return weekly


def aggregate_daily_to_monthly(daily_df: pd.DataFrame, date_col: str = "Period", vol_col: str = "Volume") -> pd.DataFrame:
    """Aggregate daily data to monthly (first day of month)"""
    df = daily_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    monthly = df[vol_col].resample("MS").sum().reset_index()
    monthly = monthly.rename(columns={date_col: "Period", vol_col: "Volume"})
    return monthly


def fit_ets(train: pd.Series, seasonal_periods: int, seasonality: str) -> Tuple[object, pd.Series]:
    if seasonality == "none" or len(train) < 2 * seasonal_periods:
        model = ExponentialSmoothing(
            train, trend="add", seasonal=None, initialization_method="estimated"
        )
    else:
        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add" if seasonality == "add" else "mul",
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        )
    fitted = model.fit(optimized=True)
    return fitted, fitted.fittedvalues


def moving_average(train: pd.Series, window: int, horizon: int) -> Tuple[pd.Series, pd.Series]:
    """Simple Moving Average forecast"""
    ma_values = train.rolling(window=window).mean()
    fitted_vals = ma_values.fillna(train.iloc[:window].mean())
    
    # Forecast: use last window average
    last_avg = train.iloc[-window:].mean()
    forecast = pd.Series([last_avg] * horizon)
    
    return fitted_vals, forecast


def weighted_moving_average(train: pd.Series, window: int, horizon: int) -> Tuple[pd.Series, pd.Series]:
    """Weighted Moving Average forecast (more recent values weighted higher)"""
    weights = np.linspace(1, window, window)
    weights = weights / weights.sum()
    
    fitted_vals = pd.Series(index=train.index, dtype=float)
    for i in range(window - 1, len(train)):
        fitted_vals.iloc[i] = (train.iloc[i-window+1:i+1] * weights).sum()
    
    # Fill initial NaN values
    fitted_vals.iloc[:window-1] = train.iloc[:window-1]
    
    # Forecast: use weighted average of last window
    last_weighted_avg = (train.iloc[-window:] * weights).sum()
    forecast = pd.Series([last_weighted_avg] * horizon)
    
    return fitted_vals, forecast


def auto_arima_grid(train: pd.Series, seasonal_periods: int, seasonal: bool) -> Tuple[object, Tuple[int, int, int], Tuple[int, int, int, int]]:
    best_aic = float("inf")
    best_fit = None
    best_order = (1, 1, 1)
    best_seasonal = (0, 0, 0, seasonal_periods)

    d_candidates = [0, 1]
    p_candidates = [0, 1, 2]
    q_candidates = [0, 1, 2]

    if seasonal:
        D_candidates = [0, 1]
        P_candidates = [0, 1]
        Q_candidates = [0, 1]
    else:
        D_candidates = [0]
        P_candidates = [0]
        Q_candidates = [0]

    for d in d_candidates:
        for p in p_candidates:
            for q in q_candidates:
                for D in D_candidates:
                    for P in P_candidates:
                        for Q in Q_candidates:
                            try:
                                fit = SARIMAX(
                                    train,
                                    order=(p, d, q),
                                    seasonal_order=(P, D, Q, seasonal_periods) if seasonal else (0, 0, 0, 0),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                ).fit(disp=False)
                                if fit.aic < best_aic:
                                    best_aic = fit.aic
                                    best_fit = fit
                                    best_order = (p, d, q)
                                    best_seasonal = (P, D, Q, seasonal_periods) if seasonal else (0, 0, 0, 0)
                            except Exception:
                                continue

    if best_fit is None:
        best_fit = SARIMAX(
            train, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)
        best_order = (1, 1, 1)
        best_seasonal = (0, 0, 0, 0)

    return best_fit, best_order, best_seasonal


def fit_prophet(train: pd.Series, horizon: int, freq: str) -> Tuple[object, pd.Series, pd.Series]:
    """
    Fit Prophet model and generate forecasts.
    For daily data, emphasizes weekly seasonality (day-of-week patterns).
    
    Args:
        train: Training time series
        horizon: Number of periods to forecast
        freq: Frequency string ('Daily', 'Weekly', 'Monthly')
    
    Returns:
        Tuple of (model, fitted_values, forecast)
    """
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet is not installed. Install with: pip install prophet")
    
    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    df_train = pd.DataFrame({
        'ds': train.index,
        'y': train.values
    })
    
    # Set seasonality based on frequency
    # For daily data, strongly emphasize weekly seasonality (day-of-week patterns)
    if freq == "Daily":
        model = Prophet(
            yearly_seasonality=False,  # Disable yearly for daily
            weekly_seasonality=True,   # Enable weekly (Sun, Mon, Tue, Wed, Thu, Fri, Sat)
            daily_seasonality=False,
            seasonality_mode='additive',
            seasonality_prior_scale=10.0  # Higher prior scale to emphasize weekly pattern
        )
        # Add custom weekly seasonality with more flexibility
        model.add_seasonality(name='weekly', period=7, fourier_order=3)
    elif freq == "Weekly":
        model = Prophet(
            yearly_seasonality=(len(train) >= 52),
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='additive'
        )
    else:  # Monthly
        model = Prophet(
            yearly_seasonality=(len(train) >= 24),
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='additive'
        )
    
    model.fit(df_train)
    
    # Get fitted values
    fitted_df = model.predict(df_train)
    fitted_vals = pd.Series(fitted_df['yhat'].values, index=train.index)
    
    # Generate future dates
    freq_map = {"Daily": "D", "Weekly": "W-SUN", "Monthly": "MS"}
    forecast_freq = freq_map.get(freq, "D")
    future_dates = pd.date_range(
        start=train.index.max(),
        periods=horizon + 1,
        freq=forecast_freq
    )[1:]
    
    future_df = pd.DataFrame({'ds': future_dates})
    forecast_df = model.predict(future_df)
    forecast = pd.Series(forecast_df['yhat'].values, index=future_dates)
    
    return model, fitted_vals, forecast


def fit_xgboost(train: pd.Series, horizon: int, freq: str) -> Tuple[object, pd.Series, pd.Series]:
    """
    Fit XGBoost model and generate forecasts using time-based features.
    
    Args:
        train: Training time series
        horizon: Number of periods to forecast
        freq: Frequency string ('Daily', 'Weekly', 'Monthly')
    
    Returns:
        Tuple of (model, fitted_values, forecast)
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
    
    # Create features from time index
    def create_features(series: pd.Series, is_future: bool = False) -> pd.DataFrame:
        df = pd.DataFrame(index=series.index)
        
        # Time-based features
        if isinstance(series.index, pd.DatetimeIndex):
            # Day of week (0=Monday, 6=Sunday) - CRITICAL for daily forecasts
            df['day_of_week'] = series.index.dayofweek
            df['day_of_month'] = series.index.day
            df['month'] = series.index.month
            df['day_of_year'] = series.index.dayofyear
            
            # Week of year (handle edge cases)
            try:
                df['week_of_year'] = series.index.isocalendar().week
            except:
                df['week_of_year'] = series.index.week
            
            # For daily data, add day-of-week as one-hot encoded features (more emphasis)
            if freq == "Daily":
                # Day of week as categorical (0-6)
                df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
                df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
                # Individual day indicators (binary features for each day)
                for day_num in range(7):
                    df[f'is_dow_{day_num}'] = (df['day_of_week'] == day_num).astype(int)
                # Weekend indicator
                df['is_weekend'] = (df['day_of_week'].isin([5, 6]).astype(int))
            
            # Lag features (only for training)
            if not is_future:
                df['value'] = series.values
                # For daily, include 7-day lag to capture weekly pattern
                lag_list = [1, 2, 3, 7] if freq == "Daily" else [1, 2, 3]
                for lag in lag_list:
                    if len(series) > lag:
                        df[f'lag_{lag}'] = series.shift(lag).values
                
                # Rolling statistics - emphasize 7-day patterns for daily
                if freq == "Daily":
                    windows = [3, 7, 14]  # 3-day, weekly, bi-weekly
                elif freq == "Weekly":
                    windows = [3, 4, 13]  # 3-week, monthly, quarterly
                else:
                    windows = [3, 6, 12]  # 3-month, half-year, yearly
                
                for window in windows:
                    if len(series) > window:
                        df[f'rolling_mean_{window}'] = series.rolling(window=window, min_periods=1).mean().values
                        df[f'rolling_std_{window}'] = series.rolling(window=window, min_periods=1).std().fillna(0).values
        else:
            # If not datetime, use simple numeric features
            df['index'] = range(len(series))
            if not is_future:
                df['value'] = series.values
                for lag in [1, 2, 3]:
                    if len(series) > lag:
                        df[f'lag_{lag}'] = series.shift(lag).values
        
        return df
    
    # Prepare training data
    train_features = create_features(train, is_future=False)
    train_features = train_features.dropna()
    
    if len(train_features) < 5:
        # Fallback to simple model if not enough data
        last_val = train.iloc[-1]
        fitted_vals = pd.Series([last_val] * len(train), index=train.index)
        freq_map = {"Daily": "D", "Weekly": "W-SUN", "Monthly": "MS"}
        forecast_freq = freq_map.get(freq, "D")
        forecast = pd.Series([last_val] * horizon, index=pd.date_range(
            start=train.index.max(), periods=horizon + 1, freq=forecast_freq
        )[1:])
        return None, fitted_vals, forecast
    
    X_train = train_features.drop('value', axis=1)
    y_train = train_features['value']
    
    # Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Get fitted values
    fitted_pred = model.predict(X_train)
    fitted_vals = pd.Series(fitted_pred, index=train_features.index)
    # Fill missing indices with last known value
    full_fitted = pd.Series(index=train.index, dtype=float)
    full_fitted.loc[train_features.index] = fitted_vals
    full_fitted = full_fitted.ffill().fillna(train.iloc[0])
    fitted_vals = full_fitted
    
    # Generate forecast
    freq_map = {"Daily": "D", "Weekly": "W-SUN", "Monthly": "MS"}
    forecast_freq = freq_map.get(freq, "D")
    future_dates = pd.date_range(
        start=train.index.max(),
        periods=horizon + 1,
        freq=forecast_freq
    )[1:]
    
    # Use recursive forecasting with last known values
    forecast_values = []
    current_series = train.copy()
    
    for i, next_date in enumerate(future_dates):
        # Use recent values for lag features
        recent_vals = current_series.iloc[-3:].values if len(current_series) >= 3 else current_series.values
        
        # Create a temporary series with the next date
        temp_series = pd.concat([current_series, pd.Series([current_series.iloc[-1]], index=[next_date])])
        next_features = create_features(temp_series, is_future=True)
        next_features = next_features.loc[[next_date]]
        
        # Add lag features manually for future dates
        if freq == "Daily":
            lag_list = [1, 2, 3, 7]  # Include 7-day lag for weekly pattern
        else:
            lag_list = [1, 2, 3]
        
        for lag in lag_list:
            if len(current_series) >= lag:
                next_features[f'lag_{lag}'] = current_series.iloc[-lag]
            else:
                next_features[f'lag_{lag}'] = current_series.iloc[-1]
        
        # Add day-of-week features for daily forecasts (if not already in next_features)
        if freq == "Daily" and isinstance(next_date, pd.Timestamp):
            dow = next_date.dayofweek
            if 'dow_sin' not in next_features.columns:
                next_features['dow_sin'] = np.sin(2 * np.pi * dow / 7)
                next_features['dow_cos'] = np.cos(2 * np.pi * dow / 7)
            for day_num in range(7):
                if f'is_dow_{day_num}' not in next_features.columns:
                    next_features[f'is_dow_{day_num}'] = 1 if dow == day_num else 0
            if 'is_weekend' not in next_features.columns:
                next_features['is_weekend'] = 1 if dow in [5, 6] else 0
        
        # Add rolling statistics
        if freq == "Daily":
            windows = [3, 7, 14]  # 3-day, weekly, bi-weekly
        elif freq == "Weekly":
            windows = [3, 4, 13]  # 3-week, monthly, quarterly
        else:
            windows = [3, 6, 12]  # 3-month, half-year, yearly
        
        for window in windows:
            if len(current_series) >= window:
                next_features[f'rolling_mean_{window}'] = current_series.iloc[-window:].mean()
                next_features[f'rolling_std_{window}'] = current_series.iloc[-window:].std() if window > 1 else 0
            else:
                next_features[f'rolling_mean_{window}'] = current_series.mean()
                next_features[f'rolling_std_{window}'] = current_series.std() if len(current_series) > 1 else 0
        
        # Ensure all training features exist
        for col in X_train.columns:
            if col not in next_features.columns:
                next_features[col] = X_train[col].iloc[-1] if len(X_train) > 0 else 0
        
        # Reorder columns to match training
        next_features = next_features[X_train.columns]
        
        # Predict
        pred = model.predict(next_features)[0]
        forecast_values.append(max(0, pred))  # Ensure non-negative
        current_series = pd.concat([current_series, pd.Series([pred], index=[next_date])])
    
    forecast = pd.Series(forecast_values, index=future_dates)
    
    return model, fitted_vals, forecast


def evaluate_forecast(y_test: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    y_test = y_test.astype(float)
    y_pred = y_pred.astype(float)
    return {
        "MAE": mae(y_test, y_pred),
        "RMSE": rmse(y_test, y_pred),
        "MAPE%": mape(y_test, y_pred),
        "sMAPE%": smape(y_test, y_pred),
        "WAPE%": wape(y_test, y_pred),
    }


def compute_requirements(
    forecast_df: pd.DataFrame,
    aht_minutes: float,
    shrinkage_pct: float,
    paid_hours_per_period: float,
    starting_backlog: float,
    target_end_backlog: float,
    sla_target_pct: float,
    sla_buffer_pct: float,
    concurrency: Optional[float] = None,
) -> pd.DataFrame:
    """Convert forecast volume into FTE requirements"""
    out = forecast_df.copy()

    shrink = max(0.0, min(0.95, shrinkage_pct / 100.0))
    productive_hours = max(0.01, paid_hours_per_period * (1.0 - shrink))

    out["Forecast_Volume"] = out["Forecast_Volume"].clip(lower=0).round()
    out["AHT_min"] = aht_minutes
    
    # For Chat with concurrency: FTE accounts for agents handling multiple chats simultaneously
    # For Email/Voice: standard AHT-based calculation
    if concurrency is not None and concurrency > 0:
        # Chat concurrency model: 
        # Total chat-minutes = Volume * AHT
        # With concurrency, agent-minutes needed = (Volume * AHT) / Concurrency
        # Agent-hours = agent-minutes / 60
        total_chat_minutes = out["Forecast_Volume"] * aht_minutes
        agent_minutes_needed = total_chat_minutes / concurrency
        out["Workload_Hours_New"] = agent_minutes_needed / 60.0
    else:
        # Standard AHT-based calculation for Email/Voice
        out["Workload_Hours_New"] = (out["Forecast_Volume"] * aht_minutes) / 60.0

    periods = len(out)
    backlog_to_clear = max(0.0, starting_backlog - target_end_backlog)
    backlog_per_period = backlog_to_clear / periods if periods > 0 else 0.0

    out["Backlog_Clear_Volume"] = backlog_per_period
    if concurrency is not None and concurrency > 0:
        # For chat: backlog clearance also uses concurrency
        backlog_chat_minutes = out["Backlog_Clear_Volume"] * aht_minutes
        backlog_agent_minutes = backlog_chat_minutes / concurrency
        out["Workload_Hours_Backlog"] = backlog_agent_minutes / 60.0
    else:
        out["Workload_Hours_Backlog"] = (out["Backlog_Clear_Volume"] * aht_minutes) / 60.0

    out["SLA_Target_%"] = sla_target_pct
    out["SLA_Buffer_%"] = sla_buffer_pct
    buffer_mult = 1.0 + max(0.0, sla_buffer_pct / 100.0)

    out["Total_Required_Hours"] = (out["Workload_Hours_New"] + out["Workload_Hours_Backlog"]) * buffer_mult
    out["PaidHours_per_Period"] = paid_hours_per_period
    out["Shrinkage_%"] = shrinkage_pct
    out["ProductiveHours_per_Period"] = productive_hours
    out["Required_FTE"] = out["Total_Required_Hours"] / productive_hours
    
    if concurrency is not None:
        out["Concurrency"] = concurrency

    return out


# ----------------------------
# Header with JustAnswer Branding
# ----------------------------
st.markdown(
    """
    <div style="background: linear-gradient(135deg, rgba(20,184,166,0.1) 0%, rgba(56,189,248,0.15) 100%); 
                border-radius: 20px; 
                padding: 40px 48px; 
                margin-bottom: 32px;
                border: 1px solid rgba(20,184,166,0.2);
                box-shadow: 0 4px 12px rgba(20,184,166,0.1);">
      <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 16px;">
        <span style="font-size: 14px;
                      font-weight: 600;
                      color: #475569;
                      padding: 6px 14px;
                      border: 1px solid rgba(20,184,166,0.25);
                      border-radius: 20px;
                      background-color: rgba(255,255,255,0.9);">
          This tool is built for JA
        </span>
        <span style="background: linear-gradient(135deg, #14B8A6, #38BDF8);
                      -webkit-background-clip: text;
                      -webkit-text-fill-color: transparent;
                      background-clip: text;
                      font-size: 14px;
                      font-weight: 600;
                      padding: 6px 14px;
                      border: 1px solid rgba(20,184,166,0.25);
                      border-radius: 20px;
                      background-color: rgba(255,255,255,0.9);">
          WFM Forecasting
        </span>
      </div>
      <div style="font-size: 32px; 
                  font-weight: 800; 
                  letter-spacing: -0.02em;
                  background: linear-gradient(135deg, #14B8A6, #38BDF8);
                  -webkit-background-clip: text;
                  -webkit-text-fill-color: transparent;
                  background-clip: text;
                  margin-bottom: 12px;">
        Forecasting Tool with Requirement Calculator
      </div>
      <div style="font-size: 16px; 
                  color: #475569; 
                  line-height: 1.6;
                  max-width: 800px;">
        Simple forecasting and FTE planning for Email, Voice, and Chat LOBs. Start with sample data or upload your own.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Main Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(
    ["1) Forecast", "2) Requirements", "3) Guide"]
)

# ----------------------------
# Tab 1: Forecast (Simplified - Data + Forecast combined)
# ----------------------------
with tab1:
    st.markdown(
        """
        <div class="card">
          <b>Step 1: Choose your channel and load data</b><br>
          Select Email, Voice, or Chat, then use sample data or upload your own file
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Enhanced LOB and Frequency Selection
    st.markdown("#### 🎯 Select your options")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            <div class="filter-card">
              <div class="filter-label">📧 Channel</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        lob_type = st.radio(
            "Channel",
            ["Email", "Voice", "Chat"],
            index=0,
            help="Choose the channel you want to forecast",
            label_visibility="collapsed",
            horizontal=True,
            key="channel_radio"
        )
    
    with col2:
        st.markdown(
            """
            <div class="filter-card">
              <div class="filter-label">📅 Planning Frequency</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        freq = st.radio(
            "Planning Frequency",
            ["Daily", "Weekly", "Monthly"],
            index=0,
            help="Daily for detailed forecasting, Weekly for operations, Monthly for planning",
            label_visibility="collapsed",
            horizontal=True,
            key="freq_radio"
        )

    # Data Source
    st.markdown("#### 📥 Data Source")
    use_sample = st.radio(
        "Choose data source:",
        ["Use sample data (recommended to start)", "Upload my own file"],
        index=0,
        horizontal=True
    )

    if use_sample.startswith("Use sample"):
        raw = sample_data(lob_type, freq)
        st.success(f"✅ Using sample {lob_type} data ({freq.lower()}) - {len(raw)} periods")
    else:
        upload = st.file_uploader(
            f"Upload {lob_type} volume data (CSV/XLSX)",
            type=["csv", "xlsx", "xls"],
            help="File should have Date and Volume columns"
        )
        if upload is None:
            st.info("👆 Please upload a file or switch to sample data")
            st.stop()
        raw = read_file(upload)
        st.success(f"✅ File loaded: {upload.name}")

    # Column mapping (only if uploaded)
    if not use_sample.startswith("Use sample"):
        st.markdown("#### 🧹 Map columns")
        c1, c2 = st.columns(2)
        with c1:
            date_col = st.selectbox("Date column", options=list(raw.columns), index=0)
        with c2:
            vol_col = st.selectbox(
                "Volume column",
                options=list(raw.columns),
                index=min(1, len(raw.columns) - 1),
            )
    else:
        date_col = "Date"
        vol_col = raw.columns[1]  # Second column is volume

    # Aggregate data
    agg = aggregate_series(raw, date_col=date_col, vol_col=vol_col, freq=freq)

    st.markdown("#### ✅ Your data")
    agg_styled = agg.style.format({"Volume": "{:,.0f}"})
    st.dataframe(agg_styled, use_container_width=True, height=200)

    fig = make_line_chart(agg, "Period", ["Volume"], f"{lob_type} Volume ({freq})")
    st.plotly_chart(fig, use_container_width=True)

    # Anomaly Detection Section
    st.markdown("#### 🔍 Anomaly Detection")
    col_anom1, col_anom2 = st.columns(2)
    with col_anom1:
        st.markdown(
            """
            <div class="filter-card">
              <div class="filter-label">🔍 Anomaly Detection</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        enable_anomaly_detection = st.radio(
            "Enable anomaly detection",
            ["Disabled", "Enabled"],
            index=0,
            help="Detect and optionally remove anomalies from the data using Z-Score method",
            label_visibility="collapsed",
            horizontal=True,
            key="anomaly_radio"
        )
        enable_anomaly_detection = (enable_anomaly_detection == "Enabled")
    with col_anom2:
        st.markdown(
            """
            <div class="filter-card">
              <div class="filter-label">Z-Score Threshold</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        anomaly_threshold = st.number_input(
            "Z-Score threshold",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.1,
            disabled=not enable_anomaly_detection,
            help="Standard deviation threshold. Higher values = fewer anomalies detected (default: 3.0)",
            label_visibility="collapsed"
        )
    
    include_anomalies_in_forecast = True
    anomalies_detected = pd.Series([False] * len(agg), index=agg["Period"])
    anomalies_df = pd.DataFrame()
    
    if enable_anomaly_detection:
        series_for_anomaly = agg.set_index("Period")["Volume"].astype(float)
        anomalies_detected = detect_anomalies(
            series_for_anomaly,
            method="zscore",
            threshold=anomaly_threshold
        )
        
        if anomalies_detected.any():
            anomalies_df = agg[anomalies_detected.values].copy()
            st.session_state["anomalies_df"] = anomalies_df  # Store in session state
            st.markdown("#### ⚠️ Detected Anomalies")
            st.warning(f"Found {anomalies_detected.sum()} anomaly/anomalies in your data")
            anomalies_styled = anomalies_df.style.format({"Volume": "{:,.0f}"})
            st.dataframe(anomalies_styled, use_container_width=True, height=150)
            
            st.markdown(
                """
                <div class="filter-card">
                  <div class="filter-label">📈 Include Anomalies in Forecast</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            include_anomalies_choice = st.radio(
                "Include anomalies in forecast",
                ["Exclude", "Include"],
                index=0,
                help="If Exclude is selected, anomalies will be removed from the data before forecasting",
                label_visibility="collapsed",
                horizontal=True,
                key="include_anomalies_radio"
            )
            include_anomalies_in_forecast = (include_anomalies_choice == "Include")
        else:
            st.info("✅ No anomalies detected with current settings")
            # Clear anomalies from session state if none detected
            if "anomalies_df" in st.session_state:
                del st.session_state["anomalies_df"]
    
    # Filter out anomalies if requested
    if enable_anomaly_detection and anomalies_detected.any() and not include_anomalies_in_forecast:
        agg_clean = agg[~anomalies_detected.values].copy()
        st.success(f"✅ Removed {anomalies_detected.sum()} anomaly/anomalies. Using {len(agg_clean)} periods for forecasting.")
        agg = agg_clean

    # Forecast Configuration
    st.markdown("#### 🔮 Forecast settings")
    
    # Model Selection with Simple Filter Card
    st.markdown(
        """
        <div class="filter-card">
          <div class="filter-label">🤖 Forecasting Model</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    model_options = [
        "ETS (Holt-Winters)",
        "ARIMA (auto grid)",
        "Moving Average",
        "Weighted Moving Average"
    ]
    
    # Add Prophet if available
    if PROPHET_AVAILABLE:
        model_options.append("Prophet")
    else:
        model_options.append("Prophet (not installed)")
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        model_options.append("XGBoost")
    else:
        model_options.append("XGBoost (not installed)")
    
    model_choice = st.selectbox(
        "Model",
        model_options,
        index=0,
        help="Choose forecasting model. Moving averages are simpler and faster. Prophet and XGBoost require additional packages.",
        label_visibility="collapsed"
    )
    
    # Display selected model in highlighted format
    st.markdown(
        f"""
        <div class="filter-card" style="margin-top: 8px;">
          <div class="filter-value model-selected">{model_choice}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    colB, colC = st.columns(2)
    with colB:
        max_horizon = 365 if freq == "Daily" else 104
        default_horizon = 30 if freq == "Daily" else 12
        horizon = st.number_input(
            "Forecast periods", min_value=1, max_value=max_horizon, value=default_horizon, step=1
        )
    with colC:
        holdout = st.number_input(
            "Holdout (for testing)",
            min_value=1,
            max_value=max(2, len(agg) // 2),
            value=min(8, max(2, len(agg) // 4)),
            step=1,
        )
    
    # Moving Average window parameter (only shown for MA models)
    ma_window = None
    if "Moving Average" in model_choice:
        ma_window = st.number_input(
            "Moving Average window",
            min_value=2,
            max_value=min(20, len(agg) - holdout - 1),
            value=min(4, len(agg) - holdout - 1),
            step=1,
            help="Number of periods to average (higher = smoother but slower to react)"
        )

    series = agg.set_index("Period")["Volume"].astype(float)

    if len(series) <= holdout + 4:
        st.error("Not enough history. Reduce holdout or add more data.")
        st.stop()

    train = series.iloc[:-holdout]
    test = series.iloc[-holdout:]

    run = st.button("🚀 Generate Forecast", type="primary", use_container_width=True)

    if run:
        with st.spinner("Fitting model…"):
            seasonal_periods = 7 if freq == "Daily" else (52 if freq == "Weekly" else 12)
            seasonality_mode = "add"
            arima_seasonal = True

            if model_choice.startswith("ETS"):
                fit, fitted_vals = fit_ets(
                    train, seasonal_periods=int(seasonal_periods), seasonality=seasonality_mode
                )
                test_fc = fit.forecast(holdout)
                fc = fit.forecast(int(horizon))
                details = {
                    "Model": "Exponential Smoothing (Holt-Winters)",
                    "Trend": "add",
                    "Seasonal": seasonality_mode,
                    "Seasonal periods": str(seasonal_periods),
                }
                model_name = "ETS"
            elif model_choice.startswith("ARIMA"):
                fit, order, seas = auto_arima_grid(
                    train,
                    seasonal_periods=int(seasonal_periods),
                    seasonal=bool(arima_seasonal),
                )
                fitted_vals = fit.fittedvalues
                test_fc = fit.get_forecast(steps=holdout).predicted_mean
                fc = fit.get_forecast(steps=int(horizon)).predicted_mean
                details = {
                    "Model": "SARIMAX / ARIMA (grid auto)",
                    "Order (p,d,q)": str(order),
                    "Seasonal": str(seas),
                    "AIC": f"{fit.aic:.2f}",
                }
                model_name = "ARIMA"
            elif model_choice == "Moving Average":
                if ma_window is None or ma_window < 2:
                    st.error("Please set a valid Moving Average window (≥2)")
                    st.stop()
                fitted_vals, fc_ma = moving_average(train, int(ma_window), int(horizon))
                # Test forecast: use last window average (same as future forecast)
                last_avg = train.iloc[-int(ma_window):].mean()
                test_fc = pd.Series([last_avg] * holdout, index=test.index)
                fc = fc_ma
                details = {
                    "Model": "Simple Moving Average",
                    "Window": str(ma_window),
                }
                model_name = "MA"
            elif model_choice == "Weighted Moving Average":
                if ma_window is None or ma_window < 2:
                    st.error("Please set a valid Moving Average window (≥2)")
                    st.stop()
                fitted_vals, fc_wma = weighted_moving_average(train, int(ma_window), int(horizon))
                # Test forecast: use last window weighted average
                weights = np.linspace(1, int(ma_window), int(ma_window))
                weights = weights / weights.sum()
                last_weighted_avg = (train.iloc[-int(ma_window):] * weights).sum()
                test_fc = pd.Series([last_weighted_avg] * holdout, index=test.index)
                fc = fc_wma
                details = {
                    "Model": "Weighted Moving Average",
                    "Window": str(ma_window),
                }
                model_name = "WMA"
            elif model_choice == "Prophet" or model_choice.startswith("Prophet"):
                if not PROPHET_AVAILABLE:
                    st.error("Prophet is not installed. Please install it with: pip install prophet")
                    st.stop()
                try:
                    fit_prophet_model, fitted_vals, fc_prophet = fit_prophet(train, int(horizon), freq)
                    # For test forecast, use Prophet to forecast holdout period
                    _, _, test_fc_prophet = fit_prophet(train, holdout, freq)
                    test_fc = test_fc_prophet
                    fc = fc_prophet
                    details = {
                        "Model": "Prophet",
                        "Yearly seasonality": str(freq == "Monthly" and len(train) >= 24),
                        "Weekly seasonality": str(freq == "Daily" and len(train) >= 14),
                    }
                    model_name = "Prophet"
                except Exception as e:
                    st.error(f"Error fitting Prophet model: {str(e)}")
                    st.stop()
            elif model_choice == "XGBoost" or model_choice.startswith("XGBoost"):
                if not XGBOOST_AVAILABLE:
                    st.error("XGBoost is not installed. Please install it with: pip install xgboost")
                    st.stop()
                try:
                    fit_xgb_model, fitted_vals, fc_xgb = fit_xgboost(train, int(horizon), freq)
                    # For test forecast, use XGBoost to forecast holdout period
                    _, _, test_fc_xgb = fit_xgboost(train, holdout, freq)
                    test_fc = test_fc_xgb
                    fc = fc_xgb
                    details = {
                        "Model": "XGBoost",
                        "Features": "Time-based + Lags + Rolling stats",
                    }
                    model_name = "XGBoost"
                except Exception as e:
                    st.error(f"Error fitting XGBoost model: {str(e)}")
                    st.stop()
            else:
                st.error(f"Unknown model: {model_choice}")
                st.stop()

            # Set index for forecast
            freq_map = {"Daily": "D", "Weekly": "W-SUN", "Monthly": "MS"}
            forecast_freq = freq_map.get(freq, "D")
            
            # Ensure indices are set correctly for all models
            if model_choice in ["Moving Average", "Weighted Moving Average"]:
                # For MA models, indices are already set
                test_fc.index = test.index
            elif model_choice in ["Prophet", "XGBoost"]:
                # Prophet and XGBoost already set indices in their functions
                # Just ensure test_fc index matches test
                if not test_fc.index.equals(test.index):
                    test_fc.index = test.index
            else:
                # For other models (ETS, ARIMA)
                test_fc.index = test.index
                fc.index = pd.date_range(
                    start=series.index.max(),
                    periods=int(horizon) + 1,
                    freq=forecast_freq,
                )[1:]

            test_fc = test_fc.clip(lower=0)
            fc = fc.clip(lower=0)

            errs = evaluate_forecast(test, test_fc)

            st.session_state["lob_type"] = lob_type
            st.session_state["freq"] = freq
            st.session_state["forecast_model"] = model_name
            st.session_state["forecast_details"] = details
            st.session_state["test_pred"] = test_fc
            st.session_state["forecast"] = fc
            st.session_state["errors"] = errs
            st.session_state["train"] = train
            st.session_state["test"] = test
            # Anomalies are already stored in session state during detection

        st.success("✅ Forecast ready!")

    if "forecast" in st.session_state:
        errs = st.session_state["errors"]
        lob_type = st.session_state.get("lob_type", "Email")
        train = st.session_state["train"]
        test = st.session_state["test"]
        test_pred = st.session_state["test_pred"]
        fc = st.session_state["forecast"]

        st.markdown("#### 📏 Forecast accuracy")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("MAE", f"{errs['MAE']:.2f}")
        m2.metric("RMSE", f"{errs['RMSE']:.2f}")
        m3.metric("MAPE", f"{errs['MAPE%']:.2f}%")
        m4.metric("sMAPE", f"{errs['sMAPE%']:.2f}%")
        m5.metric("WAPE", f"{errs['WAPE%']:.2f}%")
        
        # Simple explanation of forecast accuracy
        wape_val = errs['WAPE%']
        if wape_val < 10:
            accuracy_level = "Excellent"
            accuracy_color = "#10B981"
            recommendation = "This model is very accurate. You can confidently use it for planning."
        elif wape_val < 15:
            accuracy_level = "Good"
            accuracy_color = "#14B8A6"
            recommendation = "This model is accurate enough for most planning decisions."
        elif wape_val < 25:
            accuracy_level = "Acceptable"
            accuracy_color = "#F59E0B"
            recommendation = "This model is reasonable but consider trying other models or checking your data."
        else:
            accuracy_level = "Needs Improvement"
            accuracy_color = "#EF4444"
            recommendation = "Try a different model or check if your data has patterns the model can learn from."
        
        st.markdown(
            f"""
            <div class="card card-forecast">
              <b>What do these numbers mean?</b><br>
              • <b>WAPE</b> (Weighted Absolute Percentage Error): <b style="color: {accuracy_color};">{wape_val:.2f}%</b> - <b>{accuracy_level}</b><br>
              • This tells you how far off the forecast is on average. Lower is better<br>
              • <b>MAE</b> and <b>RMSE</b> show absolute errors (in volume units)<br>
              • <b>MAPE</b> and <b>sMAPE</b> show percentage errors (can be unstable with low volumes)<br><br>
              <b>How to pick the best method:</b><br>
              • Compare WAPE across different models - lower is better<br>
              • <b>WAPE under 15%</b> is usually good for operations<br>
              • If WAPE is high, try a different model or check your data quality<br>
              • <b>Moving Averages</b> are simple but may miss trends<br>
              • <b>ETS</b> is best for seasonal patterns<br>
              • <b>ARIMA</b> is most flexible but can be slower<br>
              • <b>Prophet</b> excels at daily data with day-of-week patterns and holidays<br>
              • <b>XGBoost</b> captures complex non-linear patterns using machine learning<br>
              <b>Recommendation:</b> {recommendation}
            </div>
            """,
            unsafe_allow_html=True,
        )

        next_period = float(fc.iloc[0])
        avg_future = float(fc.mean())
        peak_future = float(fc.max())
        peak_period = fc.idxmax()
        if isinstance(peak_period, pd.Timestamp):
            peak_label = peak_period.date()
        else:
            peak_label = str(peak_period)

        st.markdown(
            f"""
            <div class="card card-forecast">
              <b>Forecast summary</b><br>
              • <b>Next period:</b> {next_period:,.0f} {lob_type.lower()}<br>
              • <b>Average:</b> {avg_future:,.0f} per period<br>
              • <b>Peak:</b> {peak_future:,.0f} around {peak_label}<br><br>
              <b>Tip:</b> Use peak volume for busy-week staffing, average for long-term planning
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 📈 Forecast chart")
        hist_df = (
            pd.concat([train.rename("Actual"), test.rename("Actual")])
            .reset_index()
            .rename(columns={"index": "Period"})
        )
        test_df = pd.DataFrame({"Period": test.index, "Test_Pred": test_pred.values})
        fc_df = pd.DataFrame({"Period": fc.index, "Forecast": fc.values})

        plot_df = (
            hist_df.merge(test_df, on="Period", how="left")
            .merge(fc_df, on="Period", how="left")
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=plot_df["Period"], y=plot_df["Actual"], mode="lines+markers", name="Actual"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df["Period"],
                y=plot_df["Test_Pred"],
                mode="lines+markers",
                name="Test Pred",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df["Period"],
                y=plot_df["Forecast"],
                mode="lines+markers",
                name="Forecast",
            )
        )
        fig.update_layout(
            template="plotly_white",
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#1E293B"),
            title=f"{lob_type} Forecast",
            height=460,
            margin=dict(l=30, r=30, t=70, b=40),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Display forecast table
        st.markdown("#### 📋 Forecast Table")
        forecast_table_df = fc_df.copy()
        forecast_table_df["Forecast"] = forecast_table_df["Forecast"].round(2)
        
        # Add day of week for daily forecasts
        if freq == "Daily" and isinstance(forecast_table_df["Period"].iloc[0], pd.Timestamp):
            forecast_table_df["Day"] = pd.to_datetime(forecast_table_df["Period"]).dt.day_name()
            forecast_table_df = forecast_table_df[["Period", "Day", "Forecast"]]
            forecast_table_styled = forecast_table_df.style.format({"Forecast": "{:,.2f}"})
        else:
            forecast_table_styled = forecast_table_df.style.format({"Forecast": "{:,.2f}"})
        
        st.dataframe(forecast_table_styled, use_container_width=True, height=300)

        # Download buttons for forecast table
        st.markdown("#### ⬇️ Download Forecast Table")
        col_dl_daily, col_dl_weekly, col_dl_monthly = st.columns(3)
        
        with col_dl_daily:
            # Daily forecast download
            daily_fc_download = forecast_table_df.copy()
            daily_fc_download["Period"] = pd.to_datetime(daily_fc_download["Period"])
            # Include Day column if it exists (for daily forecasts)
            if "Day" in daily_fc_download.columns:
                daily_fc_download = daily_fc_download[["Period", "Day", "Forecast"]]
            else:
                daily_fc_download = daily_fc_download[["Period", "Forecast"]]
            daily_csv = daily_fc_download.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Daily Forecast",
                data=daily_csv,
                file_name=f"{lob_type.lower()}_daily_forecast.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_dl_weekly:
            # Weekly forecast download (aggregate if daily, or use existing if weekly)
            if freq == "Daily":
                if "weekly_forecast" in st.session_state:
                    weekly_fc_download = st.session_state["weekly_forecast"].copy()
                    weekly_fc_download["Period"] = pd.to_datetime(weekly_fc_download["Period"])
                    weekly_fc_download = weekly_fc_download[["Period", "Forecast"]]
                else:
                    # Aggregate on the fly
                    daily_fc_download = forecast_table_df.copy()
                    daily_fc_download["Period"] = pd.to_datetime(daily_fc_download["Period"])
                    weekly_fc_download = aggregate_daily_to_weekly(daily_fc_download, "Period", "Forecast")
                    weekly_fc_download = weekly_fc_download.rename(columns={"Volume": "Forecast"})
                    weekly_fc_download = weekly_fc_download[["Period", "Forecast"]]
            elif freq == "Weekly":
                weekly_fc_download = forecast_table_df.copy()
                weekly_fc_download["Period"] = pd.to_datetime(weekly_fc_download["Period"])
                weekly_fc_download = weekly_fc_download[["Period", "Forecast"]]
            else:
                # For monthly, aggregate to weekly if possible
                weekly_fc_download = pd.DataFrame(columns=["Period", "Forecast"])
            
            if not weekly_fc_download.empty:
                weekly_csv = weekly_fc_download.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Weekly Forecast",
                    data=weekly_csv,
                    file_name=f"{lob_type.lower()}_weekly_forecast.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("Weekly forecast not available")
        
        with col_dl_monthly:
            # Monthly forecast download (aggregate if daily/weekly, or use existing if monthly)
            if freq == "Daily":
                if "monthly_forecast" in st.session_state:
                    monthly_fc_download = st.session_state["monthly_forecast"].copy()
                    monthly_fc_download["Period"] = pd.to_datetime(monthly_fc_download["Period"])
                    monthly_fc_download = monthly_fc_download[["Period", "Forecast"]]
                else:
                    # Aggregate on the fly
                    daily_fc_download = forecast_table_df.copy()
                    daily_fc_download["Period"] = pd.to_datetime(daily_fc_download["Period"])
                    monthly_fc_download = aggregate_daily_to_monthly(daily_fc_download, "Period", "Forecast")
                    monthly_fc_download = monthly_fc_download.rename(columns={"Volume": "Forecast"})
                    monthly_fc_download = monthly_fc_download[["Period", "Forecast"]]
            elif freq == "Weekly":
                # Aggregate weekly to monthly
                weekly_fc_download = forecast_table_df.copy()
                weekly_fc_download["Period"] = pd.to_datetime(weekly_fc_download["Period"])
                monthly_fc_download = aggregate_daily_to_monthly(weekly_fc_download, "Period", "Forecast")
                monthly_fc_download = monthly_fc_download.rename(columns={"Volume": "Forecast"})
                monthly_fc_download = monthly_fc_download[["Period", "Forecast"]]
            else:  # Monthly
                monthly_fc_download = forecast_table_df.copy()
                monthly_fc_download["Period"] = pd.to_datetime(monthly_fc_download["Period"])
                monthly_fc_download = monthly_fc_download[["Period", "Forecast"]]
            
            if not monthly_fc_download.empty:
                monthly_csv = monthly_fc_download.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Monthly Forecast",
                    data=monthly_csv,
                    file_name=f"{lob_type.lower()}_monthly_forecast.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("Monthly forecast not available")

        # Aggregation section for Daily forecasts
        if freq == "Daily":
            st.markdown("#### 📊 Aggregated Forecasts")
            st.info("Daily forecasts aggregated to weekly and monthly views")
            
            # Aggregate daily forecast to weekly
            daily_fc_df = pd.DataFrame({"Period": fc.index, "Forecast": fc.values})
            weekly_fc = aggregate_daily_to_weekly(daily_fc_df, "Period", "Forecast")
            weekly_fc = weekly_fc.rename(columns={"Volume": "Forecast"})
            
            # Aggregate daily forecast to monthly
            monthly_fc = aggregate_daily_to_monthly(daily_fc_df, "Period", "Forecast")
            monthly_fc = monthly_fc.rename(columns={"Volume": "Forecast"})
            
            col_agg1, col_agg2 = st.columns(2)
            
            with col_agg1:
                st.markdown("##### 📅 Weekly Aggregation")
                weekly_styled = weekly_fc.style.format({"Forecast": "{:,.0f}"})
                st.dataframe(weekly_styled, use_container_width=True, height=200)
                
                # Weekly chart
                fig_weekly = make_line_chart(weekly_fc, "Period", ["Forecast"], f"{lob_type} Weekly Forecast")
                st.plotly_chart(fig_weekly, use_container_width=True)
            
            with col_agg2:
                st.markdown("##### 📆 Monthly Aggregation")
                monthly_styled = monthly_fc.style.format({"Forecast": "{:,.0f}"})
                st.dataframe(monthly_styled, use_container_width=True, height=200)
                
                # Monthly chart
                fig_monthly = make_line_chart(monthly_fc, "Period", ["Forecast"], f"{lob_type} Monthly Forecast")
                st.plotly_chart(fig_monthly, use_container_width=True)
            
            # Store aggregated forecasts in session state
            st.session_state["weekly_forecast"] = weekly_fc
            st.session_state["monthly_forecast"] = monthly_fc

        st.markdown("#### ⬇️ Download forecast")
        forecast_export = plot_df.sort_values("Period")
        
        # CSV download
        forecast_csv = forecast_export.to_csv(index=False).encode("utf-8")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "Download Forecast CSV",
                data=forecast_csv,
                file_name=f"{lob_type.lower()}_forecast_{freq.lower()}.csv",
                mime="text/csv",
            )
        
        # Excel download
        with col_dl2:
            # Create Excel file with multiple sheets if daily
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Main sheet with all data (Actual, Test_Pred, Forecast)
                forecast_export.to_excel(writer, sheet_name=f"{freq} Forecast", index=False)
                # Dedicated sheet with only forecasted values
                fc_df.to_excel(writer, sheet_name="Forecast Only", index=False)
                if freq == "Daily" and "weekly_forecast" in st.session_state:
                    st.session_state["weekly_forecast"].to_excel(writer, sheet_name="Weekly Aggregation", index=False)
                if freq == "Daily" and "monthly_forecast" in st.session_state:
                    st.session_state["monthly_forecast"].to_excel(writer, sheet_name="Monthly Aggregation", index=False)
                # Add anomalies sheet if they exist
                if "anomalies_df" in st.session_state and not st.session_state["anomalies_df"].empty:
                    st.session_state["anomalies_df"].to_excel(writer, sheet_name="Removed Anomalies", index=False)
            
            excel_data = output.getvalue()
            st.download_button(
                "Download Forecast Excel",
                data=excel_data,
                file_name=f"{lob_type.lower()}_forecast_{freq.lower()}.xlsx",
                mime="application/vnd.openpyxl-officedocument.spreadsheetml.sheet",
            )

# ----------------------------
# Tab 2: Requirements
# ----------------------------
with tab2:
    st.markdown(
        """
        <div class="card card-req">
          <b>Step 2: Turn forecast into FTE requirements</b><br>
          Enter your AHT, shrinkage, and SLA settings to calculate staffing needs
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "forecast" not in st.session_state:
        st.warning("⚠️ Generate a forecast first in the **Forecast** tab.")
        st.stop()

    fc = st.session_state["forecast"].copy()
    freq = st.session_state["freq"]
    lob_type = st.session_state.get("lob_type", "Email")

    st.markdown("#### 🧮 Requirements calculator")

    # Default AHT by LOB
    default_aht = {"Email": 12.0, "Voice": 8.0, "Chat": 6.0}.get(lob_type, 12.0)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div class="card card-req">
              <b>What this does</b><br>
              {"• For Chat: Uses concurrency (chats per agent) to calculate FTE<br>" if lob_type == "Chat" else "• Converts forecast volume → hours using AHT<br>"}
              {"• For Email/Voice: Converts forecast volume → hours using AHT<br>" if lob_type != "Chat" else ""}
              • Adds backlog clearance if needed<br>
              • Applies SLA buffer for safety<br>
              • Converts to FTE using paid hours & shrinkage
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        # Show concurrency for Chat, AHT for Email/Voice
        concurrency = None
        if lob_type == "Chat":
            concurrency = st.number_input(
                "Concurrency (chats per agent)",
                min_value=1.0,
                value=3.0,
                step=0.5,
                help="Average number of simultaneous chats one agent can handle"
            )
            aht_min = st.number_input(
                "AHT (minutes per chat)",
                min_value=0.1,
                value=default_aht,
                step=0.5,
                help="Average handle time in minutes (for reference)"
            )
        else:
            aht_min = st.number_input(
                f"AHT (minutes per {lob_type.lower()})",
                min_value=0.1,
                value=default_aht,
                step=0.5,
                help="Average handle time in minutes"
            )
        
        shrink_pct = st.number_input(
            "Shrinkage (%)",
            min_value=0.0,
            max_value=95.0,
            value=30.0,
            step=1.0,
            help="Time agents are paid but not handling contacts"
        )
        sla_buffer = st.number_input(
            "SLA buffer (%)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=1.0,
            help="Extra capacity to protect SLA"
        )
        if freq == "Daily":
            paid_hours = st.number_input(
                "Paid hours per agent per day",
                min_value=1.0,
                value=8.0,
                step=0.5,
            )
        elif freq == "Weekly":
            paid_hours = st.number_input(
                "Paid hours per agent per week",
                min_value=1.0,
                value=40.0,
                step=1.0,
            )
        else:  # Monthly
            paid_hours = st.number_input(
                "Paid hours per agent per month",
                min_value=1.0,
                value=160.0,
                step=5.0,
            )
        backlog_start = st.number_input(
            "Starting backlog",
            min_value=0.0,
            value=0.0,
            step=50.0,
            help="Current backlog to clear"
        )
        backlog_end = st.number_input(
            "Target end backlog",
            min_value=0.0,
            value=0.0,
            step=50.0,
        )

    forecast_df = pd.DataFrame({"Period": fc.index, "Forecast_Volume": fc.values})
    req = compute_requirements(
        forecast_df=forecast_df,
        aht_minutes=aht_min,
        shrinkage_pct=shrink_pct,
        paid_hours_per_period=paid_hours,
        starting_backlog=backlog_start,
        target_end_backlog=backlog_end,
        sla_target_pct=90.0,
        sla_buffer_pct=sla_buffer,
        concurrency=concurrency,
    )

    total_volume = float(req["Forecast_Volume"].sum())
    total_hours = float(req["Total_Required_Hours"].sum())
    avg_fte = float(req["Required_FTE"].mean())
    peak_fte = float(req["Required_FTE"].max())

    st.markdown("#### 📊 Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Forecast", f"{total_volume:,.0f}")
    m2.metric("Total Hours", f"{total_hours:,.1f}")
    m3.metric("Avg FTE", f"{avg_fte:,.2f}")
    m4.metric("Peak FTE", f"{peak_fte:,.2f}")

    peak_idx = req["Required_FTE"].idxmax()
    peak_row = req.loc[peak_idx]
    peak_date = peak_row["Period"]

    st.markdown(
        f"""
        <div class="card card-req">
          <b>Key takeaway</b><br>
          • Average FTE needed: <b>{avg_fte:,.2f}</b><br>
          • Peak FTE needed: <b>{peak_fte:,.2f}</b><br>
          • Peak occurs: <b>{pd.to_datetime(peak_date).date()}</b><br><br>
          <b>Action:</b> Plan staffing to at least the peak FTE for busy periods
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 📋 Requirements by period")
    show_cols = [
        "Period",
        "Forecast_Volume",
        "Total_Required_Hours",
        "Required_FTE",
    ]
    out_df = req[show_cols].copy()
    out_df["Period"] = pd.to_datetime(out_df["Period"])

    req_styled = out_df.style.format(
        {
            "Forecast_Volume": "{:,.0f}",
            "Total_Required_Hours": "{:,.1f}",
            "Required_FTE": "{:,.2f}",
        }
    )
    st.dataframe(req_styled, use_container_width=True, height=300)

    chart_df = out_df.copy()
    chart_df["Period"] = chart_df["Period"].dt.strftime("%Y-%m-%d")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_df["Period"],
            y=out_df["Required_FTE"],
            mode="lines+markers",
            name="Required FTE",
        )
    )
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#1E293B"),
        title="Required FTE per Period",
        height=420,
        margin=dict(l=30, r=30, t=70, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### ⬇️ Download requirements")
    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Requirements CSV",
        data=csv_bytes,
        file_name=f"{lob_type.lower()}_requirements_{freq.lower()}.csv",
        mime="text/csv",
    )

# ----------------------------
# Tab 3: Guide
# ----------------------------
with tab3:
    st.markdown(
        """
        <div class="card">
          <b>Quick guide</b><br>
          Simple explanations to help you use this tool effectively
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 📖 How to use")

    st.markdown("##### 1. Forecast tab")
    st.markdown(
        """
        1. **Choose LOB**: Select Email, Voice, or Chat
        2. **Choose frequency**: Weekly for operations, Monthly for planning
        3. **Load data**: Use sample data (recommended) or upload your file
        4. **Set forecast**: Pick model, forecast length, and holdout
        5. **Generate**: Click the button to create your forecast
        6. **Review**: Check accuracy metrics and download if needed
        """
    )

    st.markdown("##### 2. Requirements tab")
    st.markdown(
        """
        1. **Enter AHT**: Average handle time in minutes (defaults provided by LOB)
        2. **Set shrinkage**: % of time agents are paid but not handling contacts
        3. **Add SLA buffer**: Extra capacity % to protect service levels
        4. **Set paid hours**: Hours per agent per week/month
        5. **Backlog (optional)**: Starting and target backlog if applicable
        6. **Review FTE**: See average and peak FTE needed, then download
        """
    )

    st.markdown("##### 3. Forecasting models")
    st.markdown(
        """
        - **ETS (Holt-Winters)**: Best for seasonal patterns, handles trends and seasonality
        - **ARIMA (auto grid)**: Advanced model that finds best parameters automatically
        - **Moving Average**: Simple and fast, averages last N periods (good for stable data)
        - **Weighted Moving Average**: Like MA but gives more weight to recent periods
        - **Prophet**: Facebook's forecasting tool, excellent for daily data with holidays and seasonality
        - **XGBoost**: Machine learning model using time-based features, lags, and rolling statistics

        **Which to choose?**
        - **Moving Averages**: Fastest, simplest, good for stable patterns
        - **ETS**: Best for seasonal data (weekly/monthly patterns)
        - **ARIMA**: Most flexible, best accuracy but slower
        - **Prophet**: Great for daily data with complex seasonality (requires: pip install prophet)
        - **XGBoost**: Powerful ML model, good for capturing non-linear patterns (requires: pip install xgboost)
        """
    )

    st.markdown("##### 4. Key terms")
    st.markdown(
        """
        - **AHT (Average Handle Time)**: Time to fully handle one contact, in minutes
        - **Shrinkage (%)**: Time agents are paid but unavailable (training, meetings, PTO, etc.)
        - **SLA buffer (%)**: Extra capacity on top of minimum to protect service levels
        - **FTE (Full-Time Equivalent)**: One full-time person; 2 half-time = 1.0 FTE
        - **Holdout**: Data hidden from model to test accuracy
        - **WAPE (%)**: Weighted absolute percentage error - main accuracy metric
        - **Moving Average Window**: Number of periods to average (higher = smoother)
        """
    )

    st.markdown("##### 5. Tips")
    st.markdown(
        """
        - **Start with sample data** to see how it works
        - **Use peak FTE** for busy-week staffing decisions
        - **Use average FTE** for long-term hiring and capacity planning
        - **WAPE under 15%** is usually good for operations
        - **Weekly data** works best with 52+ weeks of history
        - **Monthly data** works best with 24+ months of history
        """
    )

st.markdown(
    """
    <div style="text-align: center; 
                padding: 32px 24px; 
                margin-top: 48px;
                border-top: 1px solid rgba(20,184,166,0.15);
                background: linear-gradient(135deg, rgba(20,184,166,0.03), rgba(56,189,248,0.05));
                border-radius: 20px 20px 0 0;
                color: #475569;
                font-size: 14px;
                font-weight: 500;">
      <div style="font-size: 15px; font-weight: 600; color: #0F172A; margin-bottom: 6px;">
        Forecasting Tool with Requirement Calculator
      </div>
      <div style="font-size: 13px; color: #64748B; margin-top: 4px;">
        Simple, fast, reliable • This tool is built for JA
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
