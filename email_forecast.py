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

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

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
          background: linear-gradient(90deg, rgba(20,184,166,0.12) 0%, rgba(56,189,248,0.08) 100%),
                      radial-gradient(1200px 700px at 10% 0%, rgba(20,184,166,0.1), transparent 55%),
                      radial-gradient(900px 500px at 90% 10%, rgba(56,189,248,0.08), transparent 50%),
                      linear-gradient(180deg, #F0FDFA 0%, #ECFEFF 50%, #E0F2FE 100%);
          color: var(--text);
          font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}

        html, body, [class*="css"] {{
          color: var(--text) !important;
          font-size: 14px;
        }}

        section[data-testid="stSidebar"] {{
          background: linear-gradient(180deg, #FFFFFF 0%, #F0FDFA 100%);
          border-right: 2px solid rgba(20,184,166,0.2);
          box-shadow: 2px 0 10px rgba(0,0,0,0.05);
        }}

        .card {{
          background: linear-gradient(135deg, #FFFFFF 0%, #F0FDFA 100%);
          border: 1px solid rgba(20,184,166,0.25);
          border-radius: 16px;
          padding: 20px 24px;
          box-shadow: 0 4px 20px rgba(0,0,0,0.08), 0 2px 8px rgba(20,184,166,0.15);
        }}

        .card-forecast {{
          background: linear-gradient(135deg, rgba(20,184,166,0.08) 0%, rgba(56,189,248,0.12) 100%);
          border-color: rgba(20,184,166,0.35);
          box-shadow: 0 4px 20px rgba(20,184,166,0.15), 0 2px 8px rgba(56,189,248,0.2);
        }}
        .card-req {{
          background: linear-gradient(135deg, rgba(20,184,166,0.08) 0%, rgba(56,189,248,0.12) 100%);
          border-color: rgba(20,184,166,0.35);
          box-shadow: 0 4px 20px rgba(20,184,166,0.15), 0 2px 8px rgba(56,189,248,0.2);
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
          background: linear-gradient(135deg, #FFFFFF 0%, #F0FDFA 100%);
          border: 1px solid rgba(20,184,166,0.25);
          padding: 16px 18px;
          border-radius: 14px;
          box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}

        .stButton>button {{
          border-radius: 12px;
          border: none;
          background: linear-gradient(135deg, {primary}, {accent});
          color: white;
          font-weight: 700;
          padding: 0.6rem 1rem;
          box-shadow: 0 4px 15px rgba(20,184,166,0.35);
          transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
          transform: translateY(-2px);
          box-shadow: 0 6px 20px rgba(20,184,166,0.45);
        }}

        .stTextInput input, .stNumberInput input, .stSelectbox div, .stMultiSelect div {{
          background-color: #FFFFFF !important;
          border: 1px solid rgba(20,184,166,0.3) !important;
          border-radius: 12px !important;
          color: var(--text) !important;
          box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        .stTextInput input:focus, .stNumberInput input:focus {{
          border-color: {primary} !important;
          box-shadow: 0 0 0 3px rgba(20,184,166,0.15) !important;
        }}

        .stDataFrame, .stTable {{
          border-radius: 14px;
          overflow: hidden;
          border: 1px solid rgba(20,184,166,0.25);
          box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        .stDataFrame [role="grid"] {{
          font-size: 13px;
          color: var(--text);
        }}

        a {{
          color: var(--primary) !important;
        }}

        div[data-testid="stTabs"] > div[role="tablist"] {{
          gap: 0.5rem;
          background: linear-gradient(135deg, #F0FDFA 0%, #ECFEFF 100%);
          padding: 0.5rem;
          border-radius: 12px;
          box-shadow: inset 0 2px 5px rgba(0,0,0,0.05);
        }}

        div[data-testid="stTabs"] > div[role="tablist"] button[role="tab"] {{
          color: {text} !important;
          font-weight: 600;
          background: linear-gradient(135deg, #FFFFFF 0%, #F0FDFA 100%);
          border-radius: 999px;
          padding: 0.4rem 1rem;
          border: 1px solid rgba(20,184,166,0.25);
          box-shadow: 0 2px 5px rgba(0,0,0,0.05);
          transition: all 0.3s ease;
        }}

        div[data-testid="stTabs"] > div[role="tablist"] button[role="tab"]:hover {{
          background: linear-gradient(135deg, rgba(20,184,166,0.08), rgba(56,189,248,0.12));
          border-color: rgba(20,184,166,0.4);
          transform: translateY(-1px);
        }}

        div[data-testid="stTabs"] > div[role="tablist"] button[role="tab"][aria-selected="true"] {{
          background: linear-gradient(135deg, {primary}, {accent});
          border-color: {primary};
          color: #FFFFFF !important;
          box-shadow: 0 4px 15px rgba(20,184,166,0.4);
          transform: translateY(-2px);
        }}

        div[data-testid="stTabs"] > div[role="tablist"] button[role="tab"]:focus-visible {{
          outline: 2px solid {primary};
          outline-offset: 2px;
        }}

        .js-plotly-plot {{
          background: #FFFFFF !important;
        }}

        /* Enhanced filter cards with highlight */
        .filter-card {{
          background: linear-gradient(135deg, rgba(20,184,166,0.08) 0%, rgba(56,189,248,0.12) 100%);
          border: 2px solid rgba(20,184,166,0.4);
          border-radius: 12px;
          padding: 18px 22px;
          box-shadow: 0 4px 15px rgba(20,184,166,0.2), 0 2px 8px rgba(56,189,248,0.15);
          transition: all 0.3s ease;
          position: relative;
        }}
        .filter-card::before {{
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          height: 4px;
          background: linear-gradient(90deg, {primary}, {accent});
          border-radius: 12px 12px 0 0;
        }}
        .filter-card:hover {{
          border-color: rgba(20,184,166,0.6);
          box-shadow: 0 6px 20px rgba(20,184,166,0.3), 0 4px 12px rgba(56,189,248,0.2);
          transform: translateY(-2px);
        }}
        .filter-label {{
          font-size: 12px;
          font-weight: 700;
          color: {muted};
          margin-bottom: 10px;
          text-transform: uppercase;
          letter-spacing: 0.8px;
        }}
        .filter-value {{
          font-size: 20px;
          font-weight: 800;
          background: linear-gradient(135deg, {primary}, {accent});
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          margin-top: 4px;
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
    if freq == "Weekly":
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

    if freq == "Weekly":
        s = x.set_index(date_col)[vol_col].resample("W-SUN").sum()
    else:
        s = x.set_index(date_col)[vol_col].resample("MS").sum()

    out = s.reset_index().rename(columns={date_col: "Period", vol_col: "Volume"})
    return out


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
# Header
# ----------------------------
st.markdown(
    """
    <div class="card">
      <div class="title">
        📊 Forecasting Tool with Requirement Calc
        <span class="badge">built for JA</span>
      </div>
      <div class="subtitle">
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
          Select Email, Voice, or Chat, then use sample data or upload your own file.
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
        lob_type = st.selectbox(
            "Channel",
            ["Email", "Voice", "Chat"],
            index=0,
            help="Choose the channel you want to forecast",
            label_visibility="collapsed"
        )
        st.markdown(
            f"""
            <div style="margin-top: -10px; margin-bottom: 10px;">
              <span class="filter-value">{lob_type}</span>
            </div>
            """,
            unsafe_allow_html=True,
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
        freq = st.selectbox(
            "Planning Frequency",
            ["Weekly", "Monthly"],
            index=0,
            help="Weekly for operations, Monthly for planning",
            label_visibility="collapsed"
        )
        st.markdown(
            f"""
            <div style="margin-top: -10px; margin-bottom: 10px;">
              <span class="filter-value">{freq}</span>
            </div>
            """,
            unsafe_allow_html=True,
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

    # Forecast Configuration
    st.markdown("#### 🔮 Forecast settings")
    colA, colB, colC = st.columns(3)
    with colA:
        model_choice = st.selectbox(
            "Model",
            [
                "ETS (Holt-Winters)",
                "ARIMA (auto grid)",
                "Moving Average",
                "Weighted Moving Average"
            ],
            index=0,
            help="Choose forecasting model. Moving averages are simpler and faster."
        )
    with colB:
        horizon = st.number_input(
            "Forecast periods", min_value=1, max_value=104, value=12, step=1
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
            seasonal_periods = 52 if freq == "Weekly" else 12
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
            else:  # Weighted Moving Average
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

            # Set index for forecast
            if model_choice not in ["Moving Average", "Weighted Moving Average"]:
                test_fc.index = test.index
                fc.index = pd.date_range(
                    start=series.index.max(),
                    periods=int(horizon) + 1,
                    freq=("W-SUN" if freq == "Weekly" else "MS"),
                )[1:]
            else:
                # For MA models, create index
                test_fc.index = test.index
                fc.index = pd.date_range(
                    start=series.index.max(),
                    periods=int(horizon) + 1,
                    freq=("W-SUN" if freq == "Weekly" else "MS"),
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
              • This tells you how far off the forecast is on average. Lower is better.<br>
              • <b>MAE</b> and <b>RMSE</b> show absolute errors (in volume units).<br>
              • <b>MAPE</b> and <b>sMAPE</b> show percentage errors (can be unstable with low volumes).<br><br>
              <b>How to pick the best method:</b><br>
              • Compare WAPE across different models - lower is better<br>
              • <b>WAPE under 15%</b> is usually good for operations<br>
              • If WAPE is high, try a different model or check your data quality<br>
              • <b>Moving Averages</b> are simple but may miss trends<br>
              • <b>ETS</b> is best for seasonal patterns<br>
              • <b>ARIMA</b> is most flexible but can be slower<br><br>
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
              <b>Tip:</b> Use peak volume for busy-week staffing, average for long-term planning.
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

        st.markdown("#### ⬇️ Download forecast")
        forecast_export = plot_df.sort_values("Period")
        forecast_csv = forecast_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Forecast CSV",
            data=forecast_csv,
            file_name=f"{lob_type.lower()}_forecast_{freq.lower()}.csv",
            mime="text/csv",
        )

# ----------------------------
# Tab 2: Requirements
# ----------------------------
with tab2:
    st.markdown(
        """
        <div class="card card-req">
          <b>Step 2: Turn forecast into FTE requirements</b><br>
          Enter your AHT, shrinkage, and SLA settings to calculate staffing needs.
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
              • Converts to FTE using paid hours & shrinkage<br>
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
        if freq == "Weekly":
            paid_hours = st.number_input(
                "Paid hours per agent per week",
                min_value=1.0,
                value=40.0,
                step=1.0,
            )
        else:
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
          <b>Action:</b> Plan staffing to at least the peak FTE for busy periods.
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
          Simple explanations to help you use this tool effectively.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 📖 How to use")

    st.markdown("##### 1. Forecast tab")
    st.markdown(
        """
        1. **Choose LOB**: Select Email, Voice, or Chat<br>
        2. **Choose frequency**: Weekly for operations, Monthly for planning<br>
        3. **Load data**: Use sample data (recommended) or upload your file<br>
        4. **Set forecast**: Pick model, forecast length, and holdout<br>
        5. **Generate**: Click the button to create your forecast<br>
        6. **Review**: Check accuracy metrics and download if needed
        """
    )

    st.markdown("##### 2. Requirements tab")
    st.markdown(
        """
        1. **Enter AHT**: Average handle time in minutes (defaults provided by LOB)<br>
        2. **Set shrinkage**: % of time agents are paid but not handling contacts<br>
        3. **Add SLA buffer**: Extra capacity % to protect service levels<br>
        4. **Set paid hours**: Hours per agent per week/month<br>
        5. **Backlog (optional)**: Starting and target backlog if applicable<br>
        6. **Review FTE**: See average and peak FTE needed, then download
        """
    )

    st.markdown("##### 3. Forecasting models")
    st.markdown(
        """
        - **ETS (Holt-Winters)**: Best for seasonal patterns, handles trends and seasonality<br>
        - **ARIMA (auto grid)**: Advanced model that finds best parameters automatically<br>
        - **Moving Average**: Simple and fast, averages last N periods (good for stable data)<br>
        - **Weighted Moving Average**: Like MA but gives more weight to recent periods<br><br>
        <b>Which to choose?</b><br>
        • **Moving Averages**: Fastest, simplest, good for stable patterns<br>
        • **ETS**: Best for seasonal data (weekly/monthly patterns)<br>
        • **ARIMA**: Most flexible, best accuracy but slower
        """
    )

    st.markdown("##### 4. Key terms")
    st.markdown(
        """
        - **AHT (Average Handle Time)**: Time to fully handle one contact, in minutes<br>
        - **Shrinkage (%)**: Time agents are paid but unavailable (training, meetings, PTO, etc.)<br>
        - **SLA buffer (%)**: Extra capacity on top of minimum to protect service levels<br>
        - **FTE (Full-Time Equivalent)**: One full-time person; 2 half-time = 1.0 FTE<br>
        - **Holdout**: Data hidden from model to test accuracy<br>
        - **WAPE (%)**: Weighted absolute percentage error - main accuracy metric<br>
        - **Moving Average Window**: Number of periods to average (higher = smoother)
        """
    )

    st.markdown("##### 5. Tips")
    st.markdown(
        """
        - **Start with sample data** to see how it works<br>
        - **Use peak FTE** for busy-week staffing decisions<br>
        - **Use average FTE** for long-term hiring and capacity planning<br>
        - **WAPE under 15%** is usually good for operations<br>
        - **Weekly data** works best with 52+ weeks of history<br>
        - **Monthly data** works best with 24+ months of history
        """
    )

st.caption(
    "Forecasting Tool with Requirement Calc (built for JA) - Simple, fast, reliable."
)
