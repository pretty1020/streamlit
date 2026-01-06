# app.py — Email LOB Forecasting + Requirements (Streamlit)
# --------------------------------------------------------
# Features:
# - Upload weekly or monthly email volume data (CSV/XLSX)
# - Clean + aggregate to chosen frequency (Weekly / Monthly)
# - Forecast using ETS (Holt-Winters) or ARIMA (auto grid)
# - Train/Test evaluation + error measures (MAE, RMSE, MAPE, sMAPE, WAPE)
# - Requirements calculator using AHT + Shrinkage + Paid Hours + Backlog + SLA buffer
# - Light gradient theme, simple explanations, and exports
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
DEFAULT_PRIMARY = "#14B8A6"   # bright teal (left side of gradient)
DEFAULT_ACCENT = "#38BDF8"    # sky-blue (right side of gradient)
DEFAULT_BG = "#F0FDFA"        # light teal-tinted background
DEFAULT_PANEL = "#FFFFFF"     # white
DEFAULT_TEXT = "#0F172A"      # dark slate (readable on light)
DEFAULT_MUTED = "#475569"     # muted slate
DEFAULT_WARN = "#F59E0B"      # amber
DEFAULT_DANGER = "#EF4444"    # red

st.set_page_config(
    page_title="Email LOB Forecasting • Email Forecasting Tool for JA",
    page_icon="📨",
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

        /* Gradient cards for forecast and requirements */
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

        /* Enhanced tabs: JustAnswer teal-to-sky-blue gradient theme */
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

        /* Plotly charts - light theme */
        .js-plotly-plot {{
          background: #FFFFFF !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# Apply theme
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


def sample_weekly() -> pd.DataFrame:
    rng = pd.date_range("2025-01-05", periods=52, freq="W-SUN")
    np.random.seed(42)
    base = 2200 + 250 * np.sin(np.linspace(0, 2 * np.pi, len(rng)))
    noise = np.random.normal(0, 140, size=len(rng))
    vol = np.maximum(200, (base + noise)).round().astype(int)
    return pd.DataFrame({"Date": rng, "Emails": vol})


def sample_monthly() -> pd.DataFrame:
    rng = pd.date_range("2023-01-01", periods=36, freq="MS")
    np.random.seed(7)
    trend = np.linspace(16000, 22000, len(rng))
    season = 1200 * np.sin(np.linspace(0, 2 * np.pi, len(rng)))
    noise = np.random.normal(0, 800, size=len(rng))
    vol = np.maximum(500, (trend + season + noise)).round().astype(int)
    return pd.DataFrame({"Date": rng, "Emails": vol})


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

    out = s.reset_index().rename(columns={date_col: "Period", vol_col: "Emails"})
    return out


@dataclass
class ForecastResult:
    model_name: str
    fitted_values: pd.Series
    forecast: pd.Series
    test_pred: Optional[pd.Series]
    errors: Dict[str, float]
    details: Dict[str, str]


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
) -> pd.DataFrame:
    """
    Simple requirement logic:
    - Turn forecast emails into hours: (Emails × AHT) ÷ 60
    - Spread any backlog you want to clear across all periods
    - Add an SLA buffer on top
    - Turn required hours into FTE using paid hours and shrinkage
    """
    out = forecast_df.copy()

    shrink = max(0.0, min(0.95, shrinkage_pct / 100.0))
    productive_hours = max(0.01, paid_hours_per_period * (1.0 - shrink))

    out["Forecast_Emails"] = out["Forecast_Emails"].clip(lower=0).round()
    out["AHT_min"] = aht_minutes
    out["Workload_Hours_New"] = (out["Forecast_Emails"] * aht_minutes) / 60.0

    periods = len(out)
    backlog_to_clear = max(0.0, starting_backlog - target_end_backlog)
    backlog_per_period = backlog_to_clear / periods if periods > 0 else 0.0

    out["Backlog_Clear_Emails"] = backlog_per_period
    out["Workload_Hours_Backlog"] = (out["Backlog_Clear_Emails"] * aht_minutes) / 60.0

    out["SLA_Target_%"] = sla_target_pct
    out["SLA_Buffer_%"] = sla_buffer_pct
    buffer_mult = 1.0 + max(0.0, sla_buffer_pct / 100.0)

    out["Total_Required_Hours"] = (out["Workload_Hours_New"] + out["Workload_Hours_Backlog"]) * buffer_mult
    out["PaidHours_per_Period"] = paid_hours_per_period
    out["Shrinkage_%"] = shrinkage_pct
    out["ProductiveHours_per_Period"] = productive_hours
    out["Required_FTE"] = out["Total_Required_Hours"] / productive_hours

    return out


# ----------------------------
# Header
# ----------------------------
st.markdown(
    """
    <div class="card">
      <div class="title">
        📨 Email LOB Forecasting & Requirements
        <span class="badge">Tool built for JA</span>
      </div>
      <div class="subtitle">
        Upload weekly or monthly email volumes, create a forecast, then turn it into clear FTE requirements.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Main Tabs
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["1) Data", "2) Forecast", "3) Requirements", "4) User Guide"]
)

# ----------------------------
# Tab 1: Data
# ----------------------------
with tab1:
    st.markdown(
        """
        <div class="card">
          <b>Data tab</b><br>
          Load your email history and tell the tool which columns are the date and the email volume.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 📥 Load your data")
    left, right = st.columns([1.2, 1])

    with left:
        upload = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx", "xls"])
        use_sample = st.toggle("Use sample data instead", value=(upload is None))
        sample_type = st.selectbox(
            "Sample type", ["Weekly", "Monthly"], index=0, disabled=not use_sample
        )

    with right:
        st.markdown(
            """
            <div class="card">
              <b>Expected columns</b><br>
              • Date column (any standard date format)<br>
              • Volume column (emails received per week/month OR raw daily records)<br><br>
              <b>Example</b><br>
              Date, Emails<br>
              2025-01-05, 2310<br>
              2025-01-12, 2198<br>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if use_sample:
        raw = sample_weekly() if sample_type == "Weekly" else sample_monthly()
        st.info("Using a built-in sample dataset. Upload your own file any time to replace it.")
    else:
        if upload is None:
            st.stop()
        raw = read_file(upload)

    st.markdown("#### 🧹 Map columns")
    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        date_col = st.selectbox("Date column", options=list(raw.columns), index=0)
    with c2:
        vol_col = st.selectbox(
            "Volume column (Emails)",
            options=list(raw.columns),
            index=min(1, len(raw.columns) - 1),
        )
    with c3:
        freq = st.selectbox("Your planning frequency", ["Weekly", "Monthly"], index=0)

    agg = aggregate_series(raw, date_col=date_col, vol_col=vol_col, freq=freq)

    st.markdown("#### ✅ Cleaned time series")
    agg_styled = agg.style.format({"Emails": "{:,.0f}"})
    st.dataframe(agg_styled, use_container_width=True, height=260)

    fig = make_line_chart(agg, "Period", ["Emails"], f"Incoming Emails ({freq})")
    st.plotly_chart(fig, use_container_width=True)

    st.session_state["agg"] = agg
    st.session_state["freq"] = freq

# ----------------------------
# Tab 2: Forecast
# ----------------------------
with tab2:
    st.markdown(
        """
        <div class="card card-forecast">
          <b>Forecast tab</b><br>
          Choose a model, forecast length, and test window, then generate a volume forecast for your email LOB.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "agg" not in st.session_state:
        st.warning("Load data first in **1) Data**.")
        st.stop()

    agg = st.session_state["agg"].copy()
    freq = st.session_state["freq"]

    st.markdown("#### 🔮 Forecast configuration")
    colA, colB, colC, colD = st.columns([1, 1, 1, 1])

    with colA:
        model_choice = st.selectbox(
            "Model", ["ETS (Holt-Winters)", "ARIMA (auto grid)"], index=0
        )
    with colB:
        horizon = st.number_input(
            "Forecast periods", min_value=1, max_value=104, value=12, step=1
        )
    with colC:
        holdout = st.number_input(
            "Holdout (periods for testing)",
            min_value=1,
            max_value=max(2, len(agg) // 2),
            value=min(8, max(2, len(agg) // 4)),
            step=1,
        )
    with colD:
        clamp_zero = st.toggle("Clamp negative forecasts to 0", value=True)

    series = agg.set_index("Period")["Emails"].astype(float)

    with st.expander("Seasonality settings (optional)", expanded=False):
        if freq == "Weekly":
            seasonal_periods = st.number_input(
                "Season length", min_value=2, max_value=104, value=52, step=1
            )
        else:
            seasonal_periods = st.number_input(
                "Season length", min_value=2, max_value=60, value=12, step=1
            )
        seasonality_mode = st.selectbox(
            "ETS seasonality type", ["add", "mul", "none"], index=0
        )
        arima_seasonal = st.toggle("ARIMA seasonal component", value=True)

    if len(series) <= holdout + 4:
        st.error("Not enough history for the chosen holdout. Reduce holdout or add more data.")
        st.stop()

    train = series.iloc[:-holdout]
    test = series.iloc[-holdout:]

    run = st.button("🚀 Run Forecast", type="primary")

    if run:
        with st.spinner("Fitting model…"):
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
            else:
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

            test_fc.index = test.index
            fc.index = pd.date_range(
                start=series.index.max(),
                periods=int(horizon) + 1,
                freq=("W-SUN" if freq == "Weekly" else "MS"),
            )[1:]

            if clamp_zero:
                test_fc = test_fc.clip(lower=0)
                fc = fc.clip(lower=0)

            errs = evaluate_forecast(test, test_fc)

            st.session_state["forecast_model"] = model_name
            st.session_state["forecast_details"] = details
            st.session_state["test_pred"] = test_fc
            st.session_state["forecast"] = fc
            st.session_state["errors"] = errs
            st.session_state["train"] = train
            st.session_state["test"] = test

        st.success("Forecast ready.")

    if "forecast" in st.session_state:
        errs = st.session_state["errors"]
        details = st.session_state["forecast_details"]
        train = st.session_state["train"]
        test = st.session_state["test"]
        test_pred = st.session_state["test_pred"]
        fc = st.session_state["forecast"]

        st.markdown("#### 📏 Model accuracy (holdout)")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("MAE", f"{errs['MAE']:.2f}")
        m2.metric("RMSE", f"{errs['RMSE']:.2f}")
        m3.metric("MAPE", f"{errs['MAPE%']:.2f}%")
        m4.metric("sMAPE", f"{errs['sMAPE%']:.2f}%")
        m5.metric("WAPE", f"{errs['WAPE%']:.2f}%")

        # High-level forecast summary & recommendation
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
              <b>In simple terms</b><br>
              • The <b>next</b> period forecast is about <b>{next_period:,.0f} emails</b>.<br>
              • On average over the forecast window, you expect about <b>{avg_future:,.0f} emails per period</b>.<br>
              • The highest point in the forecast is about <b>{peak_future:,.0f} emails</b> around <b>{peak_label}</b>.<br><br>
              <b>How to use this</b><br>
              • Use the <b>peak</b> volume to size staffing for the busiest weeks. <br>
              • Use the <b>average</b> volume to talk about longer‑term trends and hiring.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 📈 Forecast view")
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
            title="Actual vs Predicted vs Forecast",
            height=460,
            margin=dict(l=30, r=30, t=70, b=40),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "We hide some history (holdout) to check how well the model predicts the past. "
            "Those errors help you judge if the forecast is good enough."
        )

        # Download forecast data
        st.markdown("#### ⬇️ Download forecast data")
        forecast_export = plot_df.sort_values("Period")
        forecast_csv = forecast_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Forecast CSV",
            data=forecast_csv,
            file_name=f"email_forecast_{freq.lower()}.csv",
            mime="text/csv",
        )

        with st.expander("Model details"):
            st.json(details)

# ----------------------------
# Tab 3: Requirements
# ----------------------------
with tab3:
    st.markdown(
        """
        <div class="card card-req">
          <b>Requirements tab</b><br>
          Turn the volume forecast into FTE by applying your AHT, shrinkage, backlog and SLA buffer.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "forecast" not in st.session_state:
        st.warning("Run a forecast first in **2) Forecast**.")
        st.stop()

    fc = st.session_state["forecast"].copy()
    freq = st.session_state["freq"]

    st.markdown("#### 🧮 Requirements calculator (AHT + SLA + Shrinkage)")

    left, right = st.columns([1.05, 1])

    with left:
        st.markdown(
            """
            <div class="card card-req">
              <b>What this does</b><br>
              • Turns your <b>forecasted emails</b> into hours using AHT (minutes per email).<br>
              • Lets you spread any <b>backlog</b> you want to clear across the forecast window.<br>
              • Adds an extra <b>SLA buffer %</b> on top, so you are not always on the edge.<br>
              • Converts total hours into <b>FTE</b> using paid hours and shrinkage.<br><br>
              This gives you a simple, readable FTE plan per week or month.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        c1, c2, c3 = st.columns(3)
        with c1:
            aht_min = st.number_input(
                "AHT (minutes per email)", min_value=0.1, value=12.0, step=0.5
            )
        with c2:
            shrink_pct = st.number_input(
                "Shrinkage (%)", min_value=0.0, max_value=95.0, value=30.0, step=1.0
            )
        with c3:
            sla_target = st.number_input(
                "SLA Target (%)", min_value=1.0, max_value=100.0, value=90.0, step=1.0
            )

        d1, d2, d3 = st.columns(3)
        with d1:
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
        with d2:
            sla_buffer = st.number_input(
                "SLA buffer (%)",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=1.0,
                help="Extra capacity to cover spikes, reopens, and normal variation.",
            )
        with d3:
            backlog_start = st.number_input(
                "Starting backlog (emails)", min_value=0.0, value=0.0, step=50.0
            )

        e1, e2 = st.columns(2)
        with e1:
            backlog_end = st.number_input(
                "Target end backlog (emails)", min_value=0.0, value=0.0, step=50.0
            )
        with e2:
            round_up = st.toggle("Round FTE up (ceiling)", value=True)

    forecast_df = pd.DataFrame({"Period": fc.index, "Forecast_Emails": fc.values})
    req = compute_requirements(
        forecast_df=forecast_df,
        aht_minutes=aht_min,
        shrinkage_pct=shrink_pct,
        paid_hours_per_period=paid_hours,
        starting_backlog=backlog_start,
        target_end_backlog=backlog_end,
        sla_target_pct=sla_target,
        sla_buffer_pct=sla_buffer,
    )

    if round_up:
        req["Required_FTE_Rounded"] = np.ceil(req["Required_FTE"])
    else:
        req["Required_FTE_Rounded"] = req["Required_FTE"]

    total_emails = float(req["Forecast_Emails"].sum())
    total_hours = float(req["Total_Required_Hours"].sum())
    avg_fte = float(req["Required_FTE"].mean())
    peak_fte = float(req["Required_FTE"].max())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Forecast Emails", f"{total_emails:,.0f}")
    m2.metric("Total Required Hours", f"{total_hours:,.1f}")
    m3.metric("Average FTE", f"{avg_fte:,.2f}")
    m4.metric("Peak FTE", f"{peak_fte:,.2f}")

    # Requirement explanation & recommendation
    peak_idx = req["Required_FTE"].idxmax()
    peak_row = req.loc[peak_idx]
    peak_date = peak_row["Period"]
    peak_req = float(peak_row["Required_FTE"])
    peak_req_round = float(peak_row["Required_FTE_Rounded"])

    st.markdown(
        f"""
        <div class="card card-req">
          <b>What this means for you</b><br>
          • On average across the forecast, you need about <b>{avg_fte:,.2f} FTE</b> to keep up with email volume.<br>
          • The busiest period needs about <b>{peak_req:,.2f} FTE</b>, which we round to <b>{peak_req_round:,.0f} FTE</b> for scheduling.<br>
          • That peak happens around <b>{pd.to_datetime(peak_date).date()}</b>.<br><br>
          <b>Simple guidance</b><br>
          • Plan staffing at least to the <b>peak FTE</b> in your known busy weeks. <br>
          • Use the <b>average FTE</b> when you talk about long‑term hiring and capacity.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 📋 Requirements by period")

    show_cols = [
        "Period",
        "Forecast_Emails",
        "Workload_Hours_New",
        "Workload_Hours_Backlog",
        "Total_Required_Hours",
        "Required_FTE",
        "Required_FTE_Rounded",
    ]
    out_df = req[show_cols].copy()
    out_df["Period"] = pd.to_datetime(out_df["Period"])

    req_styled = out_df.style.format(
        {
            "Forecast_Emails": "{:,.0f}",
            "Workload_Hours_New": "{:,.1f}",
            "Workload_Hours_Backlog": "{:,.1f}",
            "Total_Required_Hours": "{:,.1f}",
            "Required_FTE": "{:,.2f}",
            "Required_FTE_Rounded": "{:,.0f}",
        }
    )
    st.dataframe(req_styled, use_container_width=True, height=360)

    chart_df = out_df.copy()
    chart_df["Period"] = chart_df["Period"].dt.strftime("%Y-%m-%d")

    fig1 = make_line_chart(chart_df, "Period", ["Forecast_Emails"], "Forecast Emails (by period)")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(
        go.Bar(
            x=chart_df["Period"],
            y=out_df["Total_Required_Hours"],
            name="Total Required Hours",
        )
    )
    fig2.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#1E293B"),
        title="Total Required Hours (New + Backlog) with SLA Buffer",
        height=420,
        margin=dict(l=30, r=30, t=70, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = go.Figure()
    fig3.add_trace(
        go.Scatter(
            x=chart_df["Period"],
            y=out_df["Required_FTE_Rounded"],
            mode="lines+markers",
            name="Required FTE (Rounded)",
        )
    )
    fig3.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#1E293B"),
        title="Required FTE per Period",
        height=420,
        margin=dict(l=30, r=30, t=70, b=40),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Export requirements
    st.markdown("#### ⬇️ Download requirements")
    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Requirements CSV",
        data=csv_bytes,
        file_name=f"email_requirements_{freq.lower()}.csv",
        mime="text/csv",
    )

# ----------------------------
# Tab 4: User Guide
# ----------------------------
with tab4:
    st.markdown(
        """
        <div class="card">
          <b>User Guide tab</b><br>
          Plain‑language explanations of the steps and the key terms used in this tool.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 📖 User Guide")

    st.markdown("##### 1. Simple workflow")
    st.markdown(
        """
        1. **Data tab** – Load your email volume file (or use the sample) and map the columns.  
        2. **Forecast tab** – Pick a model, forecast length, and holdout size, then click **Run Forecast**.  
        3. **Requirements tab** – Enter AHT, shrinkage, backlog and SLA buffer, then read the FTE results.  
        4. Use the download buttons to pull data into Excel or your planning deck.
        """
    )

    st.markdown("##### 2. Key terms (plain language)")
    st.markdown(
        """
        - **AHT (Average Handle Time)**: Average time to fully work one email, in minutes.  
        - **Shrinkage (%)**: Time people are paid but not working emails (training, meetings, PTO, coaching, etc.).  
        - **Paid hours per period**: Contracted hours per person per week or month (for example, 40 hours/week).  
        - **Backlog**: Emails waiting in the queue that are not yet answered.  
        - **SLA Target (%)**: Your service goal (for example, 90% answered within 24 hours).  
        - **SLA buffer (%)**: Extra headroom on top of the minimum needed to hit the SLA.  
        - **FTE (Full-Time Equivalent)**: One full‑time person; two half‑time people are 1.0 FTE.  
        - **Forecast horizon**: How many future weeks or months you want a forecast for.  
        """
    )

    st.markdown("##### 3. Forecast accuracy metrics")
    st.markdown(
        """
        - **MAE (Mean Absolute Error)**: Average absolute error in email counts (smaller is better).  
        - **RMSE (Root Mean Squared Error)**: Like MAE but punishes big misses more.  
        - **MAPE (%)**: Average percentage error, but can be noisy when volume is very low.  
        - **sMAPE (%)**: A more balanced version of MAPE for over‑ vs under‑forecasting.  
        - **WAPE (%)**: Very stable percentage error that works well for operations; good main KPI.  
        """
    )

    st.markdown(
        """
        **Practical tip:** Focus on whether the forecast gets the **shape** right (peaks and dips)
        and a reasonable **WAPE**, not on trying to make the error numbers perfect.
        """
    )

st.caption(
    "Built for email WFM: backlog-aware, SLA-buffered, and simple enough to operate weekly or monthly."
)

