import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="Intraday Health Check",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Colorful modern CSS
# ----------------------------
st.markdown(
    """
<style>
/* App background */
[data-testid="stAppViewContainer"]{
  background: radial-gradient(1200px 700px at 20% 10%, rgba(91,44,131,.18) 0%, rgba(31,119,180,.12) 35%, rgba(255,255,255,1) 70%);
}
[data-testid="stHeader"]{
  background: rgba(255,255,255,0);
}

/* Header */
.main-header{
  font-size: 2.35rem;
  font-weight: 900;
  text-align:center;
  margin: 0.25rem 0 1.2rem 0;
  letter-spacing: .3px;
}
.sub-header{
  text-align:center;
  color:#4b5563;
  margin-top:-0.6rem;
  margin-bottom: 1.2rem;
}

/* Cards */
.card{
  background: rgba(255,255,255,0.85);
  border: 1px solid rgba(15,23,42,0.08);
  border-radius: 18px;
  padding: 1rem 1.1rem;
  box-shadow: 0 10px 30px rgba(2,6,23,0.06);
}
.card-title{
  font-weight: 800;
  color:#0f172a;
  margin:0 0 .25rem 0;
}
.badge{
  display:inline-block;
  padding: .25rem .55rem;
  border-radius: 999px;
  font-size: .75rem;
  font-weight: 700;
  border: 1px solid rgba(15,23,42,0.08);
  background: rgba(255,255,255,0.9);
}

/* KPI tiles */
.kpi-grid{
  display:grid;
  grid-template-columns: repeat(6, minmax(0,1fr));
  gap: .75rem;
}
.kpi{
  background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(255,255,255,0.82));
  border: 1px solid rgba(15,23,42,0.08);
  border-radius: 18px;
  padding: .85rem .95rem;
  box-shadow: 0 10px 25px rgba(2,6,23,0.06);
}
.kpi .label{ color:#475569; font-size:.78rem; font-weight:700; }
.kpi .value{ font-size:1.35rem; font-weight:900; color:#0f172a; margin-top:.1rem; }
.kpi .hint{ color:#64748b; font-size:.78rem; margin-top:.15rem; }

/* Sidebar polish */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(91,44,131,0.10), rgba(31,119,180,0.08), rgba(255,255,255,0.80));
  border-right: 1px solid rgba(15,23,42,0.08);
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p{
  color: #0f172a !important;
}

/* Alerts */
.alert-danger{
  background-color: rgba(220,53,69,.08);
  border-left: 6px solid #dc3545;
  padding: .8rem 1rem;
  border-radius: 14px;
}
.alert-warning{
  background-color: rgba(255,193,7,.18);
  border-left: 6px solid #ffc107;
  padding: .8rem 1rem;
  border-radius: 14px;
}
.alert-success{
  background-color: rgba(25,135,84,.10);
  border-left: 6px solid #198754;
  padding: .8rem 1rem;
  border-radius: 14px;
}
.small-note{
  color:#64748b;
  font-size:.85rem;
}
hr{
  border: none;
  height: 1px;
  background: rgba(15,23,42,0.10);
  margin: .7rem 0;
}

/* Make tabs look nicer */
.stTabs [data-baseweb="tab"]{
  font-weight: 800;
  border-radius: 999px;
  padding: .6rem .85rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Helpers
# ----------------------------
def generate_sample_data(days=1, interval_minutes=30):
    all_data = []
    base_date = datetime.now().date()

    for day_offset in range(days):
        current_date = base_date - timedelta(days=day_offset)

        intervals_per_day = (24 * 60) // interval_minutes
        intervals = []
        base_time = datetime.combine(current_date, datetime.min.time().replace(hour=8, minute=0))

        for i in range(intervals_per_day):
            interval_time = base_time + timedelta(minutes=interval_minutes * i)
            if interval_time.hour < 20:  # Only up to 8 PM
                intervals.append(interval_time.strftime("%H:%M"))

        num_intervals = len(intervals)

        volume_pattern = []
        for i in range(num_intervals):
            pos = i / max(num_intervals - 1, 1)
            peak1 = np.exp(-((pos - 0.25) ** 2) / 0.05)
            peak2 = np.exp(-((pos - 0.6) ** 2) / 0.05)
            base_level = 0.3 + 0.2 * np.sin(pos * np.pi)
            volume_pattern.append(base_level + 0.5 * peak1 + 0.4 * peak2)

        max_volume = max(volume_pattern) if volume_pattern else 1
        volume_pattern = [v / max_volume for v in volume_pattern]
        base_volume = 60

        aht_pattern = []
        for i in range(num_intervals):
            aht_multiplier = 0.9 + 0.2 * (volume_pattern[i] if i < len(volume_pattern) else 1)
            aht_pattern.append(aht_multiplier)

        base_aht = 300

        for i, interval in enumerate(intervals):
            volume_mult = volume_pattern[i] if i < len(volume_pattern) else 0.5
            aht_mult = aht_pattern[i] if i < len(aht_pattern) else 1.0

            calls = max(5, int(base_volume * volume_mult + np.random.randint(-8, 8)))
            aht = max(120, int(base_aht * aht_mult + np.random.randint(-15, 15)))

            rough_required = (calls * aht) / (interval_minutes * 60 * 0.7)
            net_staff = max(5, int(rough_required + np.random.randint(-3, 3)))

            all_data.append(
                {
                    "date": current_date.strftime("%Y-%m-%d"),
                    "interval": interval,
                    "calls": calls,
                    "aht": aht,
                    "interval_min": interval_minutes,
                    "net_staff": net_staff,
                }
            )

    return pd.DataFrame(all_data)


def parse_datetime(df, date_col, interval_col):
    try:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        interval_str = df[interval_col].astype(str).str.strip()
        interval_str = interval_str.str.replace(r"^(\d{2})(\d{2})$", r"\1:\2", regex=True)

        df["datetime"] = pd.to_datetime(
            df[date_col].dt.strftime("%Y-%m-%d") + " " + interval_str, errors="coerce"
        )

        if df["datetime"].isna().any():
            failed_rows = df[df["datetime"].isna()]
            st.warning(
                f"⚠️ Could not parse datetime for {len(failed_rows)} rows. Please check date/interval format."
            )
            st.dataframe(failed_rows[[date_col, interval_col]].head(), use_container_width=True)
            return None

        return df
    except Exception as e:
        st.error(f"❌ Error parsing datetime: {str(e)}")
        st.info(
            "Expected formats:\n- Date: YYYY-MM-DD (e.g., 2024-01-15)\n- Interval: HH:MM (e.g., 08:30) or HHMM (e.g., 0830)"
        )
        return None


def validate_data(df):
    issues = []
    df = df.copy()

    if df.duplicated().any():
        dup_count = int(df.duplicated().sum())
        issues.append(f"⚠️ Found {dup_count} duplicate rows. Removing duplicates.")
        df = df.drop_duplicates()

    critical_cols = ["date", "interval", "calls", "aht", "net_staff"]
    missing = df[critical_cols].isnull().sum()
    if missing.any():
        missing_dict = {col: int(count) for col, count in missing[missing > 0].items()}
        issues.append(f"⚠️ Missing values found: {missing_dict}. Dropping affected rows.")
        df = df.dropna(subset=critical_cols)

    if (df["calls"] < 0).any():
        issues.append("⚠️ Found negative call values. Setting to 0.")
        df.loc[df["calls"] < 0, "calls"] = 0

    if (df["aht"] <= 0).any():
        issues.append("⚠️ Found zero/negative AHT. Setting to minimum 60 seconds.")
        df.loc[df["aht"] <= 0, "aht"] = 60

    if (df["net_staff"] < 0).any():
        issues.append("⚠️ Found negative staff values. Setting to 0.")
        df.loc[df["net_staff"] < 0, "net_staff"] = 0

    if "datetime" in df.columns:
        df = df.sort_values("datetime").reset_index(drop=True)

        if len(df) > 1:
            expected_diff = pd.Timedelta(
                minutes=int(df["interval_min"].iloc[0]) if "interval_min" in df.columns else 30
            )
            gaps = df["datetime"].diff()
            gaps = gaps[gaps > expected_diff * 1.5]
            if len(gaps) > 0:
                issues.append(f"⚠️ Found {len(gaps)} potential missing intervals (time gaps).")

    return df, issues


def calculate_metrics(df, shrinkage_pct=0.30):
    df = df.copy()

    if "interval_min" not in df.columns or df["interval_min"].isna().any():
        st.error("❌ interval_min column is missing or has missing values.")
        return None

    if shrinkage_pct >= 1.0:
        st.error("❌ Shrinkage percentage must be less than 100%.")
        return None

    df["Workload_minutes"] = df["calls"] * (df["aht"] / 60.0)
    df["Required_FTE"] = df["Workload_minutes"] / df["interval_min"]
    df["Required_FTE_net"] = df["Required_FTE"] / (1 - shrinkage_pct)

    df["Variance_HC"] = df["net_staff"] - df["Required_FTE_net"]
    df["Variance_Pct"] = np.where(
        df["Required_FTE_net"] != 0,
        (df["Variance_HC"] / df["Required_FTE_net"]) * 100,
        0,
    )

    df['Occupancy'] = np.where(
        (df['net_staff'] * df['interval_min']) != 0,
        df['Workload_minutes'] / (df['net_staff'] * df['interval_min']),
        0
    )

    # Convert to % and CAP at 100%
    df['Occupancy'] = (df['Occupancy'] * 100).clip(0, 100)

    return df


def detect_alerts(
    df,
    understaffed_threshold=-10,
    volume_spike_multiplier=1.20,
    aht_spike_multiplier=1.15,
    occupancy_threshold=90,
):
    df = df.copy()

    if "datetime" in df.columns:
        df = df.sort_values("datetime").reset_index(drop=True)

    df["Understaffed"] = df["Variance_Pct"] < understaffed_threshold

    df["Volume_Rolling_Mean"] = df["calls"].rolling(window=4, min_periods=1).mean()
    df["Volume_Spike"] = df["calls"] > (df["Volume_Rolling_Mean"] * volume_spike_multiplier)

    df["AHT_Rolling_Mean"] = df["aht"].rolling(window=4, min_periods=1).mean()
    df["AHT_Spike"] = df["aht"] > (df["AHT_Rolling_Mean"] * aht_spike_multiplier)

    df["Occupancy_Risk"] = df["Occupancy"] > occupancy_threshold

    df["Consecutive_Breach"] = False
    if len(df) >= 3:
        for i in range(2, len(df)):
            if all(df.loc[df.index[i - 2 : i + 1], "Understaffed"]):
                df.loc[df.index[i], "Consecutive_Breach"] = True

    alert_rows = []
    for _, row in df.iterrows():
        alert_types = []
        if row["Understaffed"]:
            alert_types.append("Understaffed")
        if row["Volume_Spike"]:
            alert_types.append("Volume Spike")
        if row["AHT_Spike"]:
            alert_types.append("AHT Spike")
        if row["Occupancy_Risk"]:
            alert_types.append("Occupancy Risk")
        if row["Consecutive_Breach"]:
            alert_types.append("Consecutive Breach")

        if alert_types:
            alert_rows.append(
                {
                    "datetime": row.get("datetime", f"{row.get('date', 'N/A')} {row.get('interval', 'N/A')}"),
                    "date": row.get("date", "N/A"),
                    "interval": row.get("interval", "N/A"),
                    "alert_type": ", ".join(alert_types),
                    "calls": int(row["calls"]),
                    "aht": int(row["aht"]),
                    "net_staff": round(float(row["net_staff"]), 2),
                    "required_staff": round(float(row["Required_FTE_net"]), 2),
                    "variance_pct": round(float(row["Variance_Pct"]), 2),
                    "occupancy": round(float(row["Occupancy"]), 2),
                }
            )

    alerts_df = pd.DataFrame(alert_rows) if alert_rows else pd.DataFrame()
    return df, alerts_df


def format_seconds(sec):
    sec = float(sec)
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}m {s}s"


def build_basic_interpretation(df, alerts_df, understaffed_threshold, occupancy_threshold):
    # Simple “plain English” insights
    total_calls = int(df["calls"].sum())
    avg_aht = float(df["aht"].mean())
    peak_calls_idx = int(df["calls"].idxmax())
    peak_calls_interval = df.loc[peak_calls_idx, "interval"]
    peak_calls_value = int(df.loc[peak_calls_idx, "calls"])

    worst_var_idx = int(df["Variance_Pct"].idxmin())
    worst_var_pct = float(df.loc[worst_var_idx, "Variance_Pct"])
    worst_var_interval = df.loc[worst_var_idx, "interval"]

    peak_occ = float(df["Occupancy"].max())
    peak_occ_idx = int(df["Occupancy"].idxmax())
    peak_occ_interval = df.loc[peak_occ_idx, "interval"]

    # Time split: percent of intervals that are understaffed / risky occupancy
    pct_under = float((df["Variance_Pct"] < understaffed_threshold).mean() * 100)
    pct_occ_risk = float((df["Occupancy"] > occupancy_threshold).mean() * 100)

    # Top 3 risk intervals (most negative variance)
    top_risk = df.sort_values("Variance_Pct").head(3)[["interval", "Variance_Pct", "Variance_HC"]]

    # AHT / Volume relation (rough correlation)
    corr = df[["calls", "aht", "net_staff", "Required_FTE_net", "Variance_Pct", "Occupancy"]].corr(numeric_only=True)
    corr_calls_aht = float(corr.loc["calls", "aht"])
    corr_calls_occ = float(corr.loc["calls", "Occupancy"])

    # Alert counts
    alert_count = 0 if alerts_df is None or len(alerts_df) == 0 else len(alerts_df)

    # Recommendations in simple bullets
    recs = []
    if pct_under >= 25:
        recs.append("You’re understaffed for a big chunk of the day. Consider adding short OT blocks or moving breaks away from peak.")
    elif pct_under > 0:
        recs.append("There are pockets of understaffing. Small schedule tweaks (break/lunch shift) can usually fix it.")

    if peak_occ > max(occupancy_threshold, 90):
        recs.append("Occupancy is very high at peak. High occupancy usually means longer queues and stress—add coverage near peak.")
    elif pct_occ_risk > 0:
        recs.append("Some intervals have high occupancy. Watch those windows so queues don’t snowball.")

    if corr_calls_aht > 0.25:
        recs.append("When calls go up, AHT also trends up. That can be a sign of complexity or longer holds during peak.")
    if corr_calls_occ > 0.25:
        recs.append("Call spikes are strongly tied to occupancy spikes—peak management (staffing or deflection) will help.")

    if alert_count == 0:
        status_block = '<div class="alert-success"><b>✅ Overall status:</b> Looks stable today. No alerts triggered with your current thresholds.</div>'
    else:
        status_block = f'<div class="alert-warning"><b>⚠️ Overall status:</b> {alert_count} alert rows detected. Focus on the peak windows and repeated breaches.</div>'

    return {
        "status_block": status_block,
        "total_calls": total_calls,
        "avg_aht": avg_aht,
        "peak_calls_interval": peak_calls_interval,
        "peak_calls_value": peak_calls_value,
        "worst_var_pct": worst_var_pct,
        "worst_var_interval": worst_var_interval,
        "peak_occ": peak_occ,
        "peak_occ_interval": peak_occ_interval,
        "pct_under": pct_under,
        "pct_occ_risk": pct_occ_risk,
        "top_risk": top_risk,
        "corr_calls_aht": corr_calls_aht,
        "corr_calls_occ": corr_calls_occ,
        "recs": recs,
    }


# ----------------------------
# Main App
# ----------------------------
def main():
    # Header
    st.markdown('<div class="main-header">📊 Intraday Health Check</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Interactive intraday dashboard for volume, AHT, staffing, variance, occupancy, and alerts.</div>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Setup")
        use_demo = st.checkbox("Use Demo Data", value=False)
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        st.markdown("<hr/>", unsafe_allow_html=True)

        st.markdown("### 🔔 Alert Thresholds")
        understaffed_threshold = st.number_input(
            "Understaffed Threshold (%)",
            value=-10.0,
            min_value=-50.0,
            max_value=0.0,
            step=1.0,
            help="Alert when variance % is below this threshold (more negative = worse)",
        )
        volume_spike_multiplier = st.number_input(
            "Volume Spike Multiplier",
            value=1.20,
            min_value=1.0,
            max_value=3.0,
            step=0.05,
            help="Alert when calls exceed rolling mean × this multiplier",
        )
        aht_spike_multiplier = st.number_input(
            "AHT Spike Multiplier",
            value=1.15,
            min_value=1.0,
            max_value=3.0,
            step=0.05,
            help="Alert when AHT exceeds rolling mean × this multiplier",
        )
        occupancy_threshold = st.number_input(
            "Occupancy Risk Threshold (%)",
            value=90.0,
            min_value=50.0,
            max_value=200.0,
            step=5.0,
            help="Alert when occupancy exceeds this percentage",
        )

        st.markdown("<hr/>", unsafe_allow_html=True)

        st.markdown("### 📐 Parameters")
        shrinkage_pct = st.number_input(
            "Shrinkage Percentage",
            value=0.30,
            min_value=0.0,
            max_value=0.99,
            step=0.01,
            format="%.2f",
            help="Shrinkage used to convert base requirement to net requirement (breaks, training, etc.)",
        )
        default_interval_min = st.selectbox(
            "Default Interval Length (minutes)",
            options=[15, 30, 60],
            index=1,
            help="Used if interval_min is missing in your CSV",
        )

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("### 📥 Sample Data")
        sample_df = generate_sample_data(days=1, interval_minutes=default_interval_min)
        st.download_button(
            "📥 Download Sample CSV",
            data=sample_df.to_csv(index=False),
            file_name=f"sample_intraday_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.caption("Tip: Column names can be mapped if they differ. Date = YYYY-MM-DD; Interval = HH:MM or HHMM.")

    # Load data
    if not use_demo and uploaded_file is None:
        st.info("👈 Upload a CSV file or enable **Use Demo Data** to get started.")
        st.markdown(
            """
<div class="card">
  <div class="card-title">📋 Required CSV Columns</div>
  <span class="badge">date</span> <span class="badge">interval</span> <span class="badge">calls</span>
  <span class="badge">aht</span> <span class="badge">interval_min</span> <span class="badge">net_staff</span>
  <p class="small-note" style="margin-top:.6rem;">
    • date: YYYY-MM-DD (e.g., 2026-01-02)<br/>
    • interval: HH:MM (08:00) or HHMM (0800)<br/>
    • aht is in seconds
  </p>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    if use_demo:
        df = generate_sample_data(days=1, interval_minutes=default_interval_min)
        st.markdown('<div class="alert-success"><b>Demo mode:</b> Sample data loaded. Upload your own CSV anytime.</div>', unsafe_allow_html=True)
    else:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded! ({len(df)} rows, {len(df.columns)} columns)")
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            return

    # Column mapping (only if not demo)
    required_cols = ["date", "interval", "calls", "aht", "interval_min", "net_staff"]
    if not use_demo:
        available_cols = list(df.columns)
        df_cols_lower = [c.lower() for c in df.columns]

        if all(col in df_cols_lower for col in required_cols):
            # Case-insensitive rename to standard
            mapping = {}
            for req in required_cols:
                idx = df_cols_lower.index(req)
                mapping[df.columns[idx]] = req
            df = df.rename(columns=mapping)
            df = df[required_cols]
        else:
            st.markdown("### 🔍 Column Mapping")
            st.info("Some required columns were not found. Map your CSV columns below.")
            col_mapping = {}
            cols = st.columns(3)
            for i, req_col in enumerate(required_cols):
                with cols[i % 3]:
                    default_idx = 0
                    matching = [c for c in available_cols if c.lower() == req_col.lower()]
                    if matching:
                        default_idx = available_cols.index(matching[0]) + 1
                    col_mapping[req_col] = st.selectbox(
                        f"Map to '{req_col}'",
                        options=[""] + available_cols,
                        index=default_idx,
                        key=f"map_{req_col}",
                    )
            if not all(col_mapping.values()):
                st.error("❌ Please map all required columns.")
                return

            df2 = df.copy()
            for req_col, mapped_col in col_mapping.items():
                df2[req_col] = df[mapped_col]
            df = df2[required_cols]

    # interval_min default if missing/blank
    if "interval_min" not in df.columns or df["interval_min"].isna().all():
        df["interval_min"] = default_interval_min
        st.info(f"ℹ️ interval_min missing → using default: {default_interval_min} minutes")

    # Parse datetime
    df = parse_datetime(df, "date", "interval")
    if df is None:
        return

    # Validate
    df, issues = validate_data(df)
    for issue in issues:
        st.warning(issue)
    if len(df) == 0:
        st.error("❌ No valid rows remaining after validation.")
        return

    # Calculate metrics + alerts
    df = calculate_metrics(df, shrinkage_pct)
    if df is None:
        return

    df, alerts_df = detect_alerts(
        df,
        understaffed_threshold=understaffed_threshold,
        volume_spike_multiplier=volume_spike_multiplier,
        aht_spike_multiplier=aht_spike_multiplier,
        occupancy_threshold=occupancy_threshold,
    )

    # Filters (top controls)
    left, mid, right = st.columns([1.2, 1.2, 2.0])
    with left:
        date_options = sorted(df["date"].astype(str).unique().tolist())
        selected_date = st.selectbox("📅 Date", date_options, index=0)
    with mid:
        view_mode = st.selectbox("🎛️ View", ["All intervals", "Only alerts", "Understaffed only", "Occupancy risk only"])
    with right:
        st.caption("Use filters to focus on specific risk windows. Charts update automatically.")

    dff = df[df["date"].astype(str) == str(selected_date)].copy()
    if view_mode == "Only alerts" and len(alerts_df) > 0:
        alert_times = alerts_df[alerts_df["date"].astype(str) == str(selected_date)]["interval"].astype(str).unique()
        dff = dff[dff["interval"].astype(str).isin(alert_times)]
    elif view_mode == "Understaffed only":
        dff = dff[dff["Understaffed"] == True]
    elif view_mode == "Occupancy risk only":
        dff = dff[dff["Occupancy_Risk"] == True]

    if len(dff) == 0:
        st.warning("No rows match your current filter selection.")
        dff = df[df["date"].astype(str) == str(selected_date)].copy()

    # KPI row
    total_calls = int(dff["calls"].sum())
    avg_aht = float(dff["aht"].mean())
    avg_net_staff = float(dff["net_staff"].mean())
    avg_required = float(dff["Required_FTE_net"].mean())
    worst_var_idx = int(dff["Variance_Pct"].idxmin())
    worst_var = float(dff.loc[worst_var_idx, "Variance_Pct"])
    worst_time = str(dff.loc[worst_var_idx, "interval"])
    peak_occ = float(dff["Occupancy"].max())

    st.markdown(
        f"""
<div class="kpi-grid">
  <div class="kpi"><div class="label">📞 Total Calls</div><div class="value">{total_calls:,}</div><div class="hint">for selected date/filter</div></div>
  <div class="kpi"><div class="label">⏱️ Avg AHT</div><div class="value">{avg_aht:.0f} sec</div><div class="hint">{format_seconds(avg_aht)}</div></div>
  <div class="kpi"><div class="label">👥 Avg Net Staff</div><div class="value">{avg_net_staff:.1f}</div><div class="hint">average staffed agents</div></div>
  <div class="kpi"><div class="label">🎯 Avg Required Staff</div><div class="value">{avg_required:.1f}</div><div class="hint">includes shrinkage</div></div>
  <div class="kpi"><div class="label">📉 Worst Variance</div><div class="value">{worst_var:.1f}%</div><div class="hint">@ {worst_time}</div></div>
  <div class="kpi"><div class="label">🔥 Peak Occupancy</div><div class="value">{peak_occ:.1f}%</div><div class="hint">highest interval load</div></div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("<br/>", unsafe_allow_html=True)

    # Tabs
    tab_dash, tab_alerts, tab_analysis, tab_data, tab_guide = st.tabs(
        ["📊 Dashboard", "🚨 Alerts", "🧠 Data Analysis", "📋 Data", "📖 Guide"]
    )

    # ----------------------------
    # Dashboard tab
    # ----------------------------
    with tab_dash:
        c1, c2 = st.columns(2)

        # Calls
        with c1:
            fig_calls = go.Figure()
            fig_calls.add_trace(
                go.Scatter(
                    x=dff["interval"],
                    y=dff["calls"],
                    mode="lines+markers",
                    name="Calls",
                    line=dict(width=3),
                    marker=dict(size=7),
                )
            )
            fig_calls.update_layout(
                title="📞 Calls by Interval",
                xaxis_title="Interval",
                yaxis_title="Calls",
                hovermode="x unified",
                height=360,
                template="plotly_white",
                showlegend=False,
            )
            st.plotly_chart(fig_calls, use_container_width=True)

        # AHT
        with c2:
            fig_aht = go.Figure()
            fig_aht.add_trace(
                go.Scatter(
                    x=dff["interval"],
                    y=dff["aht"],
                    mode="lines+markers",
                    name="AHT (sec)",
                    line=dict(width=3),
                    marker=dict(size=7),
                )
            )
            fig_aht.update_layout(
                title="⏱️ AHT by Interval",
                xaxis_title="Interval",
                yaxis_title="AHT (sec)",
                hovermode="x unified",
                height=360,
                template="plotly_white",
                showlegend=False,
            )
            st.plotly_chart(fig_aht, use_container_width=True)

        c3, c4 = st.columns(2)

        # Staff vs Required
        with c3:
            fig_staff = go.Figure()
            fig_staff.add_trace(
                go.Scatter(
                    x=dff["interval"],
                    y=dff["Required_FTE_net"],
                    mode="lines+markers",
                    name="Required Staff",
                    line=dict(width=3),
                    marker=dict(size=7),
                )
            )
            fig_staff.add_trace(
                go.Scatter(
                    x=dff["interval"],
                    y=dff["net_staff"],
                    mode="lines+markers",
                    name="Net Staff",
                    line=dict(width=3),
                    marker=dict(size=7),
                )
            )
            fig_staff.update_layout(
                title="👥 Required vs Net Staff",
                xaxis_title="Interval",
                yaxis_title="Staff",
                hovermode="x unified",
                height=360,
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_staff, use_container_width=True)

        # Variance bar
        with c4:
            colors = np.where(dff["Variance_HC"] < 0, "crimson", "seagreen")
            fig_var = go.Figure()
            fig_var.add_trace(
                go.Bar(
                    x=dff["interval"],
                    y=dff["Variance_HC"],
                    name="Variance (HC)",
                    marker_color=colors,
                )
            )
            fig_var.add_hline(y=0, line_dash="dash", line_width=1, line_color="gray")
            fig_var.update_layout(
                title="📉 Variance (Headcount) by Interval",
                xaxis_title="Interval",
                yaxis_title="Variance HC",
                hovermode="x unified",
                height=360,
                template="plotly_white",
                showlegend=False,
            )
            st.plotly_chart(fig_var, use_container_width=True)

        # Occupancy line
        fig_occ = go.Figure()
        fig_occ.add_trace(
            go.Scatter(
                x=dff["interval"],
                y=dff["Occupancy"],
                mode="lines+markers",
                name="Occupancy (%)",
                line=dict(width=3),
                marker=dict(size=7),
            )
        )
        fig_occ.add_hline(
            y=float(occupancy_threshold),
            line_dash="dot",
            line_width=2,
            line_color="orange",
            annotation_text=f"Risk threshold: {occupancy_threshold:.0f}%",
            annotation_position="top left",
        )
        fig_occ.update_layout(
            title="🔥 Occupancy by Interval",
            xaxis_title="Interval",
            yaxis_title="Occupancy (%)",
            hovermode="x unified",
            height=360,
            template="plotly_white",
            showlegend=False,
        )
        st.plotly_chart(fig_occ, use_container_width=True)

    # ----------------------------
    # Alerts tab
    # ----------------------------
    with tab_alerts:
        st.markdown("### 🚨 Alerts")
        if alerts_df is None or len(alerts_df) == 0:
            st.markdown(
                '<div class="alert-success"><b>✅ No alerts detected!</b> Everything is within your current thresholds.</div>',
                unsafe_allow_html=True,
            )
        else:
            alerts_today = alerts_df[alerts_df["date"].astype(str) == str(selected_date)].copy()

            a1, a2, a3, a4 = st.columns([1.1, 1.1, 1.2, 2.0])
            with a1:
                alert_types = ["All"] + sorted(alerts_df["alert_type"].unique().tolist())
                pick_type = st.selectbox("Alert Type", alert_types)
            with a2:
                min_occ = st.slider("Min Occupancy (%)", 0, 200, 0, 5)
            with a3:
                max_rows = st.selectbox("Rows to show", [25, 50, 100, 200], index=1)
            with a4:
                st.caption("Filter alerts to focus on what you need to fix first.")

            fa = alerts_today.copy()
            if pick_type != "All":
                fa = fa[fa["alert_type"].str.contains(pick_type, case=False, na=False)]
            fa = fa[fa["occupancy"] >= float(min_occ)]
            fa = fa.sort_values(["interval", "alert_type"]).head(max_rows)

            st.dataframe(fa, use_container_width=True, hide_index=True)

            st.download_button(
                "📥 Download Filtered Alerts CSV",
                data=fa.to_csv(index=False),
                file_name=f"alerts_{selected_date}_{datetime.now().strftime('%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ----------------------------
    # Data Analysis tab (with basic interpretations)
    # ----------------------------
    with tab_analysis:
        st.markdown("### 🧠 Data Analysis (Simple Interpretation)")

        insights = build_basic_interpretation(
            df=df[df["date"].astype(str) == str(selected_date)].copy(),
            alerts_df=alerts_df[alerts_df["date"].astype(str) == str(selected_date)].copy()
            if alerts_df is not None and len(alerts_df) > 0
            else pd.DataFrame(),
            understaffed_threshold=understaffed_threshold,
            occupancy_threshold=occupancy_threshold,
        )

        st.markdown(insights["status_block"], unsafe_allow_html=True)

        st.markdown(
            f"""
<div class="card">
  <div class="card-title">What the numbers are saying</div>
  <ul>
    <li><b>Peak volume</b> happens at <b>{insights["peak_calls_interval"]}</b> with about <b>{insights["peak_calls_value"]}</b> calls.</li>
    <li><b>Worst staffing gap</b> is at <b>{insights["worst_var_interval"]}</b> with variance <b>{insights["worst_var_pct"]:.1f}%</b> (negative means understaffed).</li>
    <li><b>Peak occupancy</b> is <b>{insights["peak_occ"]:.1f}%</b> at <b>{insights["peak_occ_interval"]}</b>. Higher = agents are busier.</li>
    <li><b>{insights["pct_under"]:.0f}%</b> of intervals are below your understaffing threshold (<b>{understaffed_threshold:.0f}%</b>).</li>
    <li><b>{insights["pct_occ_risk"]:.0f}%</b> of intervals are above your occupancy risk threshold (<b>{occupancy_threshold:.0f}%</b>).</li>
  </ul>
  <p class="small-note">
    Correlations (rough signals): calls vs AHT = <b>{insights["corr_calls_aht"]:.2f}</b>, calls vs occupancy = <b>{insights["corr_calls_occ"]:.2f}</b>.
    Positive values mean they tend to rise together.
  </p>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("<br/>", unsafe_allow_html=True)

        # Top risk intervals table
        st.markdown("#### 🔥 Top 3 Risk Intervals (Most Understaffed)")
        top_risk = insights["top_risk"].copy()
        top_risk.columns = ["Interval", "Variance (%)", "Variance (HC)"]
        st.dataframe(top_risk.round(2), use_container_width=True, hide_index=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # Distributions
        d1, d2 = st.columns(2)
        with d1:
            fig_hist_var = px.histogram(
                df[df["date"].astype(str) == str(selected_date)].copy(),
                x="Variance_Pct",
                nbins=20,
                title="Distribution of Variance (%)",
            )
            fig_hist_var.update_layout(template="plotly_white", height=340)
            st.plotly_chart(fig_hist_var, use_container_width=True)

        with d2:
            fig_hist_occ = px.histogram(
                df[df["date"].astype(str) == str(selected_date)].copy(),
                x="Occupancy",
                nbins=20,
                title="Distribution of Occupancy (%)",
            )
            fig_hist_occ.update_layout(template="plotly_white", height=340)
            st.plotly_chart(fig_hist_occ, use_container_width=True)

        # Correlation heatmap (simple)
        st.markdown("#### 🧩 Relationship Check (Correlation Heatmap)")
        corr_df = df[df["date"].astype(str) == str(selected_date)][
            ["calls", "aht", "net_staff", "Required_FTE_net", "Variance_Pct", "Occupancy"]
        ].corr(numeric_only=True)
        fig_corr = px.imshow(corr_df, text_auto=True, aspect="auto", title="Correlation (Higher absolute = stronger relationship)")
        fig_corr.update_layout(height=420)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Recommendations
        st.markdown("#### ✅ Simple Recommendations")
        if insights["recs"]:
            st.markdown('<div class="alert-warning"><b>Focus areas:</b><br/>' + "<br/>".join([f"• {r}" for r in insights["recs"]]) + "</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-success"><b>Nice!</b> Nothing major stands out. Keep monitoring peaks and protect breaks/lunch windows.</div>', unsafe_allow_html=True)

    # ----------------------------
    # Data tab
    # ----------------------------
    with tab_data:
        st.markdown("### 📋 Detailed Results")

        results_display = df[df["date"].astype(str) == str(selected_date)][
            [
                "date",
                "interval",
                "calls",
                "aht",
                "net_staff",
                "Required_FTE_net",
                "Variance_HC",
                "Variance_Pct",
                "Occupancy",
                "Understaffed",
                "Volume_Spike",
                "AHT_Spike",
                "Occupancy_Risk",
                "Consecutive_Breach",
            ]
        ].copy()

        results_display.rename(
            columns={
                "date": "Date",
                "interval": "Interval",
                "calls": "Calls",
                "aht": "AHT (sec)",
                "net_staff": "Net Staff",
                "Required_FTE_net": "Required Staff",
                "Variance_HC": "Variance (HC)",
                "Variance_Pct": "Variance (%)",
                "Occupancy": "Occupancy (%)",
            },
            inplace=True,
        )

        st.dataframe(results_display.round(2), use_container_width=True, hide_index=True)

        st.download_button(
            "📥 Download Results CSV",
            data=df.to_csv(index=False),
            file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ----------------------------
    # Guide tab
    # ----------------------------
    with tab_guide:
        st.markdown("### 📖 How to Read This Dashboard")
        st.markdown(
            """
**Calls** = number of contacts arriving in each interval.  
**AHT (sec)** = average handle time per call (seconds).  
**Workload (minutes)** = how many minutes of agent work your calls need.  
**Required Staff (Net)** = how many agents you need after adding shrinkage (breaks, training, meetings).  
**Variance (%)** = (Net Staff - Required Staff) / Required Staff.  
- Negative = **understaffed** (higher queues risk)  
- Positive = **overstaffed** (opportunity to pull back / redeploy)

**Occupancy (%)** = how busy agents are.  
- 70–90% is often “healthy” depending on your operation  
- > threshold means **risk** (queues can grow fast)

#### Alert rules (what triggers them)
- **Understaffed**: Variance % < your threshold  
- **Volume Spike**: Calls > rolling average × multiplier  
- **AHT Spike**: AHT > rolling average × multiplier  
- **Occupancy Risk**: Occupancy > your threshold  
- **Consecutive Breach**: Understaffed for 3+ intervals in a row

Tip: Start with **Top 3 Risk Intervals** in the Data Analysis tab. Fixing those windows usually improves the whole day.
"""
        )

        with st.expander("📌 Formulas Used"):
            st.markdown(
                """
1) **Workload (minutes)** = `calls × (aht / 60)`  
2) **Required FTE** = `Workload_minutes / interval_min`  
3) **Required FTE (Net)** = `Required_FTE / (1 - shrinkage_pct)`  
4) **Variance (HC)** = `net_staff - Required_FTE_net`  
5) **Variance (%)** = `(Variance_HC / Required_FTE_net) × 100`  
6) **Occupancy (%)** = `Workload_minutes / (net_staff × interval_min) × 100` (capped 0–200%)
"""
            )


if __name__ == "__main__":
    main()
