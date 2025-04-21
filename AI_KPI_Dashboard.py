import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import openai
from datetime import datetime, timedelta

st.set_page_config(page_title="📊 KPI & WFM Dashboard", layout="wide")

# Tabs
tabs = st.tabs(["Dashboard", "User Guide and Definition of Terms"])

with tabs[0]:
    st.title("📊 KPI & WFM Dashboard")

    # Sidebar Inputs
    st.sidebar.header("User Settings")
    api_key = st.sidebar.text_input("🔑 Enter your OpenAI API Key", type="password")
    uploaded_file = st.sidebar.file_uploader("📁 Upload your KPI CSV", type=["csv"])

    selected_team = st.sidebar.multiselect("Select Team", [], key="team_selector")
    data_frequency = st.sidebar.selectbox("Select Data Frequency", ["Daily", "Weekly", "Monthly"])

    # Load Data
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        else:
            st.error("Your uploaded data must include a 'Date' column.")
            st.stop()
    else:
        # Dummy Data
        np.random.seed(42)
        dates = pd.date_range(datetime.now() - timedelta(days=30), periods=30, freq='D')
        teams = ["Team A", "Team B", "Team C", "Team D"]
        data = {
            "Date": np.tile(dates, len(teams)),
            "Team": np.repeat(teams, len(dates)),
            "SLA": np.random.uniform(70, 100, len(dates) * len(teams)),
            "AHT": np.random.uniform(200, 500, len(dates) * len(teams)),
            "Occupancy": np.random.uniform(60, 100, len(dates) * len(teams)),
            "Adherence": np.random.uniform(80, 100, len(dates) * len(teams)),
            "Volume Delivery": np.random.randint(100, 500, len(dates) * len(teams)),
            "Attendance": np.random.uniform(85, 100, len(dates) * len(teams)),
        }
        df = pd.DataFrame(data)
        df["Date"] = pd.to_datetime(df["Date"])

    if 'Team' in df.columns:
        team_options = df['Team'].unique().tolist()
        selected_team = st.sidebar.multiselect("Select Team", options=team_options, default=team_options)
        df_filtered = df[df["Team"].isin(selected_team)]
    else:
        st.warning("No 'Team' column found. All data will be shown.")
        df_filtered = df

    if data_frequency == "Weekly":
        df_filtered = df_filtered.groupby("Team").resample("W", on="Date").mean(numeric_only=True).reset_index()
    elif data_frequency == "Monthly":
        df_filtered = df_filtered.groupby("Team").resample("M", on="Date").mean(numeric_only=True).reset_index()

    st.write("## 🤔 AI assistant for KPI Insights")
    if api_key:
        openai.api_key = api_key
        query = st.text_input("Ask the AI about KPI trends (e.g., 'How was SLA last week?')", key="chatbot")
        if query:
            prompt = f"Analyze this KPI data and answer the query: {df_filtered.to_dict()} \n\nQuery: {query}"
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            st.write("🤖 AI Response:", response['choices'][0]['message']['content'])
    else:
        st.info("Please enter your OpenAI API key to enable the AI assistant.")

    # KPI Overview Section
    st.subheader("📈 KPI Overview")
    kpi_cols = st.columns(6)
    kpis = ["SLA", "AHT", "Occupancy", "Adherence", "Volume Delivery", "Attendance"]
    for i, kpi in enumerate(kpis):
        with kpi_cols[i]:
            st.metric(label=kpi, value=f"{df_filtered[kpi].mean():.2f}")

    # Control Chart and Anomaly Detection
    st.subheader("📉 Control Chart and Anomaly Detection")
    control_kpi = st.selectbox("Select KPI for Control Chart", kpis)
    df_control = df_filtered.copy()
    mean_val = df_control[control_kpi].mean()
    std_val = df_control[control_kpi].std()
    lower_limit = mean_val - 3 * std_val
    upper_limit = mean_val + 3 * std_val
    df_control['Anomaly'] = (df_control[control_kpi] < lower_limit) | (df_control[control_kpi] > upper_limit)
    anomaly_df = df_control[df_control['Anomaly']][["Date", "Team", control_kpi]]

    fig_control = go.Figure()
    for team in df_control['Team'].unique():
        team_data = df_control[df_control['Team'] == team]
        fig_control.add_trace(go.Scatter(x=team_data['Date'], y=team_data[control_kpi], mode='lines+markers', name=team))
    fig_control.add_hline(y=mean_val, line_dash="dash", line_color="blue", annotation_text="Mean")
    fig_control.add_hline(y=lower_limit, line_dash="dot", line_color="red", annotation_text="Lower Limit")
    fig_control.add_hline(y=upper_limit, line_dash="dot", line_color="red", annotation_text="Upper Limit")

    if not anomaly_df.empty:
        fig_control.add_trace(go.Scatter(x=anomaly_df['Date'], y=anomaly_df[control_kpi], mode='markers', marker=dict(color='red', size=10, symbol='x'), name='Anomaly'))

    st.plotly_chart(fig_control, use_container_width=True)

    # Anomaly Data Table
    st.write("### 🚩 Anomaly Data Table (Showing up to 5 records)")
    if anomaly_df.empty:
        st.info("No anomalies detected for the selected KPI.")
    else:
        st.dataframe(anomaly_df.sort_values("Date").head(5).reset_index(drop=True))

    # KPI Comparison Chart
    st.subheader("📊 KPI Comparison Chart")
    primary_kpi = st.selectbox("Select Primary KPI", kpis, index=0)
    secondary_kpi = st.selectbox("Select Secondary KPI", kpis, index=1)
    fig_comp = px.line(df_filtered, x="Date", y=[primary_kpi, secondary_kpi], color="Team", markers=True, title=f"{primary_kpi} and {secondary_kpi} by Team")
    st.plotly_chart(fig_comp, use_container_width=True)

    # Drill-down Analysis
    st.subheader("📌 Drill-down Analysis")
    drill_metric = st.selectbox("Select KPI to Analyze", kpis)
    fig_drill = px.bar(df_filtered, x="Date", y=drill_metric, color="Team", barmode="group", title=f"{drill_metric} by Team")
    st.plotly_chart(fig_drill, use_container_width=True)

    # Download Button
    st.write("### 📥 Download the Filtered Data")
    csv = df_filtered.to_csv(index=False)
    st.download_button("Download CSV", csv, "filtered_kpi_data.csv", "text/csv", key="download-csv")

    st.write("---")
    st.write("🚀 Built with ❤️ by Marian")

with tabs[1]:
    st.title("User Guide & Definition of Terms")
    st.markdown("""
    ## User Guide

    Welcome to the **KPI & WFM Dashboard**. Here's how to use it:

    1. **Upload Your Data (Optional):**
       - Use the sidebar to upload your CSV file.
       - Your file should include at least a `Date` column. For full functionality, include:
         - `Team`
         - `SLA`
         - `AHT`
         - `Occupancy`
         - `Adherence`
         - `Volume Delivery`
         - `Attendance`

    2. **Enter OpenAI API Key (Optional):**
       - To enable the AI assistant, paste your API key in the sidebar.
       - Don't have one? [Get your free OpenAI API key here](https://platform.openai.com/account/api-keys)
       - The AI can answer KPI trend questions like:
         - “How did SLA perform last week?”
         - “Which team has the highest AHT?”

    3. **Filters:**
       - **Select Team:** Choose the teams to analyze.
       - **Select Data Frequency:** Toggle between Daily, Weekly, or Monthly views.

    4. **Dashboard Sections:**
       - **KPI Overview:** Quick view of average KPI values.
       - **AI Chatbot:** Ask KPI-related questions to the AI assistant.
       - **Control Chart:** Detect anomalies in selected KPI trends.
       - **Comparison Chart:** Plot and compare two KPIs across time.
       - **Drill-Down Analysis:** See KPI breakdown by team and date.
       - **Data Download:** Export the filtered data as a CSV file.

    ---

    ## Definition of Terms

    - **SLA (Service Level Agreement):** % of tasks or calls completed within a target time.
    - **AHT (Average Handle Time):** The mean duration agents take to complete a task or call.
    - **Occupancy:** % of time agents spend actively working vs. idle.
    - **Adherence:** % of time agents stick to their scheduled working hours.
    - **Volume Delivery:** Total number of handled tasks, calls, or deliverables.
    - **Attendance:** % of days agents were present and available.

    For questions or feedback, reach out to your Workforce Management (WFM) team.
    """)
