import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import base64

st.set_page_config(page_title="Forecasting & Workforce Calculator", layout="wide")

tab1, tab2 = st.tabs(["📊 Forecasting Tool", "🧮 Workforce Calculator"])

with tab1:
    st.title("Forecasting Tool with Channel Analysis")
    # Sidebar: Definitions and User Guide
    st.sidebar.header("Definitions (Forecasting tool)")
    st.sidebar.markdown("""
        - **Forecasting:** The process of predicting future values based on historical data.
        - **MAE (Mean Absolute Error):** Measures the average magnitude of errors in a set of forecasts.
        - **RMSE (Root Mean Squared Error):** A standard way to measure the error of a model in predicting quantitative data.
        - **ARIMA (AutoRegressive Integrated Moving Average):** A statistical model used for time-series analysis and forecasting.
        - **Holt-Winters:** A method that uses exponential smoothing to forecast data with a trend and seasonality.
        - **Prophet:** A forecasting procedure designed for time series data.
        """)

    st.sidebar.header("User Guide")
    st.sidebar.markdown("""
        1. **Upload Data:** Upload your own dataset. Ensure it has `Date`, `Volume`, `LOB`, and `Channel` columns.
        2. **Forecasting Levels:** Daily (Mon-Sun), Weekly (Week 1, 2, ...), Monthly (Jan-Dec).
        3. **Forecast Data:** Filter forecasted results by Channel and download the forecast as a CSV file.
        """)


    uploaded_file = st.file_uploader("Upload your data (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                data = pd.read_csv(uploaded_file, parse_dates=["Date"], dayfirst=True)
            elif uploaded_file.name.endswith(".xlsx"):
                import openpyxl
                data = pd.read_excel(uploaded_file, engine="openpyxl")

            data.columns = map(str.lower, data.columns)
            data["date"] = pd.to_datetime(data["date"], errors="coerce")

        except Exception as e:
            st.error(f"Error processing uploaded data: {e}")
            data = pd.DataFrame()
    else:
        data = pd.DataFrame()

    if not data.empty:
        st.write("**Dataset Preview:**")
        st.write(data.head())

        unique_channels = data["channel"].unique()
        selected_channel = st.selectbox("Filter Forecast by Channel:", unique_channels)
        filtered_data = data[data["channel"] == selected_channel].copy()

        level = st.selectbox("Choose Forecasting Level:", ["Daily", "Weekly", "Monthly"])

        if level == "Weekly":
            start_of_week = st.radio("Select Start of the Week:", ["Sunday", "Monday"], index=1)
            frequency = "W-SUN" if start_of_week == "Sunday" else "W-MON"
        else:
            frequency = {"Daily": "D", "Monthly": "M"}.get(level, "D")

        def aggregate_data(df, freq):
            return df.resample(freq, on="date").sum().reset_index()

        def forecast_with_methods(df, freq):
            results = {}
            error_measures = {}
            df = aggregate_data(df, freq)
            df.set_index("date", inplace=True)
            df = df.asfreq(freq).fillna(method="ffill")

            forecast_data = pd.DataFrame()
            forecast_dates = pd.date_range(start=df.index[-1], periods=60, freq=freq)
            forecast_data["Date"] = forecast_dates

            try:
                prophet_data = df.reset_index().rename(columns={"date": "ds", "volume": "y"})
                prophet_model = Prophet()
                prophet_model.fit(prophet_data)
                future = prophet_model.make_future_dataframe(periods=30, freq=freq)
                prophet_forecast = prophet_model.predict(future)[["ds", "yhat"]].iloc[-30:]
                results["Prophet"] = prophet_forecast["yhat"].values
                forecast_data["Prophet"] = prophet_forecast["yhat"].values
                error_measures["Prophet"] = {
                    "MAE": mean_absolute_error(df["volume"], prophet_model.predict(prophet_data)["yhat"][ : len(df)]),
                    "RMSE": np.sqrt(mean_squared_error(df["volume"], prophet_model.predict(prophet_data)["yhat"][ : len(df)]))
                }
            except Exception as e:
                st.warning(f"Prophet failed: {e}")

            try:
                hw_model = ExponentialSmoothing(df["volume"], seasonal="add", seasonal_periods=12).fit()
                hw_forecast = hw_model.forecast(30)
                results["Holt-Winters"] = hw_forecast.values
                forecast_data["Holt-Winters"] = hw_forecast.values
                error_measures["Holt-Winters"] = {
                    "MAE": mean_absolute_error(df["volume"], hw_model.fittedvalues),
                    "RMSE": np.sqrt(mean_squared_error(df["volume"], hw_model.fittedvalues))
                }
            except Exception as e:
                st.warning(f"Holt-Winters failed: {e}")

            try:
                arima_model = ARIMA(df["volume"], order=(5, 1, 0)).fit()
                arima_forecast = arima_model.forecast(steps=30)
                results["ARIMA"] = arima_forecast.values
                forecast_data["ARIMA"] = arima_forecast.values
                error_measures["ARIMA"] = {
                    "MAE": mean_absolute_error(df["volume"], arima_model.fittedvalues),
                    "RMSE": np.sqrt(mean_squared_error(df["volume"], arima_model.fittedvalues))
                }
            except Exception as e:
                st.warning(f"ARIMA failed: {e}")

            return results, forecast_data, error_measures

        results, forecast_data, error_measures = forecast_with_methods(filtered_data, frequency)

        st.write(f"**Forecasting Results for {selected_channel} ({level}):**")
        for method in ["Prophet", "Holt-Winters", "ARIMA"]:
            if method in results:
                st.write(f"### {method}")
                st.line_chart(results[method])

        st.write("**Forecasted Data Table:**")
        st.dataframe(forecast_data)

        st.write("**📊 Error Measure Comparison:**")
        error_df = pd.DataFrame(error_measures).T
        st.dataframe(error_df)

        csv = forecast_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="forecasted_data.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)




### 🧮 TAB 2: WORKFORCE MANAGEMENT CALCULATOR ###
with tab2:
    st.title("🧮 Workforce Management Calculator")

    # Sidebar: Definitions for Workforce Planning
    st.sidebar.header("Definitions (Workforce Calculator)")
    st.sidebar.markdown("""
    - **FTE (Full-Time Equivalent):** The number of full-time employees required.
    - **Call Volume:** Expected number of incoming calls per interval.
    - **AHT (Average Handle Time):** The average time to handle a call, including talk and wrap-up time.
    - **Occupancy Rate:** Percentage of time agents spend handling calls versus waiting.
    - **Shrinkage:** Time lost due to breaks, meetings, training, or absences.
    - **SCF (Schedule Challenge Factor):** The limitations and constraints that affect how you can schedule employees in a workforce.
    """)
    st.sidebar.markdown("[For any concerns or issues,feel free to reach out to Marian via Linkedin](https://www.linkedin.com/in/marian1020/)")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📞 Call Volume Forecast")
        volume = st.number_input("Expected Call Volume per Interval", min_value=0, value=1000)
        aht = st.number_input("Average Handle Time (AHT) in Minutes", min_value=0.1, value=13.5)
        occupancy = st.slider("Occupancy Rate (%)", min_value=50, max_value=100, value=70)
        shrinkage = st.slider("Total Shrinkage (%)", min_value=0, max_value=100, value=20)

    with col2:
        st.subheader("👥 Workforce Calculation")
        weekly_hours = st.number_input("Work Hours per Week", min_value=1, max_value=168, value=40)
        scf = st.number_input("Schedule Challenge Factor (SCF)", min_value=1.0, value=1.01)

    # FTE Calculation
    if st.button("Calculate Required FTE"):
        fte_required = ((volume * (aht / 60)) / (occupancy / 100)) / (100 - shrinkage) * scf * weekly_hours/8/2
        st.success(f"Required Full-Time Equivalent (FTE): **{round(fte_required, 2)}**")

    st.subheader("📊 Handling Capacity and Productive Hours")
    available_fte = st.number_input("Available FTE", min_value=1, value=20)
    monthly_hours = st.number_input("Monthly Work Hours", min_value=1, max_value=744, value=176)

    if st.button("Calculate Capacity"):
        handling_capacity = (available_fte * monthly_hours) * (occupancy / 100) / (aht / 60)
        st.success(f"Estimated Handling Capacity: **{round(handling_capacity)} calls per month**")

        productive_hours = available_fte * monthly_hours
        st.success(f"Estimated Productive Hours: **{round(productive_hours)} hours per month**")
