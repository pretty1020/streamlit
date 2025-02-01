import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import base64

# Define app title
st.title("Forecasting Tool with Channel Analysis")

# Sidebar: Definitions and User Guide
st.sidebar.header("Definitions")
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

# Upload dataset
uploaded_file = st.file_uploader("Upload your data (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            try:
                import openpyxl  # Ensure openpyxl is installed

                data = pd.read_excel(uploaded_file, engine="openpyxl")
            except ImportError:
                st.error(
                    "Missing dependency: `openpyxl` is required for Excel files. Install it using `pip install openpyxl`.")
                data = pd.DataFrame()

        data.columns = map(str.lower, data.columns)  # Ensure lowercase column names
        data["date"] = pd.to_datetime(data["date"])  # Convert "date" column to datetime

    except Exception as e:
        st.error(f"Error processing uploaded data: {e}")
        data = pd.DataFrame()
else:
    st.warning("Please upload a dataset to proceed.")
    data = pd.DataFrame()

# Validate required columns
required_columns = ["date", "volume", "lob", "channel"]
if not data.empty and not all(col in data.columns for col in required_columns):
    st.error(f"Uploaded data must contain the following columns: {', '.join(required_columns)}")
    data = pd.DataFrame()

# Show dataset preview if valid
if not data.empty:
    st.write("**Dataset Preview:**")
    st.write(data.head())


# Forecasting function
def forecast_with_methods(data, frequency):
    results = {}
    forecast_data = pd.DataFrame()

    # Ensure time-series format
    data.set_index("date", inplace=True)
    data = data.asfreq(frequency).fillna(method="ffill")

    # ARIMA
    try:
        arima_model = ARIMA(data["volume"], order=(5, 1, 0)).fit()
        arima_forecast = arima_model.forecast(steps=30)
        arima_dates = pd.date_range(start=data.index[-1], periods=30, freq=frequency)
        results["ARIMA"] = {
            "forecast": arima_forecast,
            "MAE": mean_absolute_error(data["volume"], arima_model.fittedvalues),
            "RMSE": np.sqrt(mean_squared_error(data["volume"], arima_model.fittedvalues)),
        }
        forecast_data["ARIMA"] = arima_forecast
        forecast_data["Date"] = arima_dates
    except Exception as e:
        st.warning(f"ARIMA failed: {e}")

    # Holt-Winters
    try:
        hw_model = ExponentialSmoothing(data["volume"], seasonal="add", seasonal_periods=12).fit()
        hw_forecast = hw_model.forecast(30)
        hw_dates = pd.date_range(start=data.index[-1], periods=30, freq=frequency)
        results["Holt-Winters"] = {
            "forecast": hw_forecast,
            "MAE": mean_absolute_error(data["volume"], hw_model.fittedvalues),
            "RMSE": np.sqrt(mean_squared_error(data["volume"], hw_model.fittedvalues)),
        }
        forecast_data["Holt-Winters"] = hw_forecast
    except Exception as e:
        st.warning(f"Holt-Winters failed: {e}")

    # Prophet
    try:
        prophet_data = data.reset_index().rename(columns={"date": "ds", "volume": "y"})
        prophet_model = Prophet()
        prophet_model.fit(prophet_data)
        future = prophet_model.make_future_dataframe(periods=30)
        prophet_forecast = prophet_model.predict(future)[["ds", "yhat"]].iloc[-30:]
        results["Prophet"] = {
            "forecast": prophet_forecast["yhat"],
            "MAE": mean_absolute_error(data["volume"], prophet_model.predict(prophet_data)["yhat"][: len(data)]),
            "RMSE": np.sqrt(
                mean_squared_error(data["volume"], prophet_model.predict(prophet_data)["yhat"][: len(data)])),
        }
        forecast_data["Prophet"] = prophet_forecast["yhat"].values
        forecast_data["Date"] = prophet_forecast["ds"].values
    except Exception as e:
        st.warning(f"Prophet failed: {e}")

    return results, forecast_data


if not data.empty:
    # Display forecast levels
    frequency_map = {"Daily": "D", "Weekly": "W-MON", "Monthly": "M"}
    level = st.selectbox("Choose Forecasting Level:", list(frequency_map.keys()))
    results, forecast_data = forecast_with_methods(data[["date", "volume"]].copy(), frequency_map[level])

    # Filter forecasted values by Channel
    unique_channels = data["channel"].unique()
    selected_channel = st.selectbox("Filter Forecast by Channel:", unique_channels)

    # Filter data by the selected channel
    filtered_data = data[data["channel"] == selected_channel]

    # Display results
    st.write(f"**Forecasting Results for {selected_channel} ({level}):**")
    for method, res in results.items():
        st.write(f"### {method}")
        st.line_chart(res["forecast"])
        st.write(f"MAE: {res['MAE']:.2f}, RMSE: {res['RMSE']:.2f}")

    # Display forecast data table and download button
    st.write("**Forecasted Data Table:**")
    st.dataframe(forecast_data)

    # Add download button
    csv = forecast_data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Convert to base64
    href = f'<a href="data:file/csv;base64,{b64}" download="forecasted_data.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)
