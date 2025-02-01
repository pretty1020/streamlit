import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import base64
import io

# Define app title
st.title("Forecasting Tool with Channel Analysis")

# Sidebar: Definitions and User Guide
st.sidebar.header("Definitions")
st.sidebar.markdown("""
- **Forecasting:** Predicting future values based on historical data.
- **MAE (Mean Absolute Error):** Measures the average magnitude of errors.
- **RMSE (Root Mean Squared Error):** Standard way to measure forecasting error.
- **ARIMA:** A statistical model for time-series analysis.
- **Holt-Winters:** Uses exponential smoothing to forecast trend/seasonality.
- **Prophet:** A forecasting model designed for time series data.
""")

st.sidebar.header("User Guide")
st.sidebar.markdown("""
1. **Upload Data:** Upload CSV or Excel with `Date`, `Volume`, `LOB`, and `Channel` columns.
2. **Default Data:** Used if no file is uploaded.
3. **Forecasting Levels:** Daily, Weekly, Monthly.
4. **Filter by Channel:** Forecast can be filtered by `Channel`.
5. **Download Forecasted Data:** CSV download available.
""")

# Load default dataset (stored within GitHub repository)
@st.cache_data
def load_default_data():
    url = "https://raw.githubusercontent.com/your-repo/default_dummy_forecasting_data.csv"
    try:
        data = pd.read_csv(url)
        data.columns = map(str.lower, data.columns)  # Ensure lowercase column names
        data["date"] = pd.to_datetime(data["date"])  # Convert "date" column to datetime
        return data[["date", "volume", "lob", "channel"]]
    except Exception as e:
        st.error(f"Error loading default data: {e}")
        return pd.DataFrame()

# File uploader for user data
uploaded_file = st.file_uploader("Upload your data (CSV or Excel)", type=["csv", "xlsx"])
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            data = pd.read_excel(uploaded_file)
        data.columns = map(str.lower, data.columns)  # Ensure lowercase column names
        data["date"] = pd.to_datetime(data["date"])  # Convert "date" column to datetime
    except Exception as e:
        st.error(f"Error processing uploaded data: {e}")
        data = pd.DataFrame()
else:
    data = load_default_data()

# Validate required columns
required_columns = ["date", "volume", "lob", "channel"]
if not data.empty and not all(col in data.columns for col in required_columns):
    st.error(f"Uploaded data must contain the following columns: {', '.join(required_columns)}")
    data = pd.DataFrame()

# Show dataset preview
if not data.empty:
    st.write("### Dataset Preview")
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
            "RMSE": np.sqrt(mean_squared_error(data["volume"], prophet_model.predict(prophet_data)["yhat"][: len(data)])),
        }
        forecast_data["Prophet"] = prophet_forecast["yhat"].values
        forecast_data["Date"] = prophet_forecast["ds"].values
    except Exception as e:
        st.warning(f"Prophet failed: {e}")

    return results, forecast_data

if not data.empty:
    frequency_map = {"Daily": "D", "Weekly": "W-MON", "Monthly": "M"}
    level = st.selectbox("Choose Forecasting Level:", list(frequency_map.keys()))
    results, forecast_data = forecast_with_methods(data[["date", "volume"]].copy(), frequency_map[level])

    unique_channels = data["channel"].unique()
    selected_channel = st.selectbox("Filter Forecast by Channel:", unique_channels)

    filtered_data = data[data["channel"] == selected_channel]

    st.write(f"### Forecasting Results for {selected_channel} ({level})")
    for method, res in results.items():
        st.write(f"#### {method}")
        st.line_chart(res["forecast"])
        st.write(f"MAE: {res['MAE']:.2f}, RMSE: {res['RMSE']:.2f}")

    # Display forecast data table and download button
    st.write("### Forecasted Data Table")
    st.dataframe(forecast_data)

    # CSV download button
    csv = forecast_data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="forecasted_data.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)
