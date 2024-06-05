import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import requests

# Function to load data for a given ticker
@st.cache_data
def load_data(ticker, period='1y', interval='1d'):
    try:
        data = yf.download(ticker, period=period, interval=interval)
        return data
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        return pd.DataFrame()

# Function to preprocess the data
@st.cache_data
def preprocess_data(data):
    try:
        data['Return'] = data['Close'].pct_change()
        data.dropna(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return pd.DataFrame()

# Function to train the prediction model
@st.cache_data
def train_model(data):
    try:
        X = data[['Open', 'High', 'Low', 'Volume']].values
        y = data['Close'].values
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None

# Function to predict prices
@st.cache_data
def predict_prices(_model, data):
    try:
        X = data[['Open', 'High', 'Low', 'Volume']].values
        predictions = _model.predict(X)
        return predictions
    except Exception as e:
        st.error(f"Error predicting prices: {e}")
        return []

# Function to predict next day price
@st.cache_data
def predict_next_day_price(_model, data):
    try:
        latest_data = data.iloc[-1]
        latest_features = latest_data[['Open', 'High', 'Low', 'Volume']].values.reshape(1, -1)
        next_day_prediction = _model.predict(latest_features)
        return next_day_prediction[0]
    except Exception as e:
        st.error(f"Error predicting next day price: {e}")
        return None

# Function to fetch exchange rate from USD to pesos
def get_usd_to_peso_rate():
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
        response_json = response.json()
        if response.status_code == 200 and "rates" in response_json:
            exchange_rate = response_json["rates"].get("PHP")
            if exchange_rate is not None:
                return exchange_rate
            else:
                st.error("Exchange rate not found in API response.")
                return None
        else:
            st.error("Failed to fetch exchange rate. Please try again later.")
            return None
    except Exception as e:
        st.error(f"Error fetching exchange rate: {e}. Please try again later.")
        return None

# Top 20 cryptocurrencies
top_20 = [
    "BTC-USD", "ETH-USD", "BNB-USD", "USDT-USD", "SOL-USD", "ADA-USD", "XRP-USD", "DOT-USD", "DOGE-USD", "USDC-USD",
    "UNI-USD", "AVAX-USD", "LUNA-USD", "SHIB-USD", "LTC-USD", "LINK-USD", "BUSD-USD", "BCH-USD", "ALGO-USD", "MATIC-USD"
]

# Streamlit interface
st.title("Cryptocurrency Price Prediction")

# Create tab panel
tabs = ["Cryptocurrency Prices", "Real-Time Data", "Trend in Last 12 Hours", "Currency Conversion to PHP"]
selected_tab = st.radio("Select a tab:", tabs)

if selected_tab == "Cryptocurrency Prices":
    # Allow the user to select a cryptocurrency from the top 20
    selected_crypto = st.selectbox("Select a cryptocurrency:", top_20)

    # Load data for the selected cryptocurrency
    data = load_data(selected_crypto)

    if not data.empty:
        # Preprocess the data
        processed_data = preprocess_data(data)

        if not processed_data.empty:
            # Train the prediction model
            model = train_model(processed_data)

            if model is not None:
                # Predict prices
                predictions = predict_prices(model, processed_data)

                if len(predictions) > 0:
                    # Add predicted values to the DataFrame
                    processed_data['Predicted'] = predictions

                    # Predict next day price
                    next_day_prediction = predict_next_day_price(model, processed_data)

                    if next_day_prediction is not None:
                        # Add next day prediction to the DataFrame
                        next_day_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
                        next_day_df = pd.DataFrame({'Date': [next_day_date], 'Predicted': [next_day_prediction]})
                        next_day_df['Date'] = pd.to_datetime(next_day_df['Date'])
                        next_day_df.set_index('Date', inplace=True)
                        processed_data = pd.concat([processed_data, next_day_df])

                        # Fetch exchange rate from USD to pesos
                        usd_to_peso_rate = get_usd_to_peso_rate()

                        if usd_to_peso_rate is not None:
                            # Convert prices from USD to pesos
                            processed_data['Close_Pesos'] = processed_data['Close'] * usd_to_peso_rate
                            processed_data['Predicted_Pesos'] = processed_data['Predicted'] * usd_to_peso_rate

                            # Display daily data with predicted values
                            st.subheader("Daily Data with Predicted Values")
                            st.write(processed_data)

                            # Display predicted vs actual prices
                            st.subheader("Predicted vs Actual Prices")
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['Close_Pesos'], mode='lines', name='Actual Price (PHP)'))
                            fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['Predicted_Pesos'], mode='lines', name='Predicted Price (PHP)', line=dict(color='red')))
                            st.plotly_chart(fig)

elif selected_tab == "Currency Conversion to PHP":
    # Fetch exchange rate from USD to pesos
    usd_to_peso_rate = get_usd_to_peso_rate()

    if usd_to_peso_rate is not None:
        # Display exchange rate
        st.subheader("Currency Conversion to PHP")
        st.write(f"1 USD = {usd_to_peso_rate} PHP")
        st.write("You can use this exchange rate to convert cryptocurrency prices from USD to PHP.")

elif selected_tab == "Real-Time Data":
    # Allow the user to select a cryptocurrency from the top 20
    selected_crypto = st.selectbox("Select a cryptocurrency:", top_20)

    # Load data for the selected cryptocurrency
    data = load_data(selected_crypto, period='1d', interval='1m')

    if not data.empty:
        # Fetch exchange rate from USD to pesos
        usd_to_peso_rate = get_usd_to_peso_rate()

        if usd_to_peso_rate is not None:
            # Display real-time data for the day
            st.subheader("Real-Time Data for the Day")
            if not data.empty:
                # Convert prices from USD to pesos
                data['Close_Pesos'] = data['Close'] * usd_to_peso_rate
                st.write(data[['Close', 'Close_Pesos']])

                # Train the prediction model
                model = train_model(data)

                if model is not None:
                    # Predict next day price
                    next_day_prediction = predict_next_day_price(model, data)

                    if next_day_prediction is not None:
                        # Convert predicted closing value from USD to PHP
                        predicted_closing_value_php = next_day_prediction * usd_to_peso_rate

                        # Display predicted closing value for the day in PHP
                        st.subheader("Predicted Closing Value for the Day")
                        st.write(f"The predicted closing value for {selected_crypto} today is: {predicted_closing_value_php:.2f} PHP")

                        # Display trend by line graph
                        st.subheader("Trend by Line Graph")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=data.index, y=data['Close_Pesos'], mode='lines', name='Actual Price (PHP)'))
                        st.plotly_chart(fig)

elif selected_tab in ["Upward Trend in Last 6 Hours", "Upward Trend in Last 3 Hours"]:
    # Define a function to calculate the price change in the last specified number of hours
    def calculate_price_change_last_hours(data, num_hours):
        now = datetime.now(data.index.tzinfo)  # Get current time with the same timezone as data
        hours_ago = now - timedelta(hours=num_hours)
        recent_data = data.loc[data.index >= hours_ago]

        if not recent_data.empty:
            price_change = recent_data.iloc[-1]['Close'] - recent_data.iloc[0]['Close']
            percentage_change = ((recent_data.iloc[-1]['Close'] - recent_data.iloc[0]['Close']) / recent_data.iloc[0]['Close']) * 100
            return price_change, percentage_change
        else:
            return 0, 0

    # Fetch exchange rate from USD to pesos
    usd_to_peso_rate = get_usd_to_peso_rate()

    # Create an empty DataFrame to store cryptocurrency and its corresponding price change
    trend_data = pd.DataFrame(columns=['Cryptocurrency', 'Price Change (USD)', 'Price Change (PHP)', 'Percentage Change (%)'])

    # Iterate over top 20 cryptocurrencies
    for crypto in top_20:
        data = load_data(crypto, period='1d', interval='1m')
        if not data.empty:
            hours = 6 if selected_tab == "Upward Trend in Last 6 Hours" else 3
            price_change, percentage_change = calculate_price_change_last_hours(data, hours)
            price_change_php = price_change * usd_to_peso_rate if usd_to_peso_rate is not None else None
            trend_data = pd.concat([trend_data, pd.DataFrame({'Cryptocurrency': [crypto], 'Price Change (USD)': [price_change], 'Price Change (PHP)': [price_change_php], 'Percentage Change (%)': [percentage_change]})], ignore_index=True)

    # Filter out the top 10 cryptocurrencies with the highest positive price change
    top_10_upward_trend = trend_data.nlargest(10, 'Price Change (USD)')

    # Display the results
    st.subheader(f"Cryptocurrencies with Upward Trend in Last {'6' if selected_tab == 'Upward Trend in Last 6 Hours' else '3'} Hours")
    st.write(top_10_upward_trend)

    # Plot trend by line graph
    st.subheader(f"Trend in Last {'6' if selected_tab == 'Upward Trend in Last 6 Hours' else '3'} Hours")
    fig = go.Figure()
    for crypto in top_10_upward_trend['Cryptocurrency']:
        data = load_data(crypto, period='1d', interval='1m')
        if not data.empty:
            hours_ago = 6 if selected_tab == "Upward Trend in Last 6 Hours" else 3
            recent_data = data.loc[data.index >= (datetime.now(data.index.tzinfo) - timedelta(hours=hours_ago))]
            if not recent_data.empty:
                fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['Close'], mode='lines', name=crypto))
    st.plotly_chart(fig)

elif selected_tab == "Trend in Last 12 Hours":
    # Allow the user to select a cryptocurrency from the top 20
    selected_crypto = st.selectbox("Select a cryptocurrency:", top_20)

    # Load data for the selected cryptocurrency
    data = load_data(selected_crypto, period='1d', interval='5m')

    if not data.empty:
        # Fetch exchange rate from USD to pesos
        usd_to_peso_rate = get_usd_to_peso_rate()

        if usd_to_peso_rate is not None:
            # Display trend data for the last 12 hours
            st.subheader("Trend Data for the Last 12 Hours")
            now = datetime.now(data.index.tzinfo)  # Get current time with the same timezone as data
            twelve_hours_ago = now - timedelta(hours=12)
            recent_data = data.loc[data.index >= twelve_hours_ago]

            if not recent_data.empty:
                # Convert prices from USD to pesos
                recent_data['Close_Pesos'] = recent_data['Close'] * usd_to_peso_rate
                st.write(recent_data[['Close', 'Close_Pesos']])

                # Display trend by line graph
                st.subheader("Trend by Line Graph for the Last 12 Hours")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['Close_Pesos'], mode='lines', name='Actual Price (PHP)'))
                st.plotly_chart(fig)
