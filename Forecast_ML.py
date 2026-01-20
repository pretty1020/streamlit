import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Import forecasting models with path handling for deployment
import sys
import os

# Add current directory to Python path to ensure models can be found
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import forecasting models
from models.arima_model import ARIMAForecaster
from models.expsmooth_model import ExpSmoothForecaster
from models.prophet_model import ProphetForecaster
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Simple forecasting models
class MovingAverageForecaster:
    def __init__(self, window=7):
        self.window = window
        self.data = None
        
    def fit(self, data):
        self.data = data.copy()
        return self
    
    def forecast(self, periods):
        # Simple moving average forecast
        last_values = self.data['y'].tail(self.window).values
        forecast_values = np.full(periods, last_values.mean())
        
        last_date = self.data['ds'].max()
        if len(self.data) > 0:
            if (self.data['ds'].iloc[1] - self.data['ds'].iloc[0]).days == 1:
                freq = 'D'
                future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
            else:
                freq = 'MS'
                future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
        else:
            future_dates = pd.date_range(start=last_date, periods=periods + 1, freq='D')[1:] if periods > 0 else pd.date_range(start=last_date, periods=1, freq='D')
        
        return pd.DataFrame({'ds': future_dates, 'yhat': forecast_values})

class WeightedMovingAverageForecaster:
    def __init__(self, window=7):
        self.window = window
        self.data = None
        
    def fit(self, data):
        self.data = data.copy()
        return self
    
    def forecast(self, periods):
        # Weighted moving average (more weight to recent values)
        last_values = self.data['y'].tail(self.window).values
        weights = np.linspace(0.5, 1.0, len(last_values))
        weights = weights / weights.sum()
        forecast_value = np.average(last_values, weights=weights)
        forecast_values = np.full(periods, forecast_value)
        
        last_date = self.data['ds'].max()
        if len(self.data) > 0:
            if (self.data['ds'].iloc[1] - self.data['ds'].iloc[0]).days == 1:
                freq = 'D'
                future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
            else:
                freq = 'MS'
                future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
        else:
            future_dates = pd.date_range(start=last_date, periods=periods + 1, freq='D')[1:] if periods > 0 else pd.date_range(start=last_date, periods=1, freq='D')
        
        return pd.DataFrame({'ds': future_dates, 'yhat': forecast_values})

class HoltWintersForecaster:
    def __init__(self):
        self.model = None
        self.data = None
        
    def fit(self, data, seasonal_periods=7):
        self.data = data.copy()
        self.data = self.data.set_index('ds')
        
        try:
            # Try with seasonality
            self.model = ExponentialSmoothing(
                self.data['y'],
                trend='add',
                seasonal='add',
                seasonal_periods=seasonal_periods
            ).fit()
        except:
            try:
                # Try without seasonality
                self.model = ExponentialSmoothing(
                    self.data['y'],
                    trend='add'
                ).fit()
            except:
                # Simple exponential smoothing
                self.model = ExponentialSmoothing(
                    self.data['y']
                ).fit()
        
        return self
    
    def forecast(self, periods):
        forecast_values = self.model.forecast(periods)
        
        last_date = self.data.index.max()
        if len(self.data) > 0:
            if (self.data.index[1] - self.data.index[0]).days == 1:
                freq = 'D'
            else:
                freq = 'MS'
            future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
        else:
            future_dates = pd.date_range(start=last_date, periods=periods + 1, freq='D')[1:] if periods > 0 else pd.date_range(start=last_date, periods=1, freq='D')
        
        return pd.DataFrame({'ds': future_dates, 'yhat': forecast_values})
    
    def evaluate(self, test_data):
        """Evaluate model performance"""
        predictions = self.model.forecast(len(test_data))
        mae = mean_absolute_error(test_data['y'], predictions)
        rmse = np.sqrt(mean_squared_error(test_data['y'], predictions))
        accuracy = max(0, min(100, (1 - mae / test_data['y'].mean()) * 100))
        return mae, rmse, accuracy

MODEL_OPTIONS = {
    "Moving Average": MovingAverageForecaster,
    "Weighted Moving Average": WeightedMovingAverageForecaster,
    "Holt-Winters": HoltWintersForecaster,
    "ARIMA": ARIMAForecaster,
    "Exponential Smoothing": ExpSmoothForecaster,
    "Prophet": ProphetForecaster
}

MODEL_DESCRIPTIONS = {
    "Moving Average": "Simple moving average - Average of last N periods",
    "Weighted Moving Average": "Weighted moving average - Recent values weighted more heavily",
    "Holt-Winters": "Exponential smoothing with trend and seasonality - Great for seasonal patterns",
    "ARIMA": "AutoRegressive Integrated Moving Average - Classic time series forecasting",
    "Exponential Smoothing": "Smooth historical data with exponential weighting - Great for trend patterns",
    "Prophet": "Facebook's Prophet - Robust to missing data and handles holidays automatically"
}

DEFAULT_EXOG = ['is_holiday', 'is_system_down', 'is_maintenance']

# ============================================================================
# DUMMY DATA GENERATOR
# ============================================================================
def generate_dummy_data(days=365):
    """Generate dummy time series data for demonstration."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Create a time series with trend, seasonality, and noise
    trend = np.linspace(100, 200, days)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(days) / 365.25)
    weekly = 10 * np.sin(2 * np.pi * np.arange(days) / 7)
    noise = np.random.normal(0, 5, days)
    
    values = trend + seasonal + weekly + noise
    
    # Create exogenous variables
    is_holiday = np.random.choice([0, 1], size=days, p=[0.95, 0.05])
    is_system_down = np.random.choice([0, 1], size=days, p=[0.98, 0.02])
    is_maintenance = np.random.choice([0, 1], size=days, p=[0.97, 0.03])
    
    df = pd.DataFrame({
        'ds': dates,
        'y': np.maximum(values, 0),  # Ensure non-negative
        'is_holiday': is_holiday,
        'is_system_down': is_system_down,
        'is_maintenance': is_maintenance
    })
    
    return df

# ============================================================================
# PAGE CONFIG & CUSTOM CSS
# ============================================================================
st.set_page_config(
    page_title="Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with Gradient Theme & Improved Typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Global Typography & Base Styles */
    * {
        font-family: 'Poppins', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    /* Techy Dark Background */
    .main .block-container {
        background: #0d1117;
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid #21262d;
    }
    
    body {
        background: #010409;
    }
    
    .stApp {
        background: #010409;
    }
    
    /* Techy Neon Header */
    .main-header {
        background: #0a0e27;
        padding: 3.5rem 2.5rem;
        border-radius: 24px;
        color: #ffffff;
        margin-bottom: 2.5rem;
        border: 2px solid #00ffff;
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.3),
                    0 0 60px rgba(0, 255, 255, 0.2),
                    inset 0 0 30px rgba(0, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, 
            transparent 30%, 
            rgba(0, 255, 255, 0.1) 50%, 
            transparent 70%);
        animation: rotate 10s linear infinite;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.6; }
        50% { transform: scale(1.1); opacity: 0.9; }
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 3.5rem;
        font-weight: 900;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.8),
                     0 0 20px rgba(0, 255, 255, 0.6),
                     0 0 30px rgba(0, 255, 255, 0.4),
                     0 0 40px rgba(0, 255, 255, 0.2);
        position: relative;
        z-index: 1;
        letter-spacing: -1px;
        line-height: 1.2;
        color: #00ffff;
        font-family: 'Courier New', monospace;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from {
            text-shadow: 0 0 10px rgba(0, 255, 255, 0.8),
                         0 0 20px rgba(0, 255, 255, 0.6),
                         0 0 30px rgba(0, 255, 255, 0.4);
        }
        to {
            text-shadow: 0 0 20px rgba(0, 255, 255, 1),
                         0 0 30px rgba(0, 255, 255, 0.8),
                         0 0 40px rgba(0, 255, 255, 0.6),
                         0 0 50px rgba(0, 255, 255, 0.4);
        }
    }
    
    .main-header p {
        margin: 1rem 0 0 0;
        color: #a0a0a0;
        font-size: 1.3rem;
        font-weight: 500;
        position: relative;
        z-index: 1;
        line-height: 1.6;
        text-shadow: 0 0 5px rgba(0, 255, 255, 0.5);
        font-family: 'Courier New', monospace;
    }
    
    /* Techy Colorful Metric Cards */
    .metric-card {
        background: #0a0e27;
        padding: 2rem;
        border-radius: 16px;
        border: 2px solid;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), 
                    0 0 20px rgba(0, 255, 255, 0.1) inset,
                    0 0 40px rgba(0, 255, 255, 0.05);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, 
            #00ffff 0%, 
            #ff00ff 25%, 
            #ffff00 50%, 
            #00ff00 75%, 
            #00ffff 100%);
        background-size: 200% 100%;
        animation: shimmer 3s linear infinite;
    }
    
    @keyframes shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
    .metric-card:nth-child(1) {
        border-color: #00ffff;
        box-shadow: 0 8px 32px rgba(0, 255, 255, 0.3), 
                    0 0 20px rgba(0, 255, 255, 0.2) inset;
    }
    
    .metric-card:nth-child(2) {
        border-color: #ff00ff;
        box-shadow: 0 8px 32px rgba(255, 0, 255, 0.3), 
                    0 0 20px rgba(255, 0, 255, 0.2) inset;
    }
    
    .metric-card:nth-child(3) {
        border-color: #ffff00;
        box-shadow: 0 8px 32px rgba(255, 255, 0, 0.3), 
                    0 0 20px rgba(255, 255, 0, 0.2) inset;
    }
    
    .metric-card:nth-child(4) {
        border-color: #00ff00;
        box-shadow: 0 8px 32px rgba(0, 255, 0, 0.3), 
                    0 0 20px rgba(0, 255, 0, 0.2) inset;
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 15px 50px rgba(0, 255, 255, 0.4), 
                    0 0 30px rgba(0, 255, 255, 0.3) inset;
    }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 900;
        color: #00ffff;
        margin: 0;
        line-height: 1.2;
        letter-spacing: -0.5px;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.8),
                     0 0 20px rgba(0, 255, 255, 0.5),
                     0 0 30px rgba(0, 255, 255, 0.3);
        font-family: 'Courier New', monospace;
    }
    
    .metric-card:nth-child(2) .metric-value {
        color: #ff00ff;
        text-shadow: 0 0 10px rgba(255, 0, 255, 0.8),
                     0 0 20px rgba(255, 0, 255, 0.5),
                     0 0 30px rgba(255, 0, 255, 0.3);
    }
    
    .metric-card:nth-child(3) .metric-value {
        color: #ffff00;
        text-shadow: 0 0 10px rgba(255, 255, 0, 0.8),
                     0 0 20px rgba(255, 255, 0, 0.5),
                     0 0 30px rgba(255, 255, 0, 0.3);
    }
    
    .metric-card:nth-child(4) .metric-value {
        color: #00ff00;
        text-shadow: 0 0 10px rgba(0, 255, 0, 0.8),
                     0 0 20px rgba(0, 255, 0, 0.5),
                     0 0 30px rgba(0, 255, 0, 0.3);
    }
    
    .metric-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: #a0a0a0;
        margin: 0.8rem 0 0 0;
        text-transform: uppercase;
        letter-spacing: 2px;
        line-height: 1.4;
        font-family: 'Courier New', monospace;
    }
    
    /* Enhanced Status Badges with Better Readability */
    .status-badge {
        padding: 0.7rem 1.6rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
        font-size: 0.95rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        text-shadow: 0 1px 3px rgba(0,0,0,0.2);
        letter-spacing: 0.3px;
        line-height: 1.4;
    }
    
    .status-success {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: #ffffff;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #ffc107 0%, #ff9800 100%);
        color: #ffffff;
    }
    
    .status-info {
        background: linear-gradient(135deg, #2196F3 0%, #21CBF3 100%);
        color: #ffffff;
    }
    
    /* Enhanced Buttons with Gradient */
    .stButton>button {
        border-radius: 12px;
        font-weight: 600;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.12);
        font-size: 1rem;
        letter-spacing: 0.3px;
        line-height: 1.5;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
    }
    
    /* Techy Primary Button */
    .stButton>button[kind="primary"] {
        background: #0a0e27;
        color: #00ffff;
        border: 2px solid #00ffff;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.5),
                    inset 0 0 20px rgba(0, 255, 255, 0.1);
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.8);
        font-family: 'Courier New', monospace;
        font-weight: 700;
    }
    
    .stButton>button[kind="primary"]:hover {
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.8),
                    inset 0 0 30px rgba(0, 255, 255, 0.2);
        transform: translateY(-2px);
    }
    
    /* Techy Info Boxes */
    .info-box {
        background: #0a0e27;
        padding: 1.8rem;
        border-radius: 16px;
        border: 2px solid #00ffff;
        margin: 1rem 0;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.3),
                    inset 0 0 20px rgba(0, 255, 255, 0.1);
        color: #a0a0a0;
        line-height: 1.7;
        font-size: 1rem;
        font-family: 'Courier New', monospace;
    }
    
    .info-box strong {
        color: #00ffff;
        font-weight: 700;
        text-shadow: 0 0 5px rgba(0, 255, 255, 0.5);
    }
    
    /* Techy Sidebar */
    [data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 2px solid #00ffff;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.2);
    }
    
    [data-testid="stSidebar"] * {
        color: #a0a0a0;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #00ffff;
        font-weight: 700;
        font-size: 1.2rem;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
        font-family: 'Courier New', monospace;
    }
    
    /* Techy Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #0a0e27;
        padding: 12px;
        border-radius: 16px;
        border: 1px solid #00ffff;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 14px 28px;
        font-weight: 600;
        transition: all 0.3s ease;
        color: #a0a0a0;
        font-size: 1rem;
        letter-spacing: 0.2px;
        font-family: 'Courier New', monospace;
        border: 1px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: #0a0e27;
        color: #00ffff;
        border: 1px solid #00ffff;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.5),
                    inset 0 0 15px rgba(0, 255, 255, 0.1);
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.8);
    }
    
    /* Enhanced Progress Bar with Gradient */
    .progress-container {
        background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
        border-radius: 12px;
        height: 18px;
        overflow: hidden;
        margin: 1rem 0;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 12px;
        transition: width 0.6s ease;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 200% 100%;
        animation: gradientShift 3s ease infinite;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.5);
    }
    
    /* Enhanced Model Card with Gradient */
    .model-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.8rem;
        border-radius: 16px;
        border: 2px solid #e9ecef;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .model-card:hover {
        border-color: #667eea;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.25);
        transform: translateY(-4px);
        background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%);
    }
    
    .model-card.selected {
        border-color: #667eea;
        background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    /* Techy Text Styling */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Courier New', monospace;
        font-weight: 700;
        line-height: 1.3;
        color: #00ffff;
        letter-spacing: 1px;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
    }
    
    h2 {
        color: #ff00ff;
        text-shadow: 0 0 10px rgba(255, 0, 255, 0.5);
    }
    
    h3 {
        color: #ffff00;
        text-shadow: 0 0 10px rgba(255, 255, 0, 0.5);
    }
    
    p, span, div, label {
        line-height: 1.6;
        color: #a0a0a0;
    }
    
    /* Streamlit Text Elements */
    .stMarkdown {
        color: #a0a0a0;
        line-height: 1.7;
    }
    
    .stMarkdown h1 {
        color: #00ffff;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.8);
    }
    
    .stMarkdown h2 {
        color: #ff00ff;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(255, 0, 255, 0.5);
    }
    
    .stMarkdown h3 {
        color: #ffff00;
        font-weight: 600;
        text-shadow: 0 0 10px rgba(255, 255, 0, 0.5);
    }
    
    /* Techy Dataframe Styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 2px solid #00ffff;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
        background: #0a0e27;
    }
    
    .stDataFrame table {
        background: #0a0e27;
        color: #a0a0a0;
    }
    
    .stDataFrame th {
        background: #0d1117;
        color: #00ffff;
        border: 1px solid #00ffff;
        font-family: 'Courier New', monospace;
        text-shadow: 0 0 5px rgba(0, 255, 255, 0.5);
    }
    
    .stDataFrame td {
        background: #0a0e27;
        color: #a0a0a0;
        border: 1px solid #1a1a2e;
        font-family: 'Courier New', monospace;
    }
    
    /* Input Fields */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        border-radius: 8px;
        border: 2px solid #e9ecef;
        transition: all 0.3s ease;
        font-family: 'Poppins', 'Inter', sans-serif;
    }
    
    .stTextInput>div>div>input:focus,
    .stNumberInput>div>div>input:focus,
    .stSelectbox>div>div>select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Radio Buttons */
    .stRadio>div {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e9ecef;
    }
    
    /* File Uploader */
    .stFileUploader {
        border-radius: 12px;
        border: 2px dashed #667eea;
        background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
    }
    
    /* Techy Success/Error/Info Messages */
    .stSuccess {
        background: #0a0e27;
        border: 2px solid #00ff00;
        border-radius: 8px;
        color: #00ff00;
        font-weight: 500;
        box-shadow: 0 0 15px rgba(0, 255, 0, 0.3);
        font-family: 'Courier New', monospace;
    }
    
    .stError {
        background: #0a0e27;
        border: 2px solid #ff0000;
        border-radius: 8px;
        color: #ff0000;
        font-weight: 500;
        box-shadow: 0 0 15px rgba(255, 0, 0, 0.3);
        font-family: 'Courier New', monospace;
    }
    
    .stInfo {
        background: #0a0e27;
        border: 2px solid #00ffff;
        border-radius: 8px;
        color: #00ffff;
        font-weight: 500;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.3);
        font-family: 'Courier New', monospace;
    }
    
    .stWarning {
        background: #0a0e27;
        border: 2px solid #ffff00;
        border-radius: 8px;
        color: #ffff00;
        font-weight: 500;
        box-shadow: 0 0 15px rgba(255, 255, 0, 0.3);
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def validate_data(df):
    """Validate the input data for forecasting."""
    issues = []
    
    if len(df) < 30:
        issues.append(("warning", "Less than 30 observations. This may lead to poor model performance."))
    
    if df['y'].isnull().any():
        issues.append(("info", "Missing values found in target variable. These will be interpolated."))
        df['y'] = df['y'].interpolate(method='linear')
    
    # Check for stationarity
    try:
        result = adfuller(df['y'].dropna())
        if result[1] > 0.05:
            issues.append(("warning", "Data may not be stationary. Consider differencing or transformation."))
    except:
        pass
    
    return df, issues

def preprocess_data(df, freq):
    """Preprocess the data for forecasting."""
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds')
    
    # Handle missing dates
    if freq == "Daily":
        date_range = pd.date_range(start=df['ds'].min(), end=df['ds'].max(), freq='D')
    else:
        date_range = pd.date_range(start=df['ds'].min(), end=df['ds'].max(), freq='MS')
    
    df = df.set_index('ds').reindex(date_range).reset_index()
    # Rename 'index' back to 'ds' if it was reset
    if 'index' in df.columns and 'ds' not in df.columns:
        df = df.rename(columns={'index': 'ds'})
    df['y'] = df['y'].interpolate(method='linear')
    
    return df

def create_forecast_chart(historical_df, forecast_df, model_name):
    """Create an interactive forecast visualization."""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_df['ds'],
        y=historical_df['y'],
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='#00ffff', width=3),
        marker=dict(size=6, color='#00ffff')
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#ff00ff', width=3, dash='dash'),
        marker=dict(size=8, color='#ff00ff', symbol='diamond')
    ))
    
    # Add confidence intervals if available
    if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_upper'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_lower'],
            mode='lines',
            name='Confidence Interval',
            fill='tonexty',
            fillcolor='rgba(240, 147, 251, 0.2)',
            line=dict(width=0)
        ))
    
    # Vertical line separating historical and forecast
    last_historical_date = historical_df['ds'].max()
    max_y = max(historical_df['y'].max(), forecast_df['yhat'].max())
    min_y = min(historical_df['y'].min(), forecast_df['yhat'].min())
    
    # Convert timestamp to proper format to avoid arithmetic issues
    if isinstance(last_historical_date, pd.Timestamp):
        # Convert to numpy datetime64 or string for plotly compatibility
        last_historical_date = pd.Timestamp(last_historical_date)
    
    # Use a shape instead of add_vline to avoid timestamp arithmetic issues
    fig.add_shape(
        type="line",
        x0=last_historical_date,
        x1=last_historical_date,
        y0=min_y * 0.9,
        y1=max_y * 1.1,
        line=dict(color="gray", width=2, dash="dot"),
    )
    
    # Add annotation for the line
    fig.add_annotation(
        x=last_historical_date,
        y=max_y * 1.05,
        text="Forecast Start",
        showarrow=False,
        font=dict(color="gray", size=12),
        bgcolor="rgba(10, 14, 39, 0.8)",
        bordercolor="gray",
        borderwidth=1
    )
    
    fig.update_layout(
        title=f'{model_name} Forecast Visualization',
        xaxis_title='Date',
        yaxis_title='Value',
        hovermode='x unified',
        height=500,
        template='plotly_dark',
        paper_bgcolor='#0a0e27',
        plot_bgcolor='#0a0e27',
        font=dict(family="Courier New, monospace", size=13, color="#a0a0a0"),
        title_font=dict(family="Courier New, monospace", size=18, color="#00ffff"),
        xaxis=dict(gridcolor='#1a1a2e', linecolor='#00ffff'),
        yaxis=dict(gridcolor='#1a1a2e', linecolor='#00ffff'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(family="Courier New, monospace", size=12),
            bgcolor='rgba(10, 14, 39, 0.8)',
            bordercolor='#00ffff',
            borderwidth=1
        ),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_metrics_cards(mae, rmse, accuracy, model_name):
    """Create beautiful metric cards."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{mae:.2f}</p>
            <p class="metric-label">Mean Absolute Error</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{rmse:.2f}</p>
            <p class="metric-label">Root Mean Squared Error</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{accuracy:.1f}%</p>
            <p class="metric-label">Model Accuracy</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# MAIN APP
# ============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>📈 Forecasting Dashboard</h1>
    <p>FREE Multi-Model Time Series Forecasting — Built by WFM Commons</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    
    uploaded = st.file_uploader(
        "📁 Upload CSV Data",
        type=["csv"],
        help="Upload a CSV with columns: ds (date), y (value), and optional exogenous variables"
    )
    
    freq = st.radio(
        "📅 Forecast Frequency",
        ["Daily", "Monthly"],
        help="Select whether to forecast daily or monthly"
    )
    
    st.markdown("---")
    st.markdown("### 🤖 Model Selection")
    
    # Enhanced model selection with descriptions
    model_name = st.selectbox(
        "Choose Forecasting Model",
        list(MODEL_OPTIONS.keys()),
        help="Select the best model for your data"
    )
    
    # Show model description
    st.markdown(f"""
    <div class="info-box">
        <strong>ℹ️ {model_name}</strong><br>
        {MODEL_DESCRIPTIONS[model_name]}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    forecast_periods = st.number_input(
        "🔮 Forecast Horizon",
        min_value=1,
        max_value=365 if freq == "Daily" else 60,
        value=7 if freq == "Daily" else 12,
        help=f"Number of {freq.lower()} periods to forecast"
    )
    
    # Handle exog column selection
    if model_name in ["ARIMA", "Prophet"]:
        st.markdown("---")
        st.markdown("### 📊 Exogenous Variables")
        exog_input = st.text_input(
            "Exogenous columns (comma-separated)",
            value=",".join(DEFAULT_EXOG),
            help="e.g., is_holiday,is_system_down,is_maintenance"
        )
        exog_columns = [col.strip() for col in exog_input.split(",") if col.strip()]
    else:
        exog_columns = []
    
    st.markdown("---")
    run = st.button("🚀 Run Forecast", use_container_width=True, type="primary")
    
    # Store button state in session state and clear old results when new forecast is requested
    if run:
        st.session_state['run_forecast'] = True
        # Clear previous forecast results to ensure fresh forecast
        if 'forecast_results' in st.session_state:
            del st.session_state['forecast_results']
        # Store current model selection to validate results match
        st.session_state['current_model'] = model_name
        st.session_state['current_freq'] = freq
        st.session_state['current_periods'] = forecast_periods
    elif 'run_forecast' not in st.session_state:
        st.session_state['run_forecast'] = False

# Main Content Area
# Use dummy data if no file uploaded, otherwise use uploaded file
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.info("Using default dummy data instead.")
        df = generate_dummy_data()
else:
    # No file uploaded - use dummy data
    df = generate_dummy_data()
    st.info("💡 **Using default demo data.** Upload a CSV file to use your own data.")

if df is not None:
    try:
        
        # Data Preview Tab
        tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🔮 Forecast Results", "📈 Analytics"])
        
        with tab1:
            st.markdown("### 📋 Data Preview")
            
            # Validate data
            df, issues = validate_data(df)
            
            # Show validation issues
            for issue_type, message in issues:
                if issue_type == "warning":
                    st.warning(f"⚠️ {message}")
                elif issue_type == "info":
                    st.info(f"ℹ️ {message}")
            
            # Preprocess data
            df = preprocess_data(df, freq)
            
            # Ensure 'ds' column exists (fix if it was renamed to 'index')
            if 'index' in df.columns and 'ds' not in df.columns:
                df = df.rename(columns={'index': 'ds'})
            if 'ds' not in df.columns:
                st.error("Error: 'ds' column not found after preprocessing. Please check your data format.")
                st.stop()
            
            # Data statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{len(df):,}</p>
                    <p class="metric-label">Total Records</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{df['y'].min():.2f}</p>
                    <p class="metric-label">Min Value</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{df['y'].max():.2f}</p>
                    <p class="metric-label">Max Value</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{df['y'].mean():.2f}</p>
                    <p class="metric-label">Mean Value</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Historical data visualization
            st.markdown("### 📈 Historical Data Trend")
            # Ensure we have the correct columns
            if 'ds' not in df.columns:
                st.error("Error: 'ds' column not found. Available columns: " + ", ".join(df.columns))
            else:
                fig_hist = px.line(
                    df,
                    x='ds',
                    y='y',
                    title='Historical Data Visualization',
                    labels={'ds': 'Date', 'y': 'Value'}
                )
                fig_hist.update_traces(
                    line=dict(color='#00ffff', width=3),
                    marker=dict(size=6, color='#00ffff')
                )
                fig_hist.update_layout(
                    height=400,
                    template='plotly_dark',
                    paper_bgcolor='#0a0e27',
                    plot_bgcolor='#0a0e27',
                    font=dict(family="Courier New, monospace", size=13, color="#a0a0a0"),
                    hovermode='x unified',
                    title_font=dict(family="Courier New, monospace", size=16, color="#00ffff"),
                    xaxis=dict(gridcolor='#1a1a2e', linecolor='#00ffff'),
                    yaxis=dict(gridcolor='#1a1a2e', linecolor='#00ffff')
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # Data table
            st.markdown("### 📋 Data Table")
            st.dataframe(df, use_container_width=True, height=300)
        
        # Forecast Results Tab - Always show, but conditionally show results
        with tab2:
            # Use session state to check if forecast should run
            should_run = st.session_state.get('run_forecast', False) or run
            
            # Always validate that stored results match current selection
            # If model/freq/periods changed, clear old results and force new forecast
            if 'forecast_results' in st.session_state:
                stored_model = st.session_state.get('current_model', '')
                stored_freq = st.session_state.get('current_freq', '')
                stored_periods = st.session_state.get('current_periods', 0)
                
                # If selection changed, clear old results
                if (stored_model != model_name or stored_freq != freq or stored_periods != forecast_periods):
                    del st.session_state['forecast_results']
                    # Force new forecast if button was clicked or if we're in a new session
                    if run or should_run:
                        should_run = True
            
            # Only run forecast if explicitly requested - no fallback to old results
            if should_run:
                st.markdown("### 🔮 Forecasting in Progress...")
                
                # Progress indicator
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.markdown('<div class="status-badge status-info">🔄 Initializing model...</div>', unsafe_allow_html=True)
                    progress_bar.progress(10)
                    
                    model_class = MODEL_OPTIONS[model_name]
                    
                    # Handle simple models (Moving Average, Weighted MA, Holt-Winters)
                    if model_name in ["Moving Average", "Weighted Moving Average", "Holt-Winters"]:
                        status_text.markdown('<div class="status-badge status-info">🔄 Training simple model...</div>', unsafe_allow_html=True)
                        progress_bar.progress(30)
                        
                        # Split data for evaluation
                        split_idx = int(len(df) * 0.8)
                        train_df = df.iloc[:split_idx].copy()
                        test_df = df.iloc[split_idx:].copy()
                        
                        # Initialize and fit model
                        if model_name == "Moving Average":
                            window = st.sidebar.slider("Moving Average Window", min_value=3, max_value=30, value=7, key="ma_window")
                            model = model_class(window=window)
                        elif model_name == "Weighted Moving Average":
                            window = st.sidebar.slider("Weighted MA Window", min_value=3, max_value=30, value=7, key="wma_window")
                            model = model_class(window=window)
                        else:  # Holt-Winters
                            seasonal_periods = st.sidebar.slider("Seasonal Periods", min_value=4, max_value=30, value=7, key="hw_seasonal")
                            model = model_class()
                            model.fit(train_df, seasonal_periods=seasonal_periods)
                            progress_bar.progress(60)
                            
                            # Evaluate
                            mae, rmse, accuracy = model.evaluate(test_df)
                            progress_bar.progress(80)
                            
                            # Forecast
                            forecast = model.forecast(forecast_periods)
                            
                        if model_name in ["Moving Average", "Weighted Moving Average"]:
                            model.fit(train_df)
                            progress_bar.progress(60)
                            
                            # Simple evaluation
                            forecast_test = model.forecast(len(test_df))
                            mae = mean_absolute_error(test_df['y'], forecast_test['yhat'])
                            rmse = np.sqrt(mean_squared_error(test_df['y'], forecast_test['yhat']))
                            accuracy = max(0, min(100, (1 - mae / test_df['y'].mean()) * 100))
                            progress_bar.progress(80)
                            
                            # Forecast future
                            forecast = model.forecast(forecast_periods)
                        
                        # Ensure ds is datetime
                        forecast['ds'] = pd.to_datetime(forecast['ds'])
                        
                    else:
                        # Complex models (ARIMA, Exponential Smoothing, Prophet)
                        # Initialize model with appropriate parameters
                        if model_name == "Prophet":
                            model = model_class(regressors=exog_columns)
                        else:
                            model = model_class()
                        
                        status_text.markdown('<div class="status-badge status-info">🔄 Training model with grid search...</div>', unsafe_allow_html=True)
                        progress_bar.progress(30)
                        
                        if freq == "Daily":
                            # Initialize data with appropriate parameters
                            if model_name == "ARIMA":
                                model.initialize_data(df, exog_columns)
                            else:
                                model.initialize_data(df)
                            
                            # Call the correct grid search method for each model
                            if model_name == "ARIMA":
                                model.grid_search_arima_daily()
                            elif model_name == "Exponential Smoothing":
                                model.grid_search_exp_smooth_daily()
                            else:  # Prophet
                                model.grid_search_daily()
                            
                            progress_bar.progress(60)
                            mae, rmse, accuracy = model.evaluate_daily()
                            progress_bar.progress(80)
                            
                            # Handle exog for ARIMA
                            if model_name == "ARIMA":
                                # Create future dates and exog
                                last_date = df['ds'].max()
                                future_dates = pd.date_range(start=last_date, periods=forecast_periods + 1, freq='D')[1:]
                                exog_future = pd.DataFrame({'ds': future_dates})
                                for col in exog_columns:
                                    exog_future[col] = 0
                                exog_future = exog_future.set_index('ds')[exog_columns] if exog_columns else None
                                forecast = model.forecast_future_daily(forecast_periods, exog_future=exog_future)
                            else:
                                forecast = model.forecast_future_daily(forecast_periods)
                        else:
                            # Initialize data with appropriate parameters
                            if model_name == "ARIMA":
                                model.initialize_data_monthly(df, exog_columns)
                            else:
                                model.initialize_data_monthly(df)
                            
                            # Call the correct grid search method for each model
                            if model_name == "ARIMA":
                                model.grid_search_arima_monthly()
                            elif model_name == "Exponential Smoothing":
                                model.grid_search_exp_smooth_monthly()
                            else:  # Prophet
                                model.grid_search_monthly()
                            
                            progress_bar.progress(60)
                            mae, rmse, accuracy = model.evaluate_monthly()
                            progress_bar.progress(80)
                            
                            # Handle exog for ARIMA
                            if model_name == "ARIMA":
                                # Create future dates and exog
                                last_date = df['ds'].max()
                                # Use MonthBegin offset properly
                                next_month = last_date + pd.offsets.MonthBegin(1)
                                future_dates = pd.date_range(start=next_month, periods=forecast_periods, freq='MS')
                                exog_future = pd.DataFrame({'ds': future_dates})
                                for col in exog_columns:
                                    exog_future[col] = 0
                                exog_future = exog_future.set_index('ds')[exog_columns] if exog_columns else None
                                forecast = model.forecast_future_monthly(forecast_periods, exog_future=exog_future)
                            else:
                                forecast = model.forecast_future_monthly(forecast_periods)
                        
                        # Normalize forecast column names
                        if 'date' in forecast.columns:
                            forecast = forecast.rename(columns={'date': 'ds', 'forecast': 'yhat'})
                        elif 'yhat' not in forecast.columns and len(forecast.columns) >= 2:
                            # Handle case where forecast might have different structure
                            forecast = forecast.rename(columns={forecast.columns[0]: 'ds', forecast.columns[1]: 'yhat'})
                        
                        # Ensure ds is datetime
                        forecast['ds'] = pd.to_datetime(forecast['ds'])
                    
                    progress_bar.progress(100)
                    status_text.markdown('<div class="status-badge status-success">✅ Forecast completed successfully!</div>', unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Model Performance Metrics
                    st.markdown("### 📊 Model Performance")
                    create_metrics_cards(mae, rmse, accuracy, model_name)
                    
                    st.markdown("---")
                    
                    # Forecast Visualization
                    st.markdown("### 📈 Forecast Visualization")
                    forecast_chart = create_forecast_chart(df, forecast, model_name)
                    st.plotly_chart(forecast_chart, use_container_width=True)
                    
                    # Forecast Table
                    st.markdown("### 📋 Forecast Results")
                    forecast_display = forecast.copy()
                    forecast_display.columns = ['Date', 'Forecasted Value']
                    forecast_display['Forecasted Value'] = forecast_display['Forecasted Value'].round(2)
                    st.dataframe(forecast_display, use_container_width=True, height=400)
                    
                    # Download button
                    st.markdown("---")
                    csv = forecast.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download Forecast as CSV",
                        data=csv,
                        file_name=f"{model_name.lower()}_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        type="primary"
                    )
                    
                    # Store results in session state for analytics tab
                    st.session_state['forecast_results'] = {
                        'model': model_name,
                        'forecast': forecast,
                        'historical': df,
                        'mae': mae,
                        'rmse': rmse,
                        'accuracy': accuracy,
                        'freq': freq,
                        'periods': forecast_periods
                    }
                    # Update current selection tracking
                    st.session_state['current_model'] = model_name
                    st.session_state['current_freq'] = freq
                    st.session_state['current_periods'] = forecast_periods
                    # Reset the run flag after successful forecast
                    st.session_state['run_forecast'] = False
                    
                except Exception as e:
                    status_text.markdown(f'<div class="status-badge status-warning">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
                    st.error(f"An error occurred: {str(e)}")
                    st.exception(e)
                    # Reset the run flag even on error
                    st.session_state['run_forecast'] = False
            else:
                st.info("👆 Click **🚀 Run Forecast** button in the sidebar to generate forecasts!")
                # No fallback results - only show results from current run
        
        # Analytics Tab
        with tab3:
            # Only show analytics if we have current forecast results that match selection
            if 'forecast_results' in st.session_state:
                # Validate results match current selection
                stored_model = st.session_state.get('current_model', '')
                stored_freq = st.session_state.get('current_freq', '')
                stored_periods = st.session_state.get('current_periods', 0)
                
                if (stored_model == model_name and stored_freq == freq and stored_periods == forecast_periods):
                    results = st.session_state['forecast_results']
                    
                    st.markdown("### 📊 Forecast Analytics")
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        forecast_mean = results['forecast']['yhat'].mean()
                        st.markdown(f"""
                        <div class="metric-card">
                            <p class="metric-value">{forecast_mean:.2f}</p>
                            <p class="metric-label">Average Forecast</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        forecast_trend = "📈 Increasing" if results['forecast']['yhat'].iloc[-1] > results['forecast']['yhat'].iloc[0] else "📉 Decreasing"
                        st.markdown(f"""
                        <div class="metric-card">
                            <p class="metric-value">{forecast_trend}</p>
                            <p class="metric-label">Forecast Trend</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        forecast_range = results['forecast']['yhat'].max() - results['forecast']['yhat'].min()
                        st.markdown(f"""
                        <div class="metric-card">
                            <p class="metric-value">{forecast_range:.2f}</p>
                            <p class="metric-label">Forecast Range</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Comparison chart
                    st.markdown("### 📊 Historical vs Forecast Comparison")
                    combined_df = pd.concat([
                        results['historical'][['ds', 'y']].rename(columns={'y': 'value'}),
                        results['forecast'][['ds', 'yhat']].rename(columns={'yhat': 'value'})
                    ])
                    combined_df['type'] = ['Historical'] * len(results['historical']) + ['Forecast'] * len(results['forecast'])
                    
                    fig_compare = px.line(
                        combined_df,
                        x='ds',
                        y='value',
                        color='type',
                        title='Historical Data vs Forecast',
                        labels={'ds': 'Date', 'value': 'Value', 'type': 'Type'}
                    )
                    fig_compare.update_traces(line=dict(width=3))
                    fig_compare.update_layout(
                        height=500,
                        template='plotly_dark',
                        paper_bgcolor='#0a0e27',
                        plot_bgcolor='#0a0e27',
                        font=dict(family="Courier New, monospace", size=13, color="#a0a0a0"),
                        title_font=dict(family="Courier New, monospace", size=18, color="#00ffff"),
                        xaxis=dict(gridcolor='#1a1a2e', linecolor='#00ffff'),
                        yaxis=dict(gridcolor='#1a1a2e', linecolor='#00ffff'),
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_compare, use_container_width=True)
                else:
                    # Results don't match current selection - clear them
                    del st.session_state['forecast_results']
                    st.info("👆 Run a forecast with the current model selection to see analytics here!")
            else:
                st.info("👆 Run a forecast first to see analytics here!")
    
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.error("Please check your data format and try again.")
        st.exception(e)
