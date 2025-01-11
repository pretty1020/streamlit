import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Page configuration
st.set_page_config(
    page_title="Call Center Shrinkage Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Title
st.title("📊 Call Center Shrinkage Analysis Dashboard")

# Sidebar
st.sidebar.header("Navigation")
view_selection = st.sidebar.radio("Select View:", ["Shrinkage Analysis", "Data Analysis and Forecast"])

st.sidebar.header("Filters")
team_filter = st.sidebar.multiselect("Select Team(s):", options=["Team A", "Team B", "Team C", "Team D"], default=["Team A", "Team B", "Team C", "Team D"])

# Dummy weekly data generation
def generate_weekly_dummy_data():
    np.random.seed(42)
    teams = ["Team A", "Team B", "Team C", "Team D"]
    weeks = pd.date_range(start="2023-01-01", periods=52, freq="W-SUN")
    data = {
        "Week": np.random.choice(weeks, 1000),
        "Team": np.random.choice(teams, 1000),
        "Planned Hours": np.random.randint(160, 200, 1000),
        "Worked Hours": np.random.randint(140, 190, 1000),
        "Absenteeism": np.random.randint(0, 20, 1000),
        "Default AUX": np.random.randint(5, 15, 1000),
        "Break": np.random.randint(5, 15, 1000),
        "Coaching": np.random.randint(2, 10, 1000),
        "Other Shrinkage": np.random.randint(1, 10, 1000),
    }
    df = pd.DataFrame(data)
    df["Total Shrinkage"] = (
        df["Absenteeism"] + df["Default AUX"] + df["Break"] + df["Coaching"] + df["Other Shrinkage"]
    ) / df["Planned Hours"] * 100
    return df

# Load data
data = generate_weekly_dummy_data()
data = data[data["Team"].isin(team_filter)]

if view_selection == "Shrinkage Analysis":
    # Metrics section
    st.markdown("### Key Metrics")
    total_planned_hours = data["Planned Hours"].sum()
    total_worked_hours = data["Worked Hours"].sum()
    average_shrinkage = data["Total Shrinkage"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Planned Hours", f"{total_planned_hours:,} hrs")
    col2.metric("Total Worked Hours", f"{total_worked_hours:,} hrs")
    col3.metric("Average Shrinkage", f"{average_shrinkage:.2f}%")

    # Visualizations
    st.markdown("### Shrinkage Analysis Visualizations")

    # Shrinkage by Team
    shrinkage_by_team = data.groupby("Team")["Total Shrinkage"].mean().reset_index()
    fig_team = px.bar(
        shrinkage_by_team,
        x="Team",
        y="Total Shrinkage",
        color="Team",
        title="Average Shrinkage by Team",
        text_auto=".2f",
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig_team.update_layout(template="plotly_white")
    st.plotly_chart(fig_team, use_container_width=True)

    # Shrinkage Breakdown
    shrinkage_breakdown = data[["Absenteeism", "Default AUX", "Break", "Coaching", "Other Shrinkage"]].mean().reset_index()
    shrinkage_breakdown.columns = ["Shrinkage Type", "Average Hours"]
    fig_breakdown = px.pie(
        shrinkage_breakdown,
        names="Shrinkage Type",
        values="Average Hours",
        title="Shrinkage Breakdown",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_breakdown.update_layout(template="plotly_white")
    st.plotly_chart(fig_breakdown, use_container_width=True)

    # Shrinkage Distribution
    fig_distribution = px.histogram(
        data,
        x="Total Shrinkage",
        nbins=20,
        title="Shrinkage Distribution",
        color_discrete_sequence=["#636EFA"],
    )
    fig_distribution.update_layout(template="plotly_white")
    st.plotly_chart(fig_distribution, use_container_width=True)

    # Worked vs Planned Hours
    data['Shrinkage_Size'] = data['Total Shrinkage'].clip(lower=0)  # Clip negative values to 0
    fig_hours = px.scatter(
        data,
        x="Planned Hours",
        y="Worked Hours",
        color="Team",
        size="Shrinkage_Size",  # Use clipped Shrinkage values
        hover_data=["Absenteeism", "Default AUX", "Break", "Coaching", "Other Shrinkage"],
        title="Worked Hours vs Planned Hours",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_hours.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    fig_hours.update_layout(template="plotly_white")
    st.plotly_chart(fig_hours, use_container_width=True)

    # Detailed Table
    st.markdown("### Detailed Data Table")
    st.dataframe(data)

else:
    st.markdown("### Data Analysis and Forecast")

    # Prepare data for forecasting
    forecast_data = data.groupby("Week").agg({
        "Total Shrinkage": "mean"
    }).reset_index()
    forecast_data.columns = ["Week", "Shrinkage"]
    forecast_data.set_index("Week", inplace=True)

    # Advanced Data Analytics
    st.markdown("#### Advanced Data Analytics")

    # Linear Regression Analysis
    X = data[["Planned Hours"]]
    y = data["Worked Hours"]
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)

    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R-squared: {r2:.2f}")

    fig_regression = px.scatter(
        data,
        x="Planned Hours",
        y="Worked Hours",
        trendline="ols",
        title="Linear Regression: Planned vs Worked Hours"
    )
    st.plotly_chart(fig_regression, use_container_width=True)

    # ARIMA Forecast
    st.markdown("#### Shrinkage Forecast for the Next 6 Weeks")
    model_arima = ARIMA(forecast_data["Shrinkage"], order=(2, 1, 2))
    model_fit_arima = model_arima.fit()
    forecast = model_fit_arima.forecast(steps=6)
    future_weeks = pd.date_range(forecast_data.index[-1], periods=7, freq="W")[1:]
    forecast_df = pd.DataFrame({"Week": future_weeks, "Predicted Shrinkage": forecast})

    # Combine existing and forecasted data
    combined_weeks = forecast_data.index.union(forecast_df["Week"])
    combined_shrinkage = pd.concat([forecast_data["Shrinkage"], forecast_df.set_index("Week")["Predicted Shrinkage"]])

    # Plot Forecast
    fig_forecast = px.line(
        x=combined_weeks,
        y=combined_shrinkage,
        labels={"x": "Week", "y": "Shrinkage (%)"},
        title="Shrinkage Forecast"
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Summary of Forecast
    st.markdown("#### Summary of Forecast")
    st.dataframe(forecast_df)
    st.write(f"Predicted shrinkage ranges between {forecast_df['Predicted Shrinkage'].min():.2f}% and {forecast_df['Predicted Shrinkage'].max():.2f}%.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by Marian")
