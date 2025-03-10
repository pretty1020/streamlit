import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Function to generate dummy data
def generate_dummy_data():
    num_agents = 4  # Number of agents for historical data
    num_weeks = 8  # Number of weeks in nesting phase

    agents = np.repeat([f"Agent {i + 1}" for i in range(num_agents)], num_weeks)
    weeks = np.tile(np.arange(1, num_weeks + 1), num_agents)
    base_aht = np.random.randint(180, 300, num_agents)
    decay_rate = np.random.uniform(0.85, 0.95, num_agents)
    call_volumes = np.random.randint(50, 200, size=len(weeks))  # Random call volumes

    aht_scores = []
    for i in range(num_agents):
        aht_scores.extend(base_aht[i] * (decay_rate[i] ** weeks[:num_weeks]))

    return pd.DataFrame({"Agent": agents, "Week": weeks, "AHT": np.round(aht_scores, 2), "Call Volume": call_volumes})


# Streamlit UI Layout with Tabs
st.set_page_config(page_title="Nesting Agent Learning Curve Analytics Tool", layout="wide")

tabs = st.tabs(
    ["📊 Learning Curve Analysis", "📈 Visualizations & Charts", "📊 Advanced Analytics", "📖 User Guide & Definitions"])

# 📖 User Guide & Definitions Tab
with tabs[3]:
    st.header("📖 User Guide & Definitions")

    st.subheader("🔹 What is this tool?")
    st.markdown("""
    This **Nesting Agent Learning Curve Analytics Tool** helps call centers monitor and analyze the performance of new agents in training (nesting phase) by:
    - **Tracking Weekly AHT improvement**
    - **Analyzing how many weeks it takes to reach a target AHT**
    - **Visualizing the learning curve of new agents**
    - **Providing advanced analytics and predictive modeling for workforce planning**
    """)

    st.subheader("🔹 How to use this tool?")
    st.markdown("""
    1. **Upload a CSV file** with the following columns:
        - **"Agent"** (Agent ID or Name)
        - **"Week"** (Week number during the nesting phase)
        - **"AHT"** (Average Handling Time per week)
        - **"Call Volume"** (Number of calls handled in that week)
    2. **Click "Analyze Learning Curve"** to generate insights.
    3. **View weekly performance trends and how agents improve over time.**
    4. **Use the Advanced Analytics tab to predict how many weeks it will take to reach a target AHT.**
    5. **Download the processed report for further analysis.**
    """)

    st.write("---")
    st.write("🚀 Built with ❤️ by Marian")
    st.write("For any concerns or issues,feel free to reach out to Marian via Linkedin: https://www.linkedin.com/in/marian1020/")

# 📊 Learning Curve Analysis Tab
with tabs[0]:
    st.title("📊 Nesting Agent Learning Curve Analytics Tool")
    st.markdown("🔍 **Upload agent performance data to analyze learning curves and AHT trends.**")

    # 📂 Upload CSV File
    st.subheader("📂 Upload Learning Data")
    uploaded_file = st.file_uploader("Upload a CSV file with 'Agent', 'Week', 'AHT', and 'Call Volume' columns",
                                     type=["csv"])

    if uploaded_file:
        df_learning = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
    else:
        st.warning("⚠️ No file uploaded. Using default dummy data.")
        df_learning = generate_dummy_data()

    st.subheader("📋 Preview of Learning Data")
    st.dataframe(df_learning)

    # 📊 Display Key Metrics (AHT Weighted by Call Volume)
    total_calls = df_learning["Call Volume"].sum()
    weighted_aht = np.sum(df_learning["AHT"] * df_learning["Call Volume"]) / total_calls
    min_aht = df_learning["AHT"].min()
    max_aht = df_learning["AHT"].max()
    st.metric(label="📊 AHT (Weighted by Call Volume)", value=f"{weighted_aht:.2f} seconds")
    st.metric(label="📉 Minimum AHT Achieved", value=f"{min_aht:.2f} seconds")
    st.metric(label="📈 Maximum AHT Observed", value=f"{max_aht:.2f} seconds")

    # 📥 Download Processed Report
    csv_download = df_learning.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Analysis Report", data=csv_download, file_name="learning_curve_analysis.csv",
                       mime="text/csv")

# 📈 Visualizations & Charts Tab
with tabs[1]:
    st.title("📈 Learning Curve Visualizations")

    st.subheader("📊 Weekly AHT Trend")
    fig, ax = plt.subplots(figsize=(6, 3))  # Smaller figure size
    sns.lineplot(data=df_learning, x="Week", y="AHT", hue="Agent", marker='o', linewidth=2.5, alpha=0.8, ax=ax)
    ax.set_xlabel("Week")
    ax.set_ylabel("AHT (Seconds)")
    ax.set_title("AHT Trend Over Weeks (Per Agent)")
    ax.grid(color='gray', linestyle='dotted', linewidth=0.5)
    st.pyplot(fig)

# 📊 Advanced Analytics Tab (Machine Learning Implementation)
with tabs[2]:
    st.title("📊 Advanced Learning Curve Predictions")

    st.subheader("📈 Machine Learning Prediction for AHT Improvement")

    # Machine Learning Model for AHT Prediction
    X = df_learning[["Week"]]  # Using Week as predictor
    y = df_learning["AHT"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Display Model Performance
    st.write("**Model Performance Metrics:**")
    st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

    # Future Prediction Input
    st.subheader("🔮 Predict Weeks to Reach Target AHT")
    target_aht = st.number_input("Enter the target AHT (seconds):", min_value=1.0, max_value=1600.0, value=120.0,
                                 step=5.0)
    predicted_weeks = (target_aht - model.intercept_) / model.coef_[0]
    predicted_weeks = max(0, predicted_weeks)  # Ensure it's not negative
    st.write(f"**Predicted Weeks to Achieve AHT {target_aht} seconds: {predicted_weeks:.2f} weeks**")

    # 📈 Visualization of Predictions & Confidence Intervals
    st.subheader("📊 Prediction Visualization")

    # Generate Predictions for Visualization
    weeks_future = np.arange(1, 12).reshape(-1, 1)  # Predict for 12 weeks
    predictions = model.predict(weeks_future)

    # Compute Confidence Intervals
    y_std = np.std(y_test - y_pred)
    lower_bound = predictions - (1.96 * y_std)
    upper_bound = predictions + (1.96 * y_std)

    fig, ax = plt.subplots(figsize=(6, 3), facecolor='black')
    ax.set_facecolor('black')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.tick_params(colors='white', which='both')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')

    # Glowy Prediction Line
    ax.plot(weeks_future, predictions, label="Predicted AHT", color="cyan", linewidth=2.5, alpha=0.8)
    ax.fill_between(weeks_future.flatten(), lower_bound, upper_bound, color='cyan', alpha=0.2,
                    label="95% Confidence Interval")

    ax.set_xlabel("Week")
    ax.set_ylabel("AHT (Seconds)")
    ax.set_title("Predicted AHT Over Time with Confidence Intervals", fontsize=12)
    ax.legend()
    ax.grid(color='gray', linestyle='dotted', linewidth=0.5)

    st.pyplot(fig)

