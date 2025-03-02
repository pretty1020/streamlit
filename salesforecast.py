import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# Set Streamlit page layout
st.set_page_config(page_title="Sales Analytics Dashboard", layout="wide")

# Title
st.title("✨ Sales Analytics and Forecasting Dashboard ✨")

# Sidebar Section: File Upload
st.sidebar.header("📂 Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Data uploaded successfully!")
else:
    # Generate default dataset with realistic correlation
    np.random.seed(42)
    num_agents = 100
    agent_tenure = np.random.randint(3, 120, num_agents)  # In months
    avg_handle_time = np.random.uniform(3, 15, num_agents)  # In minutes

    # Sales count increases with AHT but declines at extremely high AHT
    sales_count = (avg_handle_time * 5) - (0.3 * avg_handle_time ** 2) + (agent_tenure * 0.5) + np.random.randint(-5, 5,
                                                                                                                  num_agents)
    sales_count = np.maximum(sales_count, 0)  # Ensure no negative values

    data = pd.DataFrame({
        'Agent Tenure (Months)': agent_tenure,
        'Average Handle Time (AHT)': avg_handle_time,
        'Sales Count': sales_count
    })

# Sidebar: Select Variables
st.sidebar.header("🔍 Select Variables")
variable_options = list(data.columns)
sales_variable = st.sidebar.selectbox("Select Sales Count Column", variable_options,
                                      index=variable_options.index("Sales Count"))
independent_variable = st.sidebar.selectbox("Select Variable to Analyze",
                                            [col for col in variable_options if col != sales_variable])
st.sidebar.write("⚠️ **Note:** This tool uses dummy data only for presentation purposes.")
# Tabs for Forecasting, Correlation, Regression Analysis, and Guide
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Sales Forecasting", "📊 Correlation Analysis", "📉 Regression Analysis", "📄 Appendix", "📘 User Guide"])
st.sidebar.write("[For any concerns or customization, reach out to Marian via Linkedin](https://www.linkedin.com/in/marian1020/)")
# --- Tab 1: Sales Forecasting ---
with tab1:
    st.subheader("🔮 Sales Forecasting Impact")
    st.dataframe(data)

    # Regression Model
    X = data[[independent_variable]]
    y = data[sales_variable]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    # Sales Forecast Table
    reg_model = LinearRegression()
    reg_model.fit(data[[independent_variable]], data[sales_variable])
    variable_values = np.linspace(data[independent_variable].min(), data[independent_variable].max(), 10).reshape(-1, 1)
    predicted_sales = reg_model.predict(variable_values)
    predicted_sales = np.maximum(predicted_sales, 0)

    forecast_df = pd.DataFrame(
        {independent_variable: variable_values.flatten(), "Predicted Sales Count": predicted_sales})
    st.dataframe(forecast_df)

    # 📊 Regression Visualization
    st.subheader("🌟 Sales Prediction with Regression Line")
    fig1, ax1 = plt.subplots(figsize=(7, 5), facecolor="#222831")
    sns.regplot(x=data[independent_variable], y=data[sales_variable], ax=ax1, line_kws={"color": "#FFD700"},
                scatter_kws={"color": "#00FFFF", "alpha": 0.7})
    ax1.set_title(f"Impact of {independent_variable} on Sales Count", color="white", fontsize=14)
    ax1.set_xlabel(independent_variable, color="white")
    ax1.set_ylabel("Sales Count", color="white")
    ax1.set_facecolor("#222831")
    plt.xticks(color="white")
    plt.yticks(color="white")
    st.pyplot(fig1)


    # Interpretation based on predicted sales count
    min_pred = predicted_sales.min()
    max_pred = predicted_sales.max()
    avg_pred = predicted_sales.mean()

    st.write(
        f"📊 **Interpretation:** The predicted sales count ranges from {min_pred:.2f} to {max_pred:.2f}, with an average of {avg_pred:.2f}. This suggests that variations in {independent_variable} have a measurable impact on sales performance, with an optimal range that maximizes predicted sales.")

with tab3:
    st.subheader(f"📉 Regression Analysis: {independent_variable}")

    X = data[[independent_variable]]
    y = data[sales_variable]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    reg_model = LinearRegression()
    reg_model.fit(data[[independent_variable]], data[sales_variable])

    fig3, ax3 = plt.subplots(figsize=(7, 5), facecolor="#222831")
    sns.scatterplot(x=data[independent_variable], y=data[sales_variable], ax=ax3, color="#00FFFF", alpha=0.6)
    sns.lineplot(x=data[independent_variable], y=reg_model.predict(data[[independent_variable]]), ax=ax3,
                 color="#FFD700")
    ax3.set_title(f"Regression Plot: {independent_variable} vs. Sales Count", color="white", fontsize=14)
    ax3.set_xlabel(independent_variable, color="white")
    ax3.set_ylabel("Sales Count", color="white")
    ax3.set_facecolor("#222831")
    plt.xticks(color="white")
    plt.yticks(color="white")
    st.pyplot(fig3)

    if independent_variable == "Agent Tenure (Months)":
        st.write(
            f"💡 **Interpretation:** The regression analysis indicates that agent tenure positively correlates with sales count. More experienced agents tend to close more sales.")
    else:
        st.write(
            f"💡 **Interpretation:** The analysis shows that {independent_variable} initially increases sales, but extremely high values cause a decline. This suggests an optimal range for maximizing sales.")



# --- Tab 2: Correlation Analysis ---
with tab2:
    st.subheader("🔬 Correlation Analysis")
    corr = data.corr()
    fig2, ax2 = plt.subplots(figsize=(6, 4), facecolor="#7d91b0")
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax2, annot_kws={"color": "white"})
    st.pyplot(fig2)

    st.write(
        "💡 **Interpretation:** The correlation matrix shows the relationship between all variables. Below is a table for correlation strength interpretation:")

    correlation_table = pd.DataFrame({
        "Correlation Coefficient Range": ["0.00 - 0.19", "0.20 - 0.39", "0.40 - 0.59", "0.60 - 0.79", "0.80 - 1.00"],
        "Interpretation": ["Very Weak", "Weak", "Moderate", "Strong", "Very Strong"]
    })
    st.table(correlation_table)


# --- Tab 4: Appendix ---
with tab4:

    st.subheader("📊 Regression Model Summary")
    st.text(model.summary())

    st.subheader("📘 Key Insights:")
    st.write("""
    - The p-value indicates statistical significance of the independent variable's effect on sales count.
    - The R-squared value represents how well the model explains the variance in sales.
    - If the coefficient is positive, the independent variable has a positive impact on sales.
    - If the coefficient is negative, increasing the independent variable decreases sales count.
    """)

# --- Tab 5: User Guide and Definitions ---
with tab5:
    st.subheader("📘 User Guide")
    st.write("""
    - **Upload Data**: Upload your own CSV file or use the default dataset.
    - **Select Variables**: Choose the independent variable for analysis.
    - **Sales Forecasting**: Predict sales based on historical data.
    - **Correlation Analysis**: Understand the relationship between variables.
    - **Regression Analysis**: See if your chosen variable significantly impacts sales.
    """)

    st.subheader("📖 Definition of Terms")
    st.write("""
    - **AHT (Average Handle Time)**: The average duration of customer interactions.
    - **Sales Count**: The number of sales closed by an agent.
    - **Regression Analysis**: A statistical method to predict outcomes.
    - **Correlation**: A measure of the relationship between two variables.
    - **P-Value**: Determines statistical significance of an independent variable.
    - **R-Squared**: Measures the proportion of variance in sales explained by the independent variable.
    """)
