import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px


# Set page configuration
st.set_page_config(page_title="FTE Capacity Planning Tool", layout="wide", page_icon="📊")

# Sidebar Navigation

tab = st.sidebar.radio("Go to:", ["FTE Planning Tool", "User Guide"])
st.sidebar.markdown(
        "[For any concerns or issues,feel free to reach out to Marian via Linkedin](https://www.linkedin.com/in/marian1020/)")

# User Guide Section
if tab == "User Guide":
    st.title("📖 User Guide & Definitions")

    st.markdown("""
    ## **User Guide**
    - **Step 1:** Enter forecast inputs (**Call Volume, AHT (in seconds), Occupancy, Shrinkage, SCF**).
    - **Step 2:** Define **Starting FTE Count** and **Attrition Rate per Week**.
    - **Step 3:** The tool **auto-calculates** the required FTE based on inputs.
    - **Step 4:** Enter planned **New FTE hires per week**.
    - **Step 5:** View the **calculated results** and **visualizations**.
    - **Step 6:** **Download the FTE Plan** for analysis.

    ## **Definition of Terms**
    - **FTE Required:** Number of Full-Time Equivalents needed for forecasted workload.
    - **Available FTEs:** Net FTEs after accounting for shrinkage.
    - **Over/Under:** Difference between **Current FTEs** and **Required FTEs**.
    - **Attrition Rate:** Percentage of employees leaving per week.
    - **SCF (Service Capacity Factor):** Adjusts FTE needs based on efficiency.
    """)


    st.info("🔄 **Switch to the 'FTE Planning Tool' tab in the sidebar to start planning.**")
    st.stop()

# Function to calculate FTE requirement
def calculate_fte(volume, aht, occupancy, shrinkage, scf):
    productive_hours_per_fte = 40 * occupancy
    total_hours_needed = volume * aht / 3600 / (1 - shrinkage) * scf
    return round(float(total_hours_needed / productive_hours_per_fte), 2)

# Function to update FTE plan dynamically
def update_fte_plan(df):
    for i in range(len(df)):
        df.loc[i, "FTE Required"] = calculate_fte(
            df.loc[i, "Volume"], df.loc[i, "AHT"], df.loc[i, "Occupancy"], df.loc[i, "Shrinkage"], df.loc[i, "SCF"]
        )

        if i == 0:
            df.loc[i, "Attrition"] = df.loc[i, "Current FTEs"] * df.loc[i, "Attrition Rate"]
            df.loc[i, "Net FTEs"] = df.loc[i, "Current FTEs"] + df.loc[i, "New FTEs(NH Grad)"] - df.loc[i, "Attrition"]
        else:
            df.loc[i, "Current FTEs"] = round(df.loc[i - 1, "Net FTEs"], 2)
            df.loc[i, "Attrition"] = df.loc[i, "Current FTEs"] * df.loc[i, "Attrition Rate"]
            df.loc[i, "Net FTEs"] = df.loc[i, "Current FTEs"] + df.loc[i, "New FTEs(NH Grad)"] - df.loc[i, "Attrition"]

    df["Available FTEs"] = df["Net FTEs"] * (1 - df["Shrinkage"])
    df["Estimated Production Hours"] = df["Available FTEs"] * 40
    df["Estimated Handling Capacity"] = df["Estimated Production Hours"] * 3600 / df["AHT"] * df["Occupancy"]
    df["Over/Under"] = df["Current FTEs"] - df["FTE Required"]

    return df

# Function to format numbers (remove trailing zeros but keep up to 2 decimal places)
def format_numbers(val):
    if isinstance(val, (float, int)):
        return '{:.2f}'.format(val).rstrip('0').rstrip('.')
    return val

# Function for conditional formatting
def highlight_calculated_columns(val):
    return "background-color: lightblue; font-weight: bold"

def color_over_under(val):
    try:
        val = float(val)
        if val > 0:
            return "background-color: green; color: white"
        elif val < 0:
            return "background-color: red; color: white"
        else:
            return "background-color: yellow; color: black"
    except ValueError:
        return ""

# Main FTE Planning Tool
st.title("📊 FTE Capacity Planning Tool")

# User inputs for planning
st.text("Default value only:")
weeks = st.number_input("Number of Weeks to Plan", min_value=4, max_value=52, value=12)
volume = st.number_input("Weekly Call Volume", min_value=0, value=1000)
aht = st.number_input("Average Handle Time (seconds)", min_value=0, value=810)
occupancy = st.slider("Occupancy (%)", min_value=0.5, max_value=1.0, value=0.70, step=0.1)
shrinkage = st.slider("Shrinkage (%)", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
scf = st.number_input("Service Capacity Factor (SCF)", min_value=1.0, max_value=2.0, value=1.01, step=0.1)
starting_fte = st.number_input("Starting FTE Count", min_value=0, value=int(calculate_fte(volume, aht, occupancy, shrinkage, scf)))
attrition_rate = st.slider("Attrition Rate per Week (%)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)

# Create DataFrame
data = {
    "Week": [f"Week {i + 1}" for i in range(weeks)],
    "Volume": [volume] * weeks,
    "AHT": [aht] * weeks,
    "Occupancy": [occupancy] * weeks,
    "Shrinkage": [shrinkage] * weeks,
    "SCF": [scf] * weeks,
    "Current FTEs": [starting_fte] + [0.0] * (weeks - 1),
    "New FTEs(NH Grad)": [0.0] * weeks,
    "Attrition Rate": [attrition_rate] * weeks,
}

df = pd.DataFrame(data)

# User-editable table (No calculated fields shown)
st.subheader("📊 FTE Planning Table (User Inputs Only)")
edited_df = st.data_editor(df, num_rows="dynamic")

# Update calculations dynamically
updated_df = update_fte_plan(edited_df)

# Remove trailing zeros in Updated FTE Planning Table
for col in updated_df.columns:
    if col != "Week":
        updated_df[col] = updated_df[col].apply(format_numbers)

# Display updated table with calculated fields highlighted
st.subheader("📊 Updated FTE Planning Table")
styled_df = updated_df.style.map(highlight_calculated_columns, subset=["FTE Required", "Available FTEs", "Estimated Production Hours", "Estimated Handling Capacity"])\
                            .map(color_over_under, subset=["Over/Under"])
st.dataframe(styled_df)

# Charts
st.subheader("📈 Weekly FTE Trends")
st.plotly_chart(px.line(updated_df, x="Week", y=["FTE Required", "Available FTEs"], markers=True))


st.subheader("📊 FTE Shortage/Surplus (Over/Under)")
st.plotly_chart(px.line(updated_df, x="Week", y="Over/Under", markers=True))

st.download_button("📥 Download FTE Plan", data=updated_df.to_csv(index=False), file_name="fte_capacity_plan.csv")
