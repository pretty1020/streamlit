import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="FTE Capacity Planning Tool", layout="wide", page_icon="📊")



st.title("📊 FTE Capacity Planning Tool")

st.sidebar.title("User Guide & Definitions")
st.sidebar.markdown("""
## **User Guide**
- **Step 1:** Select **Channel Type** (Inbound or Back Office).
- **Step 2:** Enter forecast inputs (**Volume, AHT (Seconds), Occupancy, Shrinkage, SCF**).
- **Step 3:** If **Back Office Channel** is selected, enter **Concurrency**.
- **Step 4:** Define **Starting FTE Count** and **Attrition Rate per Week**.
- **Step 5:** Edit the **FTE Planning Table** if needed.
- **Step 6:** View **Updated FTE Planning Table**, **Graphs**, and **Download the Plan**.

## **Definition of Terms**
- **FTE Required:** Full-Time Equivalents needed for forecasted workload.
- **Available FTEs:** Net FTEs after accounting for shrinkage.
- **Estimated Production Hours:** Available FTEs multiplied by 40 productive hours per week.
- **Estimated Handling Capacity:** Capacity based on production hours, AHT, occupancy, and concurrency.
- **Over/Under:** Difference between **Current FTEs** and **Required FTEs**.
- **Attrition Rate:** Percentage of employees leaving per week.
- **Concurrency:** Number of simultaneous tasks handled by an agent (e.g., handling multiple chats).
- **SCF (Service Capacity Factor):** Adjusts FTE needs based on efficiency.
""")
st.sidebar.markdown("[Reach out to Marian via Linkedin](https://www.linkedin.com/in/marian1020/)")

channel_type = st.radio("Select Channel Type", ("Inbound", "Back Office (Cases, Chat, Email, etc.)"))

weeks = st.number_input("Number of Weeks to Plan", min_value=4, max_value=52, value=12)
volume = st.number_input("Weekly Volume", min_value=0, value=1000)
aht = st.number_input("Average Handle Time (seconds)", min_value=0, value=810)
occupancy = st.slider("Occupancy (%)", min_value=0.5, max_value=1.0, value=0.7, step=0.1)
shrinkage = st.slider("Shrinkage (%)", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
scf = st.number_input("Service Capacity Factor (SCF)", min_value=1.0, max_value=2.0, value=1.01, step=0.1)
concurrency = 1.0

if channel_type == "Back Office (Cases, Chat, Email, etc.)":
    concurrency = st.number_input("Concurrency (e.g., 1.5 for Chat)", min_value=1.0, value=1.5, step=0.1)

def calculate_fte(volume, aht, occupancy, shrinkage, scf, concurrency):
    productive_hours_per_fte = 40 * occupancy
    total_hours_needed = volume * aht / 3600 / (1 - shrinkage) * scf
    return round(total_hours_needed / (productive_hours_per_fte * concurrency), 2)

starting_fte = st.number_input("Starting FTE Count", min_value=0, value=int(calculate_fte(volume, aht, occupancy, shrinkage, scf, concurrency)))
attrition_rate = st.slider("Attrition Rate per Week (%)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)

data = {
    "Week": [f"Week {i + 1}" for i in range(weeks)],
    "Volume": [volume] * weeks,
    "AHT": [aht] * weeks,
    "Occupancy": [occupancy] * weeks,
    "Shrinkage": [shrinkage] * weeks,
    "SCF": [scf] * weeks,
    "Concurrency": [concurrency] * weeks,
    "Current FTEs": [starting_fte] + [0.0] * (weeks - 1),
    "New FTEs(NH Grad)": [0.0] * weeks,
    "Attrition Rate": [attrition_rate] * weeks,
}

df = pd.DataFrame(data)

def update_fte_plan(df):
    for i in range(len(df)):
        df.loc[i, "FTE Required"] = calculate_fte(
            df.loc[i, "Volume"], df.loc[i, "AHT"], df.loc[i, "Occupancy"], df.loc[i, "Shrinkage"], df.loc[i, "SCF"], df.loc[i, "Concurrency"]
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
    df["Estimated Handling Capacity"] = df["Estimated Production Hours"] * 3600 / df["AHT"] * df["Occupancy"] * df["Concurrency"]
    df["Over/Under"] = df["Current FTEs"] - df["FTE Required"]
    return df

edited_df = st.data_editor(df, num_rows="dynamic")

updated_df = update_fte_plan(edited_df)

st.subheader("📊 Updated FTE Planning Table")
st.dataframe(updated_df)

st.subheader("📈 Weekly FTE Trends")
st.plotly_chart(px.line(updated_df, x="Week", y=["FTE Required", "Available FTEs"], markers=True))

st.subheader("📊 FTE Shortage/Surplus (Over/Under)")
st.plotly_chart(px.line(updated_df, x="Week", y="Over/Under", markers=True))

st.download_button("Download FTE Plan", data=updated_df.to_csv(index=False), file_name="fte_capacity_plan.csv")
