import streamlit as st
import pandas as pd
import plotly.express as px
import math
from datetime import datetime, timedelta

st.set_page_config(page_title="FTE Capacity Planning Tool", layout="wide", page_icon="📊")
st.title("📊 FTE Capacity Planning Tool")

# Sidebar content remains the same.
st.sidebar.title("User Guide & Definitions")
st.sidebar.markdown("""
## **User Guide**
- **Step 1:** Select **Channel Type** (Inbound or Back Office).
- **Step 2:** Enter forecast inputs (**Volume, AHT (Seconds), Occupancy, Shrinkage, SCF**).
- **Step 3:** If **Back Office Channel** is selected, enter **Concurrency**.
- **Step 4:** Define **Starting FTE Count** and **Attrition Rate per Week**.
- **Step 5:** Edit the **FTE Planning Table** if needed.
- **Step 6:** View **Updated FTE Planning Table**, **Graphs**, and **Download the Plan**.

## **Definition of Terms (Weekly)**
- **FTE Required:** Full-Time Equivalents needed for forecasted workload using Erlang C.
- **Available FTEs:** Net FTEs after accounting for shrinkage.
- **Estimated Production Hours:** Available FTEs multiplied by 40 productive hours per week.
- **Estimated Handling Capacity:** Capacity based on production hours, AHT, occupancy, and concurrency.
- **Over/Under:** Difference between **Current FTEs** and **FTE Required**.
- **Attrition Rate:** Percentage of employees leaving per week.
- **Concurrency:** Number of simultaneous tasks handled by an agent.
- **SCF (Schedule Challenge Factor):** Adjusts FTE needs based on efficiency.
- **Net FTEs:** Total FTEs after accounting for new hires and attrition.
""")
st.sidebar.markdown("[Reach out to Marian via LinkedIn](https://www.linkedin.com/in/marian1020/)")

# Create two tabs.
tab1, tab2 = st.tabs(["Weekly FTE Planning", "Intraday FTE Planning"])

# ---------------- Weekly FTE Planning Tab ----------------
with tab1:
    channel_type = st.radio("Select Channel Type", ("Inbound", "Back Office (Cases, Chat, Email, etc.)"))
    weeks = st.number_input("Number of Weeks to Plan", min_value=4, max_value=52, value=12)
    volume = st.number_input("Weekly Volume", min_value=0, value=1000)
    aht = st.number_input("Average Handle Time (seconds)", min_value=0, value=810)
    occupancy = st.slider("Occupancy (%)", min_value=0.5, max_value=1.0, value=0.7, step=0.1)
    shrinkage = st.slider("Shrinkage (%)", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    scf = st.number_input("Schedule Challenge Factor (SCF)", min_value=1.0, max_value=2.0, value=1.01, step=0.1)
    concurrency = 1.0
    if channel_type == "Back Office (Cases, Chat, Email, etc.)":
        concurrency = st.number_input("Concurrency (e.g., 1.5 for Chat)", min_value=1.0, value=1.5, step=0.1)

    # New inputs for SLA target in weekly planning.
    weekly_sla_target = st.slider("Weekly Service Level Target (%)", min_value=0.5, max_value=1.0, value=0.8, step=0.05)
    weekly_target_time = st.number_input("Weekly SLA Target Time (seconds)", min_value=1, value=20)


    # Helper function: Erlang C formula (used in both weekly and intraday calculations).
    def erlang_c_formula(A, c):
        if c <= A:
            return 1.0
        sum_terms = sum([A ** n / math.factorial(n) for n in range(c)])
        term_c = A ** c / math.factorial(c)
        return (term_c * (c / (c - A))) / (sum_terms + term_c * (c / (c - A)))


    # New function: Calculate weekly FTE required using an Erlang C approach with SLA target.
    def erlang_c_weekly(volume, aht, shrinkage, occupancy, scf, concurrency, sla_target, target_time):
        # Convert weekly volume to an hourly average (assuming a 40-hour work week).
        hourly_volume = volume / 40.0
        # Offered load (in Erlangs) for an hour.
        A = (hourly_volume * aht) / 3600.0
        # Start with the smallest integer number of agents that can handle the load.
        c = math.ceil(A)
        max_agents = c + 100  # upper bound to avoid infinite loop.
        while c < max_agents:
            ec = erlang_c_formula(A, c)
            prob_wait = ec * math.exp(-(c - A) * target_time / aht)
            projected_sla = 1 - prob_wait
            if projected_sla >= sla_target:
                break
            c += 1
        # Convert the required concurrent agents (c) into FTE count.
        # Adjust for concurrency (an agent may handle more than one call simultaneously),
        # occupancy, and shrinkage (non-productive time).
        fte_required = (c / concurrency) / (occupancy * (1 - shrinkage))
        fte_required = round(fte_required * scf, 2)
        return fte_required


    # Set starting FTE to the computed value by default.
    starting_fte = st.number_input("Starting FTE Count", min_value=0,
                                   value=int(erlang_c_weekly(volume, aht, shrinkage, occupancy, scf, concurrency,
                                                             weekly_sla_target, weekly_target_time)))
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
            # Updated to use erlang_c_weekly with the weekly SLA target and target time.
            df.loc[i, "FTE Required"] = erlang_c_weekly(
                df.loc[i, "Volume"], df.loc[i, "AHT"], df.loc[i, "Shrinkage"],
                df.loc[i, "Occupancy"], df.loc[i, "SCF"], df.loc[i, "Concurrency"],
                weekly_sla_target, weekly_target_time
            )
            df.loc[i, "Net FTEs"] = df.loc[i, "Current FTEs"] + df.loc[i, "New FTEs(NH Grad)"] - (
                        df.loc[i, "Current FTEs"] * df.loc[i, "Attrition Rate"])
            df.loc[i, "Available FTEs"] = df.loc[i, "Net FTEs"] * (1 - df.loc[i, "Shrinkage"])
            df.loc[i, "Estimated Production Hours"] = df.loc[i, "Available FTEs"] * 40
            df.loc[i, "Estimated Handling Capacity"] = (
                        df.loc[i, "Estimated Production Hours"] * 3600 / df.loc[i, "AHT"] * df.loc[i, "Occupancy"] *
                        df.loc[i, "Concurrency"])
            df.loc[i, "Over/Under"] = df.loc[i, "Current FTEs"] - df.loc[i, "FTE Required"]
        return df


    edited_df = st.data_editor(df, num_rows="dynamic")
    updated_df = update_fte_plan(edited_df)
    st.subheader("📊 Updated FTE Planning Table")
    st.dataframe(updated_df)
    st.download_button("Download FTE Plan", data=updated_df.to_csv(index=False), file_name="fte_capacity_plan.csv")

# ---------------- Intraday FTE Planning Tab ----------------
with tab2:
    st.subheader("Intraday FTE Planning (30-Minute Intervals)")
    st.markdown("""
    **Intraday Definitions:**
    - **Volume:** Number of contacts handled during the 30-minute interval.
    - **AHT:** Average Handling Time (in seconds) for contacts in the interval.
    - **Max Occupancy:** The maximum allowed occupancy (as a fraction, e.g., 0.7 means 70%).
    - **SLA Target:** Target service level.
    - **Available FTE:** Full-Time Equivalents available for the interval.
    - **FTE Required:** Calculated number of agents needed based on workload using Erlang C.
    - **Ideal Occupancy:** Estimated occupancy if the available agents are fully utilized.
    - **Projected Service Level:** Estimated service level based on an Erlang C formula.
    """)
    target_time_intraday = st.number_input("Target Time for SLA (seconds)", min_value=1, value=20)

    # Generate all 48 intervals (each 30 minutes)
    all_intervals = []
    base_start = datetime.strptime("00:00", "%H:%M")
    for i in range(48):
        current_start = base_start + timedelta(minutes=30 * i)
        current_end = current_start + timedelta(minutes=30)
        interval_str = f"{current_start.strftime('%H:%M')}-{current_end.strftime('%H:%M')}"
        all_intervals.append(interval_str)

    # Build available start and end times.
    available_start_times = sorted({interval.split("-")[0] for interval in all_intervals})
    available_end_times = sorted({interval.split("-")[1] for interval in all_intervals})

    # Let the user choose operating hours.
    # Default operating hours: Start at "20:00" and End at "05:00".
    selected_start_time = st.selectbox("Select Operation Start Time", options=available_start_times, index=40)
    selected_end_time = st.selectbox("Select Operation End Time", options=available_end_times, index=10)

    # Create a full intraday DataFrame with only the Interval column; other columns are blank.
    intraday_data = {
        "Interval": all_intervals,
        "Volume": [None] * len(all_intervals),
        "AHT": [None] * len(all_intervals),
        "Max Occupancy": [None] * len(all_intervals),
        "SLA Target": [None] * len(all_intervals),
        "Available FTE": [None] * len(all_intervals),
    }
    full_intraday_df = pd.DataFrame(intraday_data)


    # Filter rows based on selected operating hours.
    def in_time_range(interval_str):
        start_str, end_str = interval_str.split("-")
        interval_start = datetime.strptime(start_str, "%H:%M").time()
        interval_end = datetime.strptime(end_str, "%H:%M").time()
        selected_start = datetime.strptime(selected_start_time, "%H:%M").time()
        selected_end = datetime.strptime(selected_end_time, "%H:%M").time()
        # If operating hours do not wrap around midnight.
        if selected_start <= selected_end:
            return (interval_start >= selected_start) and (interval_end <= selected_end)
        else:
            # Wrap-around: include intervals that start on or after the selected start OR end on or before the selected end.
            return (interval_start >= selected_start) or (interval_end <= selected_end)


    filtered_intraday_df = full_intraday_df[full_intraday_df["Interval"].apply(in_time_range)].reset_index(drop=True)


    # To ensure the filtered rows are in the correct order (starting from the selected start time),
    # we define a sort key. For wrap-around, intervals with start times earlier than the selected start get adjusted by adding 24 hours.
    def sort_key(interval_str):
        start_str = interval_str.split("-")[0]
        t = datetime.strptime(start_str, "%H:%M")
        if selected_start_time <= selected_end_time:
            return t
        else:
            # For wrap-around, if the interval's start is before the selected start, add 24 hours.
            if t.time() < datetime.strptime(selected_start_time, "%H:%M").time():
                return t + timedelta(hours=24)
            else:
                return t


    filtered_intraday_df = filtered_intraday_df.sort_values(by="Interval",
                                                            key=lambda col: col.apply(sort_key)).reset_index(drop=True)

    # Pre-populate the first row of the filtered DataFrame with default numeric values.
    if not filtered_intraday_df.empty:
        filtered_intraday_df.loc[0, "Volume"] = 0
        filtered_intraday_df.loc[0, "AHT"] = 300
        filtered_intraday_df.loc[0, "Max Occupancy"] = 0.7
        filtered_intraday_df.loc[0, "SLA Target"] = 0.8
        filtered_intraday_df.loc[0, "Available FTE"] = 0


    # Define calculation functions that handle missing values.
    def intraday_fte(volume, aht, max_occupancy, sla_target):
        if pd.isna(volume) or pd.isna(aht) or pd.isna(max_occupancy) or pd.isna(sla_target):
            return None
        if max_occupancy == 0:
            return 0
        base_fte = (volume * aht) / (1800 * max_occupancy)
        adjustment = 1 / ((1 - sla_target) ** 0.5) if sla_target < 1 else 1
        return round(base_fte * adjustment, 2)


    def calc_ideal_occupancy(volume, aht, available_fte):
        if pd.isna(volume) or pd.isna(aht) or pd.isna(available_fte):
            return None
        if available_fte <= 0:
            return 0
        ideal_occ = (volume * aht) / (available_fte * 1800)
        return round(min(ideal_occ, 1), 2)


    def erlang_c_formula(A, c):
        if c <= A:
            return 1.0
        sum_terms = sum([A ** n / math.factorial(n) for n in range(c)])
        term_c = A ** c / math.factorial(c)
        return (term_c * (c / (c - A))) / (sum_terms + term_c * (c / (c - A)))


    def calc_projected_service_level(volume, aht, available_fte, target_time):
        if pd.isna(volume) or pd.isna(aht) or pd.isna(available_fte):
            return None
        A = (volume * aht) / 1800.0
        c = int(math.ceil(available_fte))
        if c <= A:
            return 0.0
        erlang_c_val = erlang_c_formula(A, c)
        prob_wait = erlang_c_val * math.exp(-(c - A) * target_time / aht)
        service_level = 1 - prob_wait
        return round(service_level, 2)


    # Let the user edit only the filtered rows.
    edited_intraday_df = st.data_editor(filtered_intraday_df, num_rows="dynamic")

    # Update computed columns.
    edited_intraday_df["FTE Required"] = edited_intraday_df.apply(
        lambda row: intraday_fte(row["Volume"], row["AHT"], row["Max Occupancy"], row["SLA Target"]), axis=1)
    edited_intraday_df["Ideal Occupancy"] = edited_intraday_df.apply(
        lambda row: calc_ideal_occupancy(row["Volume"], row["AHT"], row["Available FTE"]), axis=1)
    edited_intraday_df["Projected Service Level"] = edited_intraday_df.apply(
        lambda row: calc_projected_service_level(row["Volume"], row["AHT"], row["Available FTE"], target_time_intraday),
        axis=1)

    st.subheader("📊 Updated Intraday FTE Planning Table")
    st.dataframe(edited_intraday_df)
    st.download_button("Download Intraday FTE Plan", data=edited_intraday_df.to_csv(index=False),
                       file_name="intraday_fte_plan.csv")

    # ---------------- Overall Summary Table ----------------
    summary_df = edited_intraday_df.dropna(
        subset=["Volume", "AHT", "Available FTE", "FTE Required", "Ideal Occupancy", "Projected Service Level"])
    summary_df = summary_df[summary_df["Volume"] > 0]
    total_volume = summary_df["Volume"].sum()
    if total_volume > 0:
        weighted_aht = round((summary_df["Volume"] * summary_df["AHT"]).sum() / total_volume, 2)
        weighted_available_fte = round((summary_df["Volume"] * summary_df["Available FTE"]).sum() / total_volume, 2)
        weighted_fte_required = round((summary_df["Volume"] * summary_df["FTE Required"]).sum() / total_volume, 2)
        weighted_ideal_occ = round((summary_df["Volume"] * summary_df["Ideal Occupancy"]).sum() / total_volume, 2)
        weighted_projected_sla = round(
            (summary_df["Volume"] * summary_df["Projected Service Level"]).sum() / total_volume, 2)
    else:
        weighted_aht = weighted_available_fte = weighted_fte_required = weighted_ideal_occ = weighted_projected_sla = 0

    overall_data = {
        "Metric": ["Total Volume", "AHT", "Available FTE",
                   "FTE Required", "Ideal Occupancy", "Projected Service Level"],
        "Value": [total_volume, weighted_aht, weighted_available_fte, weighted_fte_required, weighted_ideal_occ,
                  weighted_projected_sla]
    }
    overall_df = pd.DataFrame(overall_data)
    overall_df = overall_df[overall_df["Value"] != 0]

    st.subheader("📊 Overall Intraday Summary")
    st.dataframe(overall_df)
    st.download_button("Download Overall Summary", data=overall_df.to_csv(index=False),
                       file_name="overall_intraday_summary.csv")

st.write("---")
st.write("Built with ❤️ by Marian")
