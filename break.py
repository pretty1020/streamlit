import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px

# Title
st.title("Agent Breaks Scheduler")

# User Guide



# Sidebar Inputs
st.sidebar.header("Break Schedule Settings")
break_duration = st.sidebar.number_input("Break Duration (minutes)", min_value=5, value=15)
break_interval = st.sidebar.number_input("Interval Between Breaks (minutes)", min_value=30, value=120)
breaks_per_agent = st.sidebar.number_input("Number of Breaks per Agent", min_value=1, value=2)

st.sidebar.markdown("""
### How to Use the Application:
1. **Input Settings:** Use the sidebar to set the following parameters:
   - Break Duration: Duration of each break in minutes.
   - Interval Between Breaks: Time gap between the start of one break and the start of the next.
   - Number of Breaks per Agent: Number of breaks each agent should receive.
2. **Upload Shift Schedule:** Upload a CSV file with the following columns:
   - Agent
   - Shift Start (HH:MM format)
   - Shift End (HH:MM format)
   If no file is uploaded, sample dummy data will be used.
3. **Generate Break Schedule:** Click the button to create a break schedule.
4. **View & Download Schedule:** The generated schedule will be displayed and can be downloaded as a CSV file.

### Definition of Terms:
- **Break Duration:** The amount of time an agent is on break.
- **Interval Between Breaks:** The time between the start of one break and the next.
- **Number of Breaks per Agent:** Total breaks each agent is allowed during their shift.
- **Shift Start & End:** The working hours of an agent.
""")
st.sidebar.markdown("[For any concerns or issues,feel free to reach out to Marian via Linkedin](https://www.linkedin.com/in/marian1020/)")

# File uploader for shift schedule
uploaded_file = st.file_uploader("Upload Shift Schedule (CSV with columns: Agent, Shift Start, Shift End)", type=["csv"])

# Dummy Data
dummy_data = pd.DataFrame({
    'Agent': [f'Agent {i}' for i in range(1, 6)],
    'Shift Start': ['08:00', '09:00', '10:00', '11:00', '12:00'],
    'Shift End': ['16:00', '17:00', '18:00', '19:00', '20:00']
})

shift_df = dummy_data
if uploaded_file is not None:
    shift_df = pd.read_csv(uploaded_file)

st.write("Sample Shift Schedule:")
st.dataframe(shift_df)

def parse_time(time_str):
    return datetime.strptime(time_str, "%H:%M").time()

if st.button("Generate Break Schedule"):
    schedule_data = []
    for _, row in shift_df.iterrows():
        agent = row['Agent']
        shift_start = parse_time(row['Shift Start'])
        shift_end = parse_time(row['Shift End'])

        shift_start_dt = datetime.combine(datetime.today(), shift_start)
        shift_end_dt = datetime.combine(datetime.today(), shift_end)

        current_time = shift_start_dt
        breaks_added = 0
        break_times = []

        while current_time < shift_end_dt and breaks_added < breaks_per_agent:
            if current_time + timedelta(minutes=break_duration) <= shift_end_dt:
                break_times.append((current_time.strftime("%H:%M"), (current_time + timedelta(minutes=break_duration)).strftime("%H:%M")))
                breaks_added += 1
            current_time += timedelta(minutes=break_interval)

        schedule_data.append({
            'Agent': agent,
            'Shift Start': shift_start.strftime("%H:%M"),
            'Shift End': shift_end.strftime("%H:%M"),
            'Break1 Start': break_times[0][0] if len(break_times) > 0 else '',
            'Break1 End': break_times[0][1] if len(break_times) > 0 else '',
            'Break2 Start': break_times[1][0] if len(break_times) > 1 else '',
            'Break2 End': break_times[1][1] if len(break_times) > 1 else ''
        })

    schedule_df = pd.DataFrame(schedule_data)
    st.header("Generated Break Schedule")
    st.dataframe(schedule_df)

    # Download Schedule as CSV
    csv = schedule_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Break Schedule as CSV", csv, "break_schedule.csv", "text/csv")

    st.success("Break Schedule Generated Successfully!")
