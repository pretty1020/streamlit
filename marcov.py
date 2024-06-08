import streamlit as st
import pandas as pd
import plotly.express as px

# Initial Data
positions = ["Tellers", "Customer Service Representatives", "Teller Supervisors", "Assistant Branch Managers",
             "Branch Managers"]
current_workforce_initial = [600, 300, 200, 75, 100]
transition_probabilities_initial = {
    "Tellers": [0.56, 0.11, 0.03, 0.00, 0.00],
    "Customer Service Representatives": [0.00, 0.46, 0.08, 0.05, 0.00],
    "Teller Supervisors": [0.00, 0.00, 0.63, 0.12, 0.0],
    "Assistant Branch Managers": [0.00, 0.00, 0.00, 0.52, 0.08],
    "Branch Managers": [0.00, 0.00, 0.00, 0.00, 0.70],
}

# Function to calculate next year's projection and gap analysis
def forecast_and_gap_analysis(current_workforce, transition_probabilities):
    projections = []
    for i, position in enumerate(positions):
        stay = current_workforce[i] * transition_probabilities[position][i]
        move_up = sum(
            current_workforce[j] * transition_probabilities[positions[j]][i] for j in range(len(positions)) if j != i)
        projected = stay + move_up
        projections.append(projected)

    next_year_projections = []
    for i, position in enumerate(positions):
        projected = current_workforce_initial[i] * transition_probabilities_initial[position][i]
        next_year_projections.append(projected)

    gap_analysis = {
        "Position": positions,
        "Current Workforce": current_workforce,
        "Next Year Projected": next_year_projections,
        "Year-End Total": projections,
        "External Hires Needed": [current_workforce[i] - projections[i] for i in range(len(current_workforce))]
    }
    return pd.DataFrame(gap_analysis)


# Streamlit App
st.title("Workforce Forecasting App")
st.write("Forecasting the workforce requirements for next year based on transition probabilities.")

# Display Initial Data and Forecast
st.subheader("Initial Data and Forecast")
df_initial = forecast_and_gap_analysis(current_workforce_initial, transition_probabilities_initial)
st.write(df_initial)

# Display Transition Matrix
st.subheader("Transition Probability Matrix")
transition_matrix = pd.DataFrame(transition_probabilities_initial, index=positions)
st.write(transition_matrix)

# Visualization - Current vs. Projected Workforce
df_visualization = pd.DataFrame({
    "Position": positions,
    "Current Workforce": current_workforce_initial,
    "Projected Workforce": df_initial["Next Year Projected"]
})

fig = px.bar(df_visualization, x="Position", y=["Current Workforce", "Projected Workforce"],
             title="Current vs. Projected Workforce",
             labels={"value": "Number of Employees", "variable": "Workforce"},
             barmode="group")
st.plotly_chart(fig)

# User input for current workforce
st.header("Calculator:")
st.subheader("Input Current Workforce")
current_workforce = []
for position in positions:
    current_workforce.append(st.number_input(f"Current {position}", min_value=0,
                                             value=current_workforce_initial[positions.index(position)]))

# User input for transition probabilities
st.subheader("Input Transition Probabilities")
transition_probabilities = {}
for position in positions:
    probabilities = []
    for to_position in positions + ["Exit"]:
        default_value = transition_probabilities_initial[position][
            positions.index(to_position)] if to_position in positions else \
            transition_probabilities_initial[position][-1]
        prob = st.number_input(f"Probability of {position} moving to {to_position}", min_value=0.0, max_value=1.0,
                               value=default_value)
        probabilities.append(prob)
    transition_probabilities[position] = probabilities

# Display the formulas used for the calculations
st.subheader("Formulas Used")
st.markdown("""
- **Stay in Position**: The number of employees who remain in their current position.
  - Formula: `current_workforce[i] * transition_probabilities[position][i]`
- **Move to Another Position**: The number of employees who move to another position.
  - Formula: `sum(current_workforce[j] * transition_probabilities[positions[j]][i] for all j ≠ i)`
- **Next Year Projected**: The total number of employees projected in each position for the next year.
  - Formula: `Stay in Position + Move to Another Position`
- **External Hires Needed**: The number of external hires needed to meet the workforce requirements.
  - Formula: `Current Workforce - Next Year Projected`
""")

# Forecast and Gap Analysis based on user input
st.subheader("User Input Forecast and Gap Analysis")
df = forecast_and_gap_analysis(current_workforce, transition_probabilities)
st.write(df)

# External Hires Needed
st.subheader("External Hires Needed")
for i, position in enumerate(positions):
    if df["External Hires Needed"][i] > 0:
        st.write(f"{position}: {df['External Hires Needed'][i]} hires needed")
    else:
        st.write(f"{position}: No external hires needed (surplus of {abs(df['External Hires Needed'][i])})")
