import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt
import networkx as nx

# Set Streamlit app title
st.title("📞 Monte Carlo Simulation for Outbound Sales Channel")

# User Guide & Definitions
st.markdown("""
## 📌 **User Guide & Definition of Terms**
### **1. Parameters:**
- **Connect Rate (%)** - Probability that a call gets connected to a recipient.
- **Contacted Rate (%)** - Probability that a connected call leads to an actual conversation.
- **Sale Rate (%)** - Probability that a contacted customer results in a successful sale.
- **AHT (Average Handling Time in seconds)** - Time spent on each stage.

### **2. How It Works:**
- This simulation mimics real-world outbound call campaigns.
- Calls flow through different stages based on probabilities set by the user.
- The system runs **multiple iterations (Monte Carlo method)** to estimate average results.

### **3. How to Use:**
1. Adjust the sliders for **Connect Rate, Contacted Rate, and Sale Rate**.
2. Enter **AHT values (in seconds)** for each stage.
3. Set the **Target Number of Sales** to estimate required calls.
4. Click **"Simulate"** to run the Monte Carlo analysis.
5. View:
   - **Total Time Spent on Achieved Sales (Minutes)**
   - **Predicted Number of Calls to Achieve Target Sales**
   - **Time Spent in Minutes for Predicted Calls**
   - **Graphical Flow of Calls**
""")

# User Input: Sliders for probabilities
st.sidebar.header("**Historical Data**")
connect_rate = st.sidebar.slider("📶 Connect Rate (%)", 10, 100, 70)
contacted_rate = st.sidebar.slider("📞 Contacted Rate (%)", 10, 100, 50)
sale_rate = st.sidebar.slider("💰 Sale Rate (%)", 10, 100, 30)
target_sales = st.sidebar.number_input("🎯 Target Number of Sales", min_value=1, value=100)
iterations = st.sidebar.slider("🔄 Monte Carlo Simulations", 100, 5000, 1000)

# User Input: AHT per stage
st.sidebar.header("⏳ **Average Handling Time (AHT) per Stage (seconds)**")
aht_values = {
    "Not Connected": st.sidebar.number_input("AHT for Not Connected", min_value=1, value=45),
    "Connected": st.sidebar.number_input("AHT for Connected", min_value=1, value=80),
    "Not Contacted": st.sidebar.number_input("AHT for Not Contacted", min_value=1, value=120),
    "Contacted": st.sidebar.number_input("AHT for Contacted", min_value=1, value=180),
    "Sale": st.sidebar.number_input("AHT for Sale", min_value=1, value=300),
    "Not Sale": st.sidebar.number_input("AHT for Not Sale", min_value=1, value=280),
}
   st.sidebar.markdown("[For any concerns or issues,feel free to reach out to Marian via Linkedin](https://www.linkedin.com/in/marian1020/)")

# Define call flow structure
call_flow = {
    "Start": ["Not Connected", "Connected"],
    "Not Connected": ["End Call"],
    "Connected": ["Not Contacted", "Contacted"],
    "Not Contacted": ["End Call"],
    "Contacted": ["Sale", "Not Sale"],
    "Sale": ["End Call"],
    "Not Sale": ["End Call"],
}

def simulate_calls():
    results = {key: 0 for key in call_flow.keys()}
    results["Start"] = 1000
    results["End Call"] = 0

    for _ in range(results["Start"]):
        if np.random.rand() < (1 - connect_rate / 100):
            results["Not Connected"] += 1
            results["End Call"] += 1
        else:
            results["Connected"] += 1
            if np.random.rand() < (1 - contacted_rate / 100):
                results["Not Contacted"] += 1
                results["End Call"] += 1
            else:
                results["Contacted"] += 1
                if np.random.rand() < (1 - sale_rate / 100):
                    results["Not Sale"] += 1
                else:
                    results["Sale"] += 1
                results["End Call"] += 1

    return results

def monte_carlo_simulation(iterations):
    total_results = {key: [] for key in call_flow.keys()}
    total_results["End Call"] = []

    for _ in range(iterations):
        results = simulate_calls()
        for key in results:
            total_results[key].append(results[key])

    avg_results = {key: np.mean(values) for key, values in total_results.items()}
    return avg_results

def predict_call_volume(target_sales, results):
    if results["Sale"] == 0:
        return "Insufficient data to predict"
    required_calls = (results["Start"] / results["Sale"]) * target_sales
    return int(np.ceil(required_calls))

def calculate_total_time_spent(sales, calls):
    total_time_sales = (sales * aht_values["Sale"]) / 60  # Convert to minutes
    total_time_calls = (calls * aht_values["Sale"]) / 60  # Convert to minutes
    return round(total_time_sales, 2), round(total_time_calls, 2)

def plot_interactive_simulation(results):
    G = nx.DiGraph()
    for source, targets in call_flow.items():
        for target in targets:
            G.add_edge(source, target)

    pos = nx.spring_layout(G, seed=42)
    base_color = "skyblue"
    highlight_color = "orange"

    fig, ax = plt.subplots(figsize=(8, 6))
    node_colors = [base_color] * len(G.nodes())
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=2000, edge_color="gray", font_size=10,
            font_weight="bold", ax=ax)
    plot = st.pyplot(fig)

    for stage in results.keys():
        time.sleep(1)
        fig, ax = plt.subplots(figsize=(8, 6))
        node_colors = [highlight_color if node == stage else base_color for node in G.nodes()]
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=2000, edge_color="gray", font_size=10,
                font_weight="bold", ax=ax)
        plot.pyplot(fig)

        if stage == "End Call":
            break

if st.button("Simulate"):
    st.subheader("🚀 Running Simulation...")
    time.sleep(2)
    avg_results = monte_carlo_simulation(iterations)
    required_calls = predict_call_volume(target_sales, avg_results)
    total_time_spent_sales, total_time_spent_calls = calculate_total_time_spent(avg_results["Sale"], required_calls)
    st.success("✅ Simulation Complete!")



    # Call Flow Visualization
    st.subheader("🔗 Call Flow Visualization")
    plot_interactive_simulation(avg_results)

    # Display Predicted Number of Calls to Achieve Target Sales
    st.subheader("📞 Predicted Number of Calls to Achieve Target Sales")
    st.write(f"Estimated calls required to achieve **{target_sales}** sales: **{required_calls} calls**")

    # Display Time Spent in Minutes for Predicted Calls
    st.subheader("⏳ Time Spent for Predicted Calls to Achieve Target Sales (Minutes)")
    st.write(
        f"Estimated time required for **{required_calls}** calls to reach **{target_sales}** sales: **{total_time_spent_calls} minutes**")
