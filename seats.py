import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="📊 Seat Capacity Dashboard", layout="wide")
st.title("📊 DCX Seat Capacity Dashboard")

uploaded_file = st.file_uploader("📁 Upload Excel file", type=["xlsx"])

if uploaded_file:
    df_raw = pd.read_excel(uploaded_file, sheet_name='PH')
    df = df_raw.iloc[2:].copy()
    df.columns = [
        "Index", "Campaign", "Total Agent HC", "Billable FTE",
        "Total Alloc Agent Seats", "Site", "WFH Agents", "Comments", "Empty1",
        "Total Seat Alloc", "Support", "Onsite Agents", "Empty2", "Updated By"
    ]

    df = df[[
        "Campaign", "Total Agent HC", "WFH Agents", "Onsite Agents", "Support", "Site",
        "Total Alloc Agent Seats", "Total Seat Alloc"
    ]]

    numeric_cols = ["Total Agent HC", "WFH Agents", "Onsite Agents", "Support", "Total Alloc Agent Seats", "Total Seat Alloc"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df[df["Campaign"].notna()]

    # Derived columns
    df["Seat Requirement"] = df["Total Agent HC"]
    df["Variance"] = df["Total Alloc Agent Seats"] - df["Seat Requirement"]

    st.subheader("📍 Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Headcount", int(df["Total Agent HC"].sum()))
    col2.metric("WFH Agents", int(df["WFH Agents"].sum()))
    col3.metric("Onsite Agents", int(df["Onsite Agents"].sum()))
    col4.metric("Total Seats Allocated", int(df["Total Alloc Agent Seats"].sum()))

    st.subheader("📌 Detailed Table")
    st.dataframe(df.reset_index(drop=True), use_container_width=True)


    # Total Agent Headcount per Campaign
    st.markdown("### 🎯 Total Agent HC by Campaign and Site")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="Campaign", y="Total Agent HC", hue="Site", palette="Set2", ax=ax1)
    ax1.set_title("Total Agent Headcount per Campaign")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    st.pyplot(fig1)

    # WFH vs Onsite
    st.markdown("### 🏠 WFH vs Onsite Agents")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    df.set_index("Campaign")[["WFH Agents", "Onsite Agents"]].plot(kind="bar", stacked=True, colormap="coolwarm", ax=ax2)
    ax2.set_ylabel("Agent Count")
    ax2.set_title("WFH vs Onsite per Campaign")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    st.pyplot(fig2)

    # Seat Requirements vs Allocated
    st.markdown("### 🪑 Seat Allocation vs Requirement")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    df.set_index("Campaign")[["Total Alloc Agent Seats", "Seat Requirement"]].plot(kind="bar", colormap="viridis", ax=ax3)
    ax3.set_title("Seats Allocated vs Required")
    ax3.set_ylabel("Seats")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
    st.pyplot(fig3)

    st.caption("Contact Marian for Modification")
else:
    st.info("Upload the Excel file to generate the dashboard.")
