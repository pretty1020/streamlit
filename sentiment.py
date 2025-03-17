import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from collections import Counter
import re

# Function to analyze sentiment
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
    return polarity, subjectivity, sentiment

# Function to process data
def process_data(df):
    df['Polarity'], df['Subjectivity'], df['Sentiment'] = zip(*df['Comment'].apply(analyze_sentiment))
    df['Comment_Length'] = df['Comment'].apply(len)  # Add comment length analysis
    return df

# Dummy Data
dummy_data = {
    "Customer": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "Comment": [
        "I love the service, it was fantastic!",
        "The support team was rude and unhelpful.",
        "Everything was fine, nothing special.",
        "I had a terrible experience, very dissatisfied!",
        "The call was smooth and the agent was very polite."
    ]
}
df = pd.DataFrame(dummy_data)
df = process_data(df)

# Streamlit UI
st.set_page_config(page_title="✨ Advanced Sentiment Analysis ✨", layout="wide")
st.title("🌟 Advanced Customer Sentiment Analysis 🌟")

# Sidebar - File Upload
st.sidebar.header("Upload Customer Feedback Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'Comment' not in df.columns:
        st.error("CSV must contain a 'Comment' column!")
    else:
        df = process_data(df)

# Tabs
tabs = st.tabs(["✨ Sentiment Analysis", "📖 Summary & Recommendations"])

with tabs[0]:
    st.subheader("📊 Sentiment Summary")
    st.dataframe(df)

    # Download button for sentiment summary
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Sentiment Summary",
        data=csv,
        file_name="sentiment_summary.csv",
        mime='text/csv'
    )

    # 📊 ✨ Glowy Sentiment Distribution ✨
    st.subheader("🌟 Sentiment Distribution (Glowy Chart) 🌟")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.set_style("darkgrid")
    sns.countplot(x=df['Sentiment'], palette='coolwarm', ax=ax, edgecolor='gold', linewidth=3)

    # Sparkle effect
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")
    ax.spines['bottom'].set_color('#FFD700')
    ax.spines['top'].set_color('#FFD700')
    ax.spines['right'].set_color('#FFD700')
    ax.spines['left'].set_color('#FFD700')
    plt.xticks(color='white')
    plt.yticks(color='white')
    plt.setp(ax.patches, linewidth=2, edgecolor='gold')

    st.pyplot(fig)

    # Interpretation
    st.markdown("""
    **📌 Interpretation:**  
    - **More positive comments** indicate customer satisfaction.  
    - **A high number of negative comments** may require urgent attention.  
    - **Neutral responses** suggest room for engagement improvement.  
    """)

    # 📊 ✨ Sentiment Trends Over Time ✨
    if "Date" in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values("Date", inplace=True)
        df['Rolling Sentiment'] = df['Polarity'].rolling(window=5, min_periods=1).mean()

        st.subheader("🌟 Sentiment Trends Over Time (Glowy Line Chart) 🌟")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=df["Date"], y=df["Rolling Sentiment"], color="cyan", marker="o", ax=ax)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle="--", linewidth=0.5, color="white")
        st.pyplot(fig)

        st.markdown("""
        **📌 Interpretation:**  
        - **Upward trends** suggest increasing customer satisfaction.  
        - **Downward trends** indicate rising concerns and dissatisfaction.  
        """)

    # 📊 ✨ Sentiment by Comment Length ✨
    st.subheader("🌟 Sentiment by Comment Length (Glowy Boxplot) 🌟")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x="Sentiment", y="Comment_Length", data=df, ax=ax, palette="coolwarm")
    st.pyplot(fig)

    st.markdown("""
    **📌 Interpretation:**  
    - **Longer negative comments** suggest detailed complaints.  
    - **Short positive comments** show quick expressions of satisfaction.  
    """)

with tabs[1]:
    st.subheader("📖 Summary & Actionable Recommendations")

    st.markdown("""
    ### **📊 Summary of Findings:**
    - **Overall Sentiment:**  
      - **{positive_count}% positive**, indicating high customer satisfaction.  
      - **{neutral_count}% neutral**, meaning some customers have mixed experiences.  
      - **{negative_count}% negative**, showing areas needing improvement.  
    - **Key Trends Identified:**  
      - Customers frequently mention **"{top_positive_words}"** in positive reviews.  
      - Common complaints include **"{top_negative_words}"**, requiring immediate attention.  
      - **Sentiment has been {sentiment_trend} over time**, suggesting **{trend_conclusion}**.  

    ### **🚀 Actionable Recommendations for Customer Service Teams:**
    **1️⃣ Improve Response Time & Interaction Quality**  
    ✅ Reduce **first response time (FRT)** and **average handling time (AHT)**.  
    ✅ Offer **real-time solutions & knowledge base integration**.  

    **2️⃣ Train Agents on Soft Skills & Empathy**  
    ✅ Conduct **role-play training** based on negative feedback themes.  
    ✅ Reward **top-performing agents** for outstanding customer interactions.  

    **3️⃣ Leverage Positive Feedback for Customer Engagement**  
    ✅ Use **common positive keywords** in testimonials & marketing.  
    ✅ Engage with happy customers to build loyalty.  

    **4️⃣ Address Negative Sentiment with Process Improvements**  
    ✅ If complaints are about **slow service**, optimize agent workflow.  
    ✅ If **rudeness** is mentioned, reinforce professionalism training.  
    ✅ Follow up with **dissatisfied customers** for resolution.  

    ### **🔮 Next Steps**
    ✅ Implement **Sentiment-Based Training & Quality Control**.  
    ✅ Use **AI-driven analytics** to predict customer satisfaction trends.  
    ✅ Track sentiment trends **over time** to measure improvement.  
    """)

