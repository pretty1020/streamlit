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
st.title("🌟 Customer Sentiment Analysis Tool🌟")

# Sidebar - File Upload
st.sidebar.header("Upload Customer Feedback Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'Comment' not in df.columns:
        st.error("CSV must contain a 'Comment' column!")
    else:
        df = process_data(df)

st.sidebar.markdown("[For any concerns or issues,feel free to reach out to Marian via Linkedin](https://www.linkedin.com/in/marian1020/)")

# Tabs
tabs = st.tabs(["✨ Sentiment Analysis", "📖 User Guide & Definitions"])

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

    # 📊 ✨ Sentiment Distribution ✨
    st.subheader("🌟 Sentiment Distribution 🌟")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.set_style("darkgrid")
    sns.countplot(x=df['Sentiment'], palette='coolwarm', ax=ax, edgecolor='gold', linewidth=3)

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
    st.markdown(
        "**Interpretation:** A higher count of positive comments indicates customer satisfaction, while an increase in negative comments highlights areas needing improvement.")

    # 📊 ✨ Polarity & Subjectivity Distributions ✨
    st.subheader("🌟 Polarity & Subjectivity Distributions 🌟")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(df['Polarity'], bins=20, kde=True, ax=axes[0], color="cyan")
    axes[0].set_title("Polarity Distribution", color='white')
    axes[0].set_xlabel("Polarity (-1 to 1)", color='white')
    axes[0].set_ylabel("Frequency", color='white')
    axes[0].set_facecolor("black")

    sns.histplot(df['Subjectivity'], bins=20, kde=True, ax=axes[1], color="magenta")
    axes[1].set_title("Subjectivity Distribution", color='white')
    axes[1].set_xlabel("Subjectivity (0 to 1)", color='white')
    axes[1].set_ylabel("Frequency", color='white')
    axes[1].set_facecolor("black")

    plt.setp(axes, xticks=[], facecolor='black')
    st.pyplot(fig)
    st.markdown(
        "**Interpretation:** A higher subjectivity score suggests more opinionated feedback, while a balanced polarity distribution indicates a mix of positive and negative sentiments.")

    # 📊 ✨ Sentiment by Comment Length ✨
    st.subheader("🌟 Sentiment by Comment Length 🌟")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x="Sentiment", y="Comment_Length", data=df, ax=ax, palette="coolwarm")
    ax.set_facecolor("black")
    st.pyplot(fig)
    st.markdown(
        "**Interpretation:** Longer negative comments may indicate detailed complaints, whereas shorter positive comments suggest quick, satisfied feedback.")

    # 📊 ✨ Correlation Analysis ✨
    st.subheader("🌟 Sentiment Correlation Analysis 🌟")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df[['Polarity', 'Subjectivity', 'Comment_Length']].corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_facecolor("black")
    st.pyplot(fig)
    st.markdown(
        "**Interpretation:** A strong correlation between polarity and subjectivity suggests that highly opinionated comments tend to be either very positive or very negative.")

with tabs[1]:
    st.subheader("📖 User Guide")
    st.markdown("""
    **How to Use this App:**
    1. Upload a CSV file with a **'Comment'** column.
    2. The system will analyze customer feedback using **sentiment analysis**.
    3. The dashboard displays:
       - **Sentiment Summary Table**
       - **Sentiment Distribution Chart with Glow**
       - **Polarity & Subjectivity Distributions**
       - **Sentiment by Comment Length**
       - **Sentiment Correlation Analysis**
    """)

    st.subheader("📌 Definition of Terms")
    st.markdown("""
    - **Polarity**: Measures the positivity or negativity of a comment, ranging from -1 (negative) to +1 (positive).
    - **Subjectivity**: Reflects how opinion-based the feedback is (0 = fact, 1 = opinion).
    - **Sentiment Categories**:
      - **Positive**: Feedback with a polarity score > 0.
      - **Negative**: Feedback with a polarity score < 0.
      - **Neutral**: Feedback with a polarity of 0.
    - **Comment Length**: The total number of characters in a comment, which can indicate detailed feedback vs. brief remarks.
    - **Correlation Analysis**: Measures how strongly two variables are related. For example, it helps us see if longer comments tend to be more positive or negative.
    """)
