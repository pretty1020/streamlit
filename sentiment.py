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
st.set_page_config(page_title="Advanced Sentiment Analysis", layout="wide")
st.title("📊 Advanced Customer Sentiment Analysis")

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
tabs = st.tabs(["📊 Sentiment Analysis", "📖 User Guide & Recommendations"])

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

    # 📊 Sentiment Distribution
    st.subheader("📌 Sentiment Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=df['Sentiment'], palette='coolwarm', ax=ax, edgecolor='gold', linewidth=2)
    st.pyplot(fig)

    # Interpretation
    st.markdown("""
    **📌 Interpretation:**  
    - A **higher percentage of positive comments** suggests strong customer satisfaction.  
    - A **higher percentage of negative comments** signals dissatisfaction, requiring urgent action.  
    - **Neutral comments** indicate customers who are neither happy nor unhappy.  
    """)

    # 📈 Sentiment Trends by Customer
    if "Customer" in df.columns:
        st.subheader("📈 Sentiment Trends by Customer")
        fig, ax = plt.subplots()
        sns.boxplot(x="Sentiment", y="Polarity", data=df, ax=ax, palette="coolwarm")
        st.pyplot(fig)

        st.markdown("""
        **📌 Interpretation:**  
        - A **wide spread of polarity values** for a customer may indicate inconsistent experiences.  
        - **Highly positive or negative scores** suggest clear customer satisfaction or frustration.  
        """)

    # 📊 Polarity & Subjectivity Distributions
    st.subheader("📌 Polarity & Subjectivity Distributions")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(df['Polarity'], bins=20, kde=True, ax=axes[0], color="blue")
    axes[0].set_title("Polarity Distribution")
    axes[0].set_xlabel("Polarity (-1 to 1)")

    sns.histplot(df['Subjectivity'], bins=20, kde=True, ax=axes[1], color="green")
    axes[1].set_title("Subjectivity Distribution")
    axes[1].set_xlabel("Subjectivity (0 to 1)")

    st.pyplot(fig)

    st.markdown("""
    **📌 Interpretation:**  
    - A **high subjectivity score** means customers share more opinions than facts.  
    - A **wide spread in polarity values** suggests mixed customer experiences.  
    """)

    # 📊 Sentiment by Comment Length
    st.subheader("📊 Sentiment by Comment Length")
    fig, ax = plt.subplots()
    sns.boxplot(x="Sentiment", y="Comment_Length", data=df, ax=ax, palette="coolwarm")
    st.pyplot(fig)

    st.markdown("""
    **📌 Interpretation:**  
    - **Longer negative comments** may indicate customers expressing detailed dissatisfaction.  
    - **Short positive comments** suggest quick but strong satisfaction.  
    """)

    # 📊 Correlation Analysis
    st.subheader("📊 Sentiment Correlation Analysis")
    fig, ax = plt.subplots()
    sns.heatmap(df[['Polarity', 'Subjectivity', 'Comment_Length']].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.markdown("""
    **📌 Interpretation:**  
    - **Strong correlation between polarity & subjectivity** means customers provide strong opinions.  
    - **Negative correlation with comment length** may suggest longer complaints.  
    """)

    # 🔍 Most Common Words in Positive & Negative Feedback
    def get_common_words(sentiment, num_words=10):
        words = " ".join(df[df['Sentiment'] == sentiment]['Comment']).lower()
        words = re.findall(r'\b\w+\b', words)
        return Counter(words).most_common(num_words)

    st.subheader("🔍 Most Common Words in Positive & Negative Feedback")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Most Frequent Words in Positive Comments**")
        common_positive = get_common_words("Positive")
        st.write(pd.DataFrame(common_positive, columns=["Word", "Frequency"]))

    with col2:
        st.write("**Most Frequent Words in Negative Comments**")
        common_negative = get_common_words("Negative")
        st.write(pd.DataFrame(common_negative, columns=["Word", "Frequency"]))

with tabs[1]:
    st.subheader("📖 Summary of Findings & Actionable Recommendations")
    
    st.markdown("""
    ### **📊 Summary of Findings:**
    - **Overall Sentiment:**  
      - Majority of comments are **[Positive/Negative/Neutral]**  
      - This suggests **[customer satisfaction/dissatisfaction/mixed opinions]**  
    - **Key Trends:**  
      - **Frequent keywords in negative comments**: [Common words]  
      - **Frequent keywords in positive comments**: [Common words]  
      - **Sentiment trends over time suggest**: [stable, fluctuating, declining, improving sentiment]  

    ### **🚀 Actionable Recommendations:**
    **For Leadership Teams:**
    - If **negative sentiment is high**, consider **service quality improvements & proactive issue resolution**.  
    - If **neutral comments dominate**, identify factors that drive stronger satisfaction.  

    **For Customer Service Managers:**
    - **Train support teams** to improve areas mentioned in frequent negative comments.  
    - **Monitor long complaints** for deeper dissatisfaction insights.  

    **For Product & Marketing Teams:**
    - **Leverage common positive keywords** in advertising & testimonials.  
    - Address **frequent negative concerns** to improve product-market fit.  
    """)

