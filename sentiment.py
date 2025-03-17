import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
import re
from collections import Counter


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
st.set_page_config(page_title="Customer Sentiment Analysis", layout="wide")
st.title("📊 Customer Sentiment Analysis")

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
tabs = st.tabs(["📊 Sentiment Analysis", "✨ Word Cloud & Advanced Analysis", "📖 User Guide & Definitions"])

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

    # 📌 Glowy Sentiment Distribution
    st.subheader("✨ Sentiment Distribution")
    fig, ax = plt.subplots()
    sns.set_style("darkgrid")
    sns.countplot(x=df['Sentiment'], palette='coolwarm', ax=ax, edgecolor='gold', linewidth=3)

    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")

    # Add glow effect to spines
    ax.spines['bottom'].set_color('#FFD700')
    ax.spines['top'].set_color('#FFD700')
    ax.spines['right'].set_color('#FFD700')
    ax.spines['left'].set_color('#FFD700')

    plt.xticks(color='white')
    plt.yticks(color='white')
    plt.setp(ax.patches, linewidth=2, edgecolor='gold')

    st.pyplot(fig)

    # 📌 Interpretation of results
    st.subheader("📌 Interpretation of Results")
    positive_count = df[df['Sentiment'] == 'Positive'].shape[0]
    neutral_count = df[df['Sentiment'] == 'Neutral'].shape[0]
    negative_count = df[df['Sentiment'] == 'Negative'].shape[0]
    total = df.shape[0]

    st.markdown(f"""
    - **Positive Comments**: {positive_count} ({(positive_count / total) * 100:.2f}%) of the total.
    - **Neutral Comments**: {neutral_count} ({(neutral_count / total) * 100:.2f}%) of the total.
    - **Negative Comments**: {negative_count} ({(negative_count / total) * 100:.2f}%) of the total.

    Based on the data, the overall sentiment suggests that the majority of customer feedback is 
    {'positive' if positive_count > negative_count else 'negative' if negative_count > positive_count else 'neutral'}.
    """)

# Advanced Analysis Tab
with tabs[1]:
    st.subheader("✨ Word Cloud")


    def generate_wordcloud(text):
        """Generate a simple colorful Word Cloud."""
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='black',
            colormap='cool',
            max_words=100
        ).generate(text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)


    # Generate Word Cloud if data exists
    if not df.empty:
        all_text = " ".join(df['Comment'].dropna())
        generate_wordcloud(all_text)
    else:
        st.warning("No comments available for Word Cloud generation.")

    # 🔍 Most Common Words in Sentiments
    st.subheader("🔍 Most Common Words in Positive & Negative Feedback")


    def get_common_words(sentiment, num_words=10):
        words = " ".join(df[df['Sentiment'] == sentiment]['Comment']).lower()
        words = re.findall(r'\b\w+\b', words)
        return Counter(words).most_common(num_words)


    col1, col2 = st.columns(2)
    with col1:
        st.write("**Most Frequent Words in Positive Comments**")
        common_positive = get_common_words("Positive")
        st.write(pd.DataFrame(common_positive, columns=["Word", "Frequency"]))

    with col2:
        st.write("**Most Frequent Words in Negative Comments**")
        common_negative = get_common_words("Negative")
        st.write(pd.DataFrame(common_negative, columns=["Word", "Frequency"]))

# 📖 User Guide Tab
with tabs[2]:
    st.subheader("📖 User Guide")
    st.markdown("""
    **How to Use this App:**
    1. Upload a CSV file with a **'Comment'** column.
    2. The system will analyze customer feedback using **sentiment analysis**.
    3. The dashboard displays:
       - **Sentiment Summary Table**
       - **Sentiment Distribution Chart**
       - **Sparkle-Style Word Cloud**
       - **Most Frequent Words in Positive & Negative Feedback**
    """)

    st.subheader("📌 Definitions")
    st.markdown("""
    - **Polarity**: Sentiment score ranging from -1 (negative) to +1 (positive).
    - **Subjectivity**: How opinion-based the feedback is (0 = fact, 1 = opinion).
    - **Sentiment Categories**:
      - **Positive**: Customer feedback with a polarity score > 0.
      - **Negative**: Customer feedback with a polarity score < 0.
      - **Neutral**: Feedback with a polarity of 0.
    """)

