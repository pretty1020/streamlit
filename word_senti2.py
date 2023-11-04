import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import numpy as np
import base64
import pandas as pd


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    color: gold; /* Change font color to purple */
    background-attachment: local;
    background-position: auto;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


set_background('wfmcalc3.png')

# Define the Streamlit app title and description
st.title("Word Cloud and Sentiment Analysis")
st.write("This app allows you to analyze text data using Word Cloud and Sentiment Analysis.")

# Define user instructions and definitions
st.header("User Guide")
st.markdown("1. Paste Text: You can paste your text data into the text box below.")
st.markdown("2. Upload Data: Alternatively, you can upload a text file.")
st.markdown("3. Choose Analysis:")
st.markdown("   - Select 'Generate Word Cloud' to create a word cloud based on the text data.")
st.markdown("   - Select 'Sentiment Analysis' to get the sentiment of the text (positive, negative, or neutral).")
st.markdown("4. Click 'Analyze' to perform the selected analysis.")

# Text input
st.subheader("Input Text Data")
input_text = st.text_area("Paste your text here:")

# File upload
st.subheader("Upload Text Data")
uploaded_file = st.file_uploader("Choose a file...", type=["txt"])

# Choose analysis
analysis_option = st.selectbox("Choose Analysis", ["Generate Word Cloud", "Sentiment Analysis"])


# Function to generate word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)


# Function to perform sentiment analysis and create colorful visualization
def perform_sentiment_analysis(text):
    rows = text.split('\n')
    sentiment_results = []
    for i, row in enumerate(rows):
        blob = TextBlob(row)
        sentiment = blob.sentiment.polarity
        if sentiment > 0:
            color = "green"
            emoji = "😃"
            sentiment_label = "Positive"
        elif sentiment < 0:
            color = "red"
            emoji = "😞"
            sentiment_label = "Negative"
        else:
            color = "yellow"
            emoji = "😐"
            sentiment_label = "Neutral"

        sentiment_results.append([i + 1, sentiment, sentiment_label, emoji, color])

    return pd.DataFrame(sentiment_results,
                        columns=["Row", "Polarity", "Sentiment", "Sentiment Emoji", "Sentiment Color"])


# Analyze button
if st.button("Analyze"):
    if input_text or uploaded_file:
        if analysis_option == "Generate Word Cloud":
            if input_text:
                generate_wordcloud(input_text)
            else:
                with uploaded_file:
                    text = uploaded_file.read().decode("utf-8")
                    generate_wordcloud(text)
        elif analysis_option == "Sentiment Analysis":
            if input_text:
                st.subheader("Sentiment Analysis Result")
                sentiment_results = perform_sentiment_analysis(input_text)
                st.write(sentiment_results)

                # Visualize the count and frequency of sentiment results
                sentiment_counts = sentiment_results["Sentiment"].value_counts()
                st.subheader("Sentiment Distribution")
                st.bar_chart(sentiment_counts)
            else:
                with uploaded_file:
                    text = uploaded_file.read().decode("utf-8")
                    st.subheader("Sentiment Analysis Result")
                    sentiment_results = perform_sentiment_analysis(text)
                    st.write(sentiment_results)

                    # Visualize the count and frequency of sentiment results
                    sentiment_counts = sentiment_results["Sentiment"].value_counts()
                    st.subheader("Sentiment Distribution")
                    st.bar_chart(sentiment_counts)
    else:
        st.warning("Please enter or upload text data for analysis.")

# Definitions
st.header("Definitions")
st.markdown(
    "**Word Cloud:** A visual representation of frequently occurring words in a text, where the size of each word is proportional to its frequency.")
st.markdown(
    "**Sentiment Analysis:** The process of determining the sentiment (positive, negative, or neutral) expressed in a piece of text.")
st.markdown(
    "**Polarity:** A measure of sentiment, where positive values indicate positivity, negative values indicate negativity, and zero indicates neutrality.")
st.markdown(
    "**Sentiment Emoji:** Emoji representation of sentiment - 😃 for positive, 😞 for negative, and 😐 for neutral.")
st.markdown(
    "**Sentiment Color:** Visual indication of sentiment, where green represents positive, red represents negative, and yellow represents neutral.")

# Add additional information or explanations as needed

