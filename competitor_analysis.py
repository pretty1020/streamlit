import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster, Search
import matplotlib.pyplot as plt

# Load data
file_path = 'https://raw.githubusercontent.com/pretty1020/streamlit/main/dental_clinic_raw_marikina_1.csv'
df = pd.read_csv(file_path)


# Display the first few rows of the DataFrame to understand its structure
st.write(df.head())

# Extract latitude and longitude from 'Map location (latitude,longitude)' column
def extract_lat_long(location):
    if pd.isna(location):
        return None, None
    try:
        lat, long = location.strip("()").split(",")
        return float(lat), float(long)
    except:
        return None, None

df['LAT'], df['LONG'] = zip(*df['Map location (latitude,longitude)'].map(extract_lat_long))

# Data preprocessing
df['Average Rating'] = pd.to_numeric(df['Average Rating'], errors='coerce')
df['Google Rating'] = pd.to_numeric(df['Google Rating'], errors='coerce')
df['Facebook Rating'] = pd.to_numeric(df['Facebook Rating'], errors='coerce')
df.dropna(subset=['Average Rating'], inplace=True)

# Sentiment analysis
def get_sentiment(review):
    if pd.isna(review):
        return ""
    analysis = TextBlob(review)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity == 0:
        return "Neutral"
    else:
        return "Negative"

# Assuming 'Customer Reviews Google' and 'Customer Reviews Facebook' contain the reviews
df['Google Sentiment'] = df['Customer Reviews Google'].apply(get_sentiment)
df['Facebook Sentiment'] = df['Customer Reviews Facebook'].apply(get_sentiment)

# Calculate average rating for each clinic
clinic_avg_rating = df.groupby(['Clinic Name', 'Address'], as_index=False)['Average Rating'].mean()
clinic_avg_rating.columns = ['Clinic Name', 'Address', 'Average Rating']

# Merge with sentiment analysis
clinic_data = df.groupby(['Clinic Name', 'Address']).agg({
    'Average Rating': 'mean',
    'Google Sentiment': lambda x: ','.join(x[x != ""]).split(',')[0] if len(x[x != ""]) > 0 else "",
    'Facebook Sentiment': lambda x: ','.join(x[x != ""]).split(',')[0] if len(x[x != ""]) > 0 else "",
    'Google Rating': 'mean',
    'Facebook Rating': 'mean'
}).reset_index()

clinic_data.columns = ['Clinic Name', 'Address', 'Average Rating', 'Google Sentiment', 'Facebook Sentiment', 'Google Rating', 'Facebook Rating']

# Adding latitude and longitude to clinic_data
clinic_data = pd.merge(clinic_data, df[['Clinic Name', 'LAT', 'LONG']], on='Clinic Name', how='left')

# Ranking
clinic_data['Ranking'] = clinic_data['Average Rating'].rank(ascending=False)

# Streamlit app
st.title('Dental Clinics Analysis in Marikina')

# Display table
st.subheader('Clinic Rankings')
clinic_data_sorted = clinic_data.sort_values(by='Ranking')
st.dataframe(clinic_data_sorted[['Clinic Name', 'Address', 'Average Rating', 'Google Rating', 'Facebook Rating', 'Google Sentiment', 'Facebook Sentiment', 'Ranking']])

# Geographic clusters or patterns
st.subheader('Geographic Clusters in Average Scores')
geo_data = clinic_data[['Clinic Name', 'Address', 'LAT', 'LONG', 'Average Rating']].dropna()

if not geo_data.empty:
    geo_clusters = folium.Map(location=[14.6507, 121.1029], zoom_start=13)
    marker_cluster = MarkerCluster().add_to(geo_clusters)

    for idx, row in geo_data.iterrows():
        folium.CircleMarker(
            location=[row['LAT'], row['LONG']],
            radius=5,
            weight=2,
            color='green' if row['Average Rating'] >= 4 else 'red',
            fill=True,
            fill_color='green' if row['Average Rating'] >= 4 else 'red',
            fill_opacity=0.6,
            popup=f"Clinic Name: {row['Clinic Name']}<br>Address: {row['Address']}<br>Average Rating: {row['Average Rating']:.2f}"
        ).add_to(marker_cluster)

    # Add search functionality
    search = Search(
        layer=marker_cluster,
        search_label='Clinic Name',
        placeholder='Search for a clinic...',
        collapsed=False
    ).add_to(geo_clusters)

    st_folium(geo_clusters, width=1000, height=800)
else:
    st.write("No geographic data available to display clusters.")

# Analyze the location of clinics with most numbers of red or low scores
st.subheader('Analysis of Clinics with Low Scores')
low_score_clinics = clinic_data[clinic_data['Average Rating'] < 4]

if not low_score_clinics.empty:
    low_score_count = low_score_clinics.groupby(['LAT', 'LONG', 'Clinic Name', 'Address']).size().reset_index(name='Count')
    max_low_score_clinics = low_score_count[low_score_count['Count'] == low_score_count['Count'].max()]

    st.write("Clinics with the most number of low scores (red markers):")
    st.dataframe(max_low_score_clinics[['Clinic Name', 'Address', 'Count']])

    low_score_map = folium.Map(location=[14.6507, 121.1029], zoom_start=13)
    for idx, row in max_low_score_clinics.iterrows():
        folium.Marker(
            location=[row['LAT'], row['LONG']],
            popup=f"{row['Clinic Name']}<br>Address: {row['Address']}<br>Low Score Count: {row['Count']}",
            icon=folium.Icon(color='red')
        ).add_to(low_score_map)

    st_folium(low_score_map, width=1000, height=800)
else:
    st.write("No clinics with low scores found.")

# Sentiment Analysis Summary
st.subheader('Summary of Sentiments Based on Customer Reviews')
sentiment_summary = pd.concat([clinic_data['Google Sentiment'], clinic_data['Facebook Sentiment']]).value_counts()
st.write(sentiment_summary)

# Additional Insights

# Distribution of Ratings
st.subheader('Distribution of Average Ratings')
plt.figure(figsize=(10, 6))
plt.hist(clinic_data['Average Rating'], bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Average Rating')
plt.ylabel('Number of Clinics')
plt.title('Distribution of Average Ratings')
st.pyplot(plt)

# Best Practices for Higher Ratings
st.subheader('Best Practices for Achieving Higher Ratings')
st.write("""
1. **Provide Excellent Customer Service**: Ensure that your staff is friendly, professional, and attentive to patients' needs.
2. **Maintain Clean and Comfortable Facilities**: A clean and welcoming environment can significantly impact patient satisfaction.
3. **Follow Up with Patients**: Post-appointment follow-ups show that you care about their well-being and can help address any issues promptly.
4. **Encourage Reviews**: Politely ask satisfied patients to leave positive reviews on Google and Facebook.
5. **Respond to Reviews**: Engage with your reviewers by thanking them for positive feedback and addressing any negative comments constructively.
6. **Offer Competitive Pricing**: Ensure your pricing is competitive and transparent.
7. **Keep Up with Technology**: Use the latest dental technology to provide the best care possible.
8. **Provide Clear Communication**: Clearly explain procedures, costs, and aftercare to patients to build trust.
9. **Offer Convenient Scheduling**: Provide flexible scheduling options to accommodate patients' busy lives.
10. **Continuously Improve**: Regularly seek feedback and make improvements based on patient suggestions and reviews.
""")
