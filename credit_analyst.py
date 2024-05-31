import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px

# Title
st.title('Credit Analyst: Creditworthiness Prediction Tool')


# User Guide
st.sidebar.markdown('## User Guide')
st.sidebar.markdown('1. Adjust the number of samples to generate synthetic data using the slider.')
st.sidebar.markdown('2. Enter customer data in the sidebar input fields.')
st.sidebar.markdown('3. Click the "Predict Creditworthiness" button to see the prediction.')
st.sidebar.markdown('4. The prediction result will be displayed in the sidebar.')

# Generate synthetic data
def generate_synthetic_data(num_samples):
    np.random.seed(42)
    age = np.random.randint(18, 71, num_samples)
    income = np.random.normal(50000, 10000, num_samples)
    loan_amount = np.random.randint(1000, 50001, num_samples)
    loan_term = np.random.randint(6, 61, num_samples)
    target = np.random.randint(0, 2, num_samples)
    return pd.DataFrame({
        'age': age,
        'income': income,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'target': target
    })

# Input field for number of samples
num_samples = st.sidebar.slider('Select the number of samples:', min_value=100, max_value=2000, value=1000)

# Generate or load synthetic data
@st.cache_data()
def load_data(num_samples):
    return generate_synthetic_data(num_samples)

data = load_data(num_samples)

# Split data
X = data.drop('target', axis=1)
y = data['target']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Evaluate model
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)

# Input fields
st.sidebar.header('Input Customer Data')
age = st.sidebar.number_input('Age', min_value=18, max_value=70, value=30)
income = st.sidebar.number_input('Income', min_value=0, value=50000)
loan_amount = st.sidebar.number_input('Loan Amount', min_value=1000, value=10000)
loan_term = st.sidebar.number_input('Loan Term (months)', min_value=6, value=12)

# Collect inputs into a DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'income': [income],
    'loan_amount': [loan_amount],
    'loan_term': [loan_term]
})

# Predict button
if st.sidebar.button('Predict Creditworthiness'):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.sidebar.success('The customer is likely to repay the loan.')
    else:
        st.sidebar.error('The customer is likely to default on the loan.')

st.sidebar.markdown('This app uses fake data.For customization, contact the app developer')

# Display model accuracy
st.write(f'Model Accuracy: {accuracy:.2f}')

# Visualization of feature importance
st.header('Feature Importance')
importance = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
st.pyplot(plt)

# Confusion Matrix
st.header('Confusion Matrix')
cm = confusion_matrix(y, y_pred)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Default', 'Repay'], yticklabels=['Default', 'Repay'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
st.pyplot(fig)
