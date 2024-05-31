import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Title
st.title('Credit Analyst: Creditworthiness Prediction')

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

# Display model accuracy
st.write(f'Model Accuracy: {accuracy:.2f}')
