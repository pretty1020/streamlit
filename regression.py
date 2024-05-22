import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Function to preprocess data
def preprocess_data(df):
    df.dropna(inplace=True)
    df['Marketing'] = df['Marketing'].map({'no': 0, 'yes': 1})
    df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    return df

# Function to train model
def train_model(X_train, y_train, model_type='LinearRegression'):
    if model_type == 'LinearRegression':
        model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to predict using trained model
def predict(model, input_values):
    return model.predict([input_values])[0]

# Function to display results
def visualize_results(y_test, y_pred_linear):
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred_linear, label='Linear Regression')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Patient Volume')
    plt.ylabel('Predicted Patient Volume')
    plt.title('Actual vs. Predicted Patient Volume')
    plt.legend()
    st.pyplot(plt)

    plt.figure(figsize=(10, 5))
    sns.histplot(y_test - y_pred_linear, kde=True, color='blue', label='Linear Regression Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.legend()
    st.pyplot(plt)

# Main Streamlit app
def main():
    st.title('Patient Volume Prediction')

    # Sidebar for predictor inputs
    st.sidebar.header('Enter Predictor Values:')
    staff_scheduled = st.sidebar.number_input('Staff Scheduled', min_value=0, step=1)
    service_time = st.sidebar.number_input('Service Time (mins)', min_value=0, step=1)
    marketing = st.sidebar.selectbox('Marketing', ['no', 'yes'])
    day_of_week = st.sidebar.selectbox('Day of Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    month = st.sidebar.selectbox('Month', ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])

    # Encode categorical variables
    marketing_encoded = 1 if marketing == 'yes' else 0
    day_of_week_encoded = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].index(day_of_week)
    month_encoded = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'].index(month) + 1

    # Upload data
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write('Uploaded dataset:')
        st.write(df.head())

        # Preprocess data
        df = preprocess_data(df)

        # Select features and target variable
        X = df[['Staff Scheduled', 'Service Time (mins)', 'Marketing', 'DayOfWeek', 'Month']]
        y = df['Patient Volume']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        linear_model = train_model(X_train, y_train, model_type='LinearRegression')

        # Predict
        y_pred_linear = linear_model.predict(X_test)

        # Evaluate the model
        mse_linear = mean_squared_error(y_test, y_pred_linear)

        st.write("## Model Results")
        st.write("Mean Squared Error (Linear Regression):", mse_linear)
       
        
        # Visualize results
        visualize_results(y_test, y_pred_linear)

        # Predict using sidebar inputs
        input_values = [staff_scheduled, service_time, marketing_encoded, day_of_week_encoded, month_encoded]
        predicted_volume_linear = predict(linear_model, input_values)

        # Display predicted patient volume
        st.write('## Predicted Patient Volume:')
        st.write('Linear Regression:', predicted_volume_linear)

        st.write("""
        ### Definitions and Interpretations
        - **Residuals**: Residuals are the differences between the actual values and the predicted values. They represent the error in the predictions made by the model. In other words, residuals show how far off the model's predictions are from the actual values.
        - **Day of Week**: Day of Week is the day of the week when the patient service was provided.
        - **Month**: Month is the month when the patient service was provided.
        - **Marketing Campaign**: Marketing Campaign - whether a marketing campaign was active during the patient service.
        - **Mean Squared Error (MSE)**: MSE is a metric that measures the average of the squares of the errors, that is, the average squared difference between the actual and predicted values. Lower MSE values indicate better model performance as they imply smaller errors.
        """)
    
        
if __name__ == '__main__':
    main()
