# -----------------------------
# STEP 15: Streamlit App for Employee Performance Prediction
# -----------------------------
import streamlit as st
import joblib
import numpy as np

# Load saved model and scaler
model = joblib.load("model/linear_svc_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# App title
st.title("Employee Performance Prediction")

# Input fields
years = st.number_input("Years at Company", 0, 40, 5)
salary = st.number_input("Monthly Salary", 1000, 20000, 5000)
overtime = st.number_input("Overtime Hours", 0, 100, 10)
projects = st.number_input("Projects Handled", 0, 50, 2)
promotions = st.number_input("Number of Promotions", 0, 20, 1)
satisfaction = st.slider("Employee Satisfaction Score", 0.0, 1.0, 0.5)

# Predict button
if st.button("Predict"):
    # Create input array and scale features
    input_data = np.array([[years, salary, overtime, projects, promotions, satisfaction]])
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    st.success(f"Predicted Performance Score: {prediction[0]}")
    st.balloons()
