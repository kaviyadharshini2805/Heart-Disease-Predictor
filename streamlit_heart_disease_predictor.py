import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# Load Model & Features
model = joblib.load("heart_disease_model.pkl")
feature_names = joblib.load("model_features.pkl")

st.title("🫀 Heart Disease Risk Prediction")
st.write("Enter patient details to predict **10-year heart disease risk**.")

# Input Form
user_input = {}

col1, col2, col3 = st.columns(3)

with col1:
    user_input["male"] = st.selectbox("Male (1=Yes, 0=No)", [0, 1])
    user_input["age"] = st.number_input("Age", 20, 100, 40)
    user_input["currentSmoker"] = st.selectbox("Current Smoker", [0, 1])
    user_input["cigsPerDay"] = st.number_input("Cigarettes Per Day", 0, 60, 10)
    user_input["BPMeds"] = st.number_input("BP Meds (0/1)", 0, 1, 0)

with col2:
    user_input["prevalentStroke"] = st.selectbox("Prevalent Stroke (0/1)", [0, 1])
    user_input["prevalentHyp"] = st.selectbox("Prevalent Hypertension (0/1)", [0, 1])
    user_input["diabetes"] = st.selectbox("Diabetes (0/1)", [0, 1])
    user_input["totChol"] = st.number_input("Total Cholesterol", 100, 600, 200)
    user_input["sysBP"] = st.number_input("Systolic BP", 80, 250, 120)

with col3:
    user_input["diaBP"] = st.number_input("Diastolic BP", 40, 150, 80)
    user_input["BMI"] = st.number_input("BMI", 10.0, 60.0, 25.0)
    user_input["heartRate"] = st.number_input("Heart Rate", 40, 200, 80)
    user_input["glucose"] = st.number_input("Glucose", 50, 400, 90)

#Precit Button
if st.button("Predict"):
    
    # Convert dict to DataFrame
    input_df = pd.DataFrame([user_input])[feature_names]

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1] * 100

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease\nProbability: {prob:.2f}%")
    else:
        st.success(f"💚 Low Risk of Heart Disease\nProbability: {prob:.2f}%")
