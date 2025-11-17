# ❤️ Heart Disease Predictor

A machine learning–powered web application built with Logistic Regression and Streamlit to predict the likelihood of heart disease based on medical attributes.

## 📌 Features

🔍 Predict heart disease using a trained Logistic Regression model

📊 Preprocessed and scaled dataset for accurate predictions

🧠 Model trained using scikit-learn

🌐 Interactive UI built with Streamlit

💾 Model saved & loaded using Joblib

## 📁 Project Structure

Heart-Disease-Predictor/

│

  ├── heart_disease_prediction.py       # Model training script
  
  ├── streamlit_heart_disease_predictor.py   # Streamlit web app
  
  ├── heart.csv                          # Dataset
  
  ├── heart_disease_model.pkl            # Saved ML model
  
  ├── scaler.pkl                         # StandardScaler object
  
└  ── README.md                          # Project documentation

## 🧠 Machine Learning Model

This project uses:

Logistic Regression

StandardScaler for feature scaling

Train-test split (80-20)

Model evaluation using:

Accuracy Score

Confusion Matrix

Classification Report

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies
pip install -r requirements.txt


Or install manually:

pip install pandas numpy scikit-learn streamlit joblib

### 2️⃣ Train the Model (optional)

If you want to retrain:

python heart_disease_prediction.py


This will generate:

heart_disease_model.pkl

scaler.pkl

### 3️⃣ Run the Streamlit Web App
streamlit run streamlit_heart_disease_predictor.py


Your app will open in the browser automatically.

## 📊 Dataset

The dataset contains medical features such as:

Age

Sex

Chest Pain Type

Blood Pressure

Cholesterol

Fasting Blood Sugar

ECG

Max Heart Rate

Exercise Induced Angina

Oldpeak

Slope

Major Vessels

Thal

## 🚀 Future Improvements

Add more ML models (Random Forest, SVM, XGBoost)

Add visualization of patient inputs

Deploy on cloud (Streamlit Cloud / Render / HuggingFace Spaces)
