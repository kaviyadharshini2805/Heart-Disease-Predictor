import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("framingham_heart_disease.csv")

print("\n--- Dataset Head ---")
print(df.head())

print("\n--- Columns ---")
print(df.columns)

# -----------------------------
# Drop Unnecessary Columns
# -----------------------------
df = df.drop(["education"], axis=1)
print("\n--- After Dropping 'education' ---")
print(df.head())

# -----------------------------
# Check Null Values
# -----------------------------
print("\n--- Null Values ---")
print(df.isnull().sum())

# -----------------------------
# Filling Missing Values Using Mean
# -----------------------------
fill_cols = ["cigsPerDay", "BPMeds", "totChol", "BMI", "heartRate", "glucose"]

for col in fill_cols:
    df[col].fillna(round(df[col].mean()), inplace=True)

print("\n--- After Filling Nulls ---")
print(df.isnull().sum())

# -----------------------------
# Visualization (Pairplot)
# -----------------------------
cols = ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]

sns.pairplot(df[cols])
plt.show()

# -----------------------------
# Feature & Target
# -----------------------------
X = df.drop("TenYearCHD", axis=1)
y = df["TenYearCHD"]

# -----------------------------
# Split Data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -----------------------------
# Train Logistic Regression Model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# Accuracy Scores
# -----------------------------
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print(f"\nTraining Accuracy: {accuracy_score(y_train, train_pred)}")
print(f"Testing Accuracy: {accuracy_score(y_test, test_pred)}\n")

# -----------------------------
# Reports and Confusion Matrix
# -----------------------------
print("\n--- Classification Report ---")
print(classification_report(y_test, test_pred))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, test_pred))

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "heart_disease_model.pkl")
joblib.dump(list(X.columns), "model_features.pkl")

print("\nModel saved successfully!")
