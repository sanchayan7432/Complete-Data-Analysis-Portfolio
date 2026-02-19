# healthcare_analysis.py
# Week 2 – Healthcare Data Analysis & Disease Risk Prediction

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# Paths & Setup
# -------------------------------
BASE_DIR = "project2_healthcare_analysis"
DATA_PATH = f"{BASE_DIR}/datasets/diabetes.csv"
VIZ_DIR = f"{BASE_DIR}/visualizations"
REPORT_DIR = f"{BASE_DIR}/reports"

os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

sns.set(style="whitegrid")

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv(DATA_PATH)

print("\n📊 DATASET OVERVIEW")
print("-------------------")
print(df.head())
print("\nShape:", df.shape)

# -------------------------------
# Data Cleaning (CORRECT WAY)
# -------------------------------
cols_with_zero_as_missing = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
]

# Replace 0 with NaN
df[cols_with_zero_as_missing] = df[cols_with_zero_as_missing].replace(0, np.nan)

# Fill NaN with median (NO chained assignment)
df[cols_with_zero_as_missing] = df[cols_with_zero_as_missing].apply(
    lambda col: col.fillna(col.median())
)

# Final safety check
assert df.isna().sum().sum() == 0, "❌ NaN values still exist!"

# -------------------------------
# Exploratory Data Analysis (EDA)
# -------------------------------

# Outcome Distribution
plt.figure()
sns.countplot(x="Outcome", data=df)
plt.title("Diabetes Outcome Distribution")
plt.savefig(f"{VIZ_DIR}/outcome_distribution.png")
plt.close()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig(f"{VIZ_DIR}/correlation_heatmap.png")
plt.close()

# Age vs Glucose
plt.figure()
sns.scatterplot(x="Age", y="Glucose", hue="Outcome", data=df)
plt.title("Age vs Glucose Level")
plt.savefig(f"{VIZ_DIR}/age_vs_glucose.png")
plt.close()

# BMI vs Outcome
plt.figure()
sns.boxplot(x="Outcome", y="BMI", data=df)
plt.title("BMI vs Diabetes Outcome")
plt.savefig(f"{VIZ_DIR}/bmi_vs_outcome.png")
plt.close()

# -------------------------------
# Statistical Analysis
# -------------------------------
stats_summary = df.groupby("Outcome").mean()

print("\n📈 STATISTICAL SUMMARY (Mean Values)")
print("-----------------------------------")
print(stats_summary)

# -------------------------------
# Machine Learning
# -------------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n🤖 MODEL PERFORMANCE")
print("-------------------")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix Plot
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(f"{VIZ_DIR}/confusion_matrix.png")
plt.close()

# -------------------------------
# Business / Medical Insights
# -------------------------------
avg_age_diabetic = df[df["Outcome"] == 1]["Age"].mean()
avg_bmi_diabetic = df[df["Outcome"] == 1]["BMI"].mean()
diabetes_rate = df["Outcome"].mean() * 100

report = f"""
HEALTHCARE DATA ANALYSIS REPORT
===============================

Dataset: Pima Indians Diabetes Dataset

KEY FINDINGS
------------
• Diabetes Prevalence: {diabetes_rate:.2f}%
• Average Age (Diabetic Patients): {avg_age_diabetic:.1f} years
• Average BMI (Diabetic Patients): {avg_bmi_diabetic:.1f}

MODEL PERFORMANCE
-----------------
• Logistic Regression Accuracy: {accuracy * 100:.2f}%

MEDICAL INSIGHTS
----------------
1. Higher glucose and BMI strongly correlate with diabetes.
2. Patients above age 35 show significantly higher risk.
3. Preventive lifestyle interventions can reduce risk.
4. Early screening should prioritize high-BMI individuals.
"""

with open(f"{REPORT_DIR}/healthcare_report.txt", "w") as f:
    f.write(report)

print("\n📄 Report saved to reports/healthcare_report.txt")
print("📊 Visualizations saved to visualizations/")
print("\n✅ WEEK 2 PROJECT COMPLETED SUCCESSFULLY")
