import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay
)
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# 📂 Paths
# -----------------------------
base_path = r"D:\data_analysis_portfolio\project5_ecommerce_analytics\datasets"
viz_path = r"D:\data_analysis_portfolio\project5_ecommerce_analytics\visualizations"
report_path = r"D:\data_analysis_portfolio\project5_ecommerce_analytics\reports"

os.makedirs(viz_path, exist_ok=True)
os.makedirs(report_path, exist_ok=True)

print("📥 Loading datasets...")
print(f"📂 Using dataset path: {base_path}")

# -----------------------------
# 📥 Load datasets
# -----------------------------
demographics = pd.read_csv(os.path.join(base_path, "Telco_customer_churn_demographics.csv"))
location = pd.read_csv(os.path.join(base_path, "Telco_customer_churn_location.csv"))
services = pd.read_csv(os.path.join(base_path, "Telco_customer_churn_services.csv"))
status = pd.read_csv(os.path.join(base_path, "Telco_customer_churn_status.csv"))
population = pd.read_csv(os.path.join(base_path, "Telco_customer_churn_population.csv"))

# -----------------------------
# 🔧 Normalize column names
# -----------------------------
def normalize_customer_id(df, name):
    if "CustomerID" in df.columns:
        return df
    elif "Customer ID" in df.columns:
        df = df.rename(columns={"Customer ID": "CustomerID"})
        return df
    else:
        raise ValueError(f"❌ CustomerID missing in {name}")

datasets = {
    "demographics": demographics,
    "location": location,
    "services": services,
    "status": status
}

for name, df in datasets.items():
    datasets[name] = normalize_customer_id(df, name)

demographics = datasets["demographics"]
location = datasets["location"]
services = datasets["services"]
status = datasets["status"]

# -----------------------------
# 🔗 Merge datasets
# -----------------------------
print("🔗 Merging datasets...")

df = demographics.merge(location, on="CustomerID", how="left") \
                 .merge(services, on="CustomerID", how="left") \
                 .merge(status, on="CustomerID", how="left")

# Merge population via Zip Code
if "Zip Code" in df.columns and "Zip Code" in population.columns:
    population["Population"] = (
        population["Population"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    df = df.merge(population[["Zip Code", "Population"]], on="Zip Code", how="left")

# -----------------------------
# 📊 Overview
# -----------------------------
print("\n📊 DATASET OVERVIEW")
print("-------------------")
print(df.head())
print(f"\nShape: {df.shape}")

# -----------------------------
# 🧹 Data Cleaning
# -----------------------------
print("\n🧹 Cleaning data...")

# Convert numeric columns safely
numeric_cols = ["Total Charges", "Monthly Charge", "CLTV", "Population"]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill missing numeric values
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Fill categorical NaNs
cat_cols = df.select_dtypes(include=["object", "string"]).columns
for col in cat_cols:
    df[col] = df[col].fillna("Unknown")

# -----------------------------
# 🎯 Target Variable
# -----------------------------
if "Churn Value" not in df.columns:
    raise ValueError("❌ Target column 'Churn Value' not found")

y = df["Churn Value"]

# Drop leakage columns
leakage_cols = [
    "Churn Label",
    "Churn Value",
    "Churn Score",
    "Churn Category",
    "Churn Reason",
    "Customer Status"
]

X = df.drop(columns=[col for col in leakage_cols if col in df.columns])

# -----------------------------
# 🔠 Encode categorical features
# -----------------------------
print("🔠 Encoding categorical features...")

label_encoders = {}
for col in X.select_dtypes(include=["object", "string"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# -----------------------------
# 📏 Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# ✂ Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 🤖 Model Training
# -----------------------------
print("\n🤖 Training churn prediction model...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# 📈 Predictions
# -----------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# -----------------------------
# 📊 Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n🤖 MODEL PERFORMANCE")
print("-------------------")
print(f"Accuracy: {accuracy:.2%}")
print(f"ROC-AUC: {roc_auc:.3f}\n")

report = classification_report(y_test, y_pred)
print(report)

# -----------------------------
# 🔲 Confusion Matrix Plot
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.savefig(os.path.join(viz_path, "confusion_matrix.png"))
plt.close()

# -----------------------------
# 📉 ROC Curve
# -----------------------------
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.savefig(os.path.join(viz_path, "roc_curve.png"))
plt.close()

# -----------------------------
# ⭐ Feature Importance
# -----------------------------
importances = model.feature_importances_
feature_names = X.columns

fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)[:15]

plt.figure()
fi.plot(kind="bar")
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(viz_path, "feature_importance.png"))
plt.close()

# -----------------------------
# 📝 Save Report
# -----------------------------
with open(os.path.join(report_path, "churn_report.txt"), "w", encoding="utf-8") as f:
    f.write("TELCO CUSTOMER CHURN ANALYSIS REPORT\n")
    f.write("====================================\n\n")
    f.write(f"Dataset Shape: {df.shape}\n\n")
    f.write(f"Accuracy: {accuracy:.2%}\n")
    f.write(f"ROC-AUC: {roc_auc:.3f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print(f"\n📄 Report saved to: {report_path}")
print(f"📊 Visualizations saved to: {viz_path}")
print("\n✅ PROJECT 5 COMPLETED SUCCESSFULLY 🚀")

