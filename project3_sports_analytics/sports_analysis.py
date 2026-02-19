# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report
# from pathlib import Path

# # -----------------------------
# # Paths
# # -----------------------------
# BASE_DIR = Path(__file__).resolve().parent
# DATA_PATH = BASE_DIR / "datasets" / "sports_data.xlsx"
# VIS_DIR = BASE_DIR / "visualizations"
# REP_DIR = BASE_DIR / "reports"

# VIS_DIR.mkdir(exist_ok=True)
# REP_DIR.mkdir(exist_ok=True)

# # -----------------------------
# # Load Dataset
# # -----------------------------
# df = pd.read_excel(DATA_PATH)

# print("\n📊 DATASET OVERVIEW")
# print("-------------------")
# print(df.head())
# print("\nShape:", df.shape)

# # -----------------------------
# # Data Cleaning
# # -----------------------------
# df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# numeric_cols = ['mat', 'won', 'lost', 'tied', 'n/r', 'points', 'net_r/r', 'year', 'position']
# numeric_cols = [c for c in numeric_cols if c in df.columns]

# df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
# df = df.dropna()

# # -----------------------------
# # Team Performance Analysis
# # -----------------------------
# team_stats = df.groupby('team').agg({
#     'mat': 'sum',
#     'won': 'sum',
#     'lost': 'sum',
#     'points': 'sum'
# }).reset_index()

# team_stats['win_percentage'] = (team_stats['won'] / team_stats['mat']) * 100

# # -----------------------------
# # Visualization 1: Win Percentage
# # -----------------------------
# plt.figure(figsize=(10, 6))
# sns.barplot(
#     data=team_stats.sort_values("win_percentage", ascending=False),
#     x="win_percentage",
#     y="team"
# )
# plt.title("Team Win Percentage")
# plt.xlabel("Win %")
# plt.ylabel("Team")
# plt.tight_layout()
# plt.savefig(VIS_DIR / "team_win_percentage.png")
# plt.close()

# # -----------------------------
# # Visualization 2: Points Distribution
# # -----------------------------
# plt.figure(figsize=(8, 5))
# sns.histplot(df['points'], bins=10, kde=True)
# plt.title("Distribution of Points")
# plt.xlabel("Points")
# plt.tight_layout()
# plt.savefig(VIS_DIR / "points_distribution.png")
# plt.close()

# # -----------------------------
# # Match Outcome Prediction
# # -----------------------------
# # Create binary target: Top 4 finish = 1 else 0
# df['top4'] = (df['position'] <= 4).astype(int)

# features = ['mat', 'won', 'lost', 'points', 'net_r/r']
# X = df[features]
# y = df['top4']

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.25, random_state=42, stratify=y
# )

# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)

# print("\n🤖 MODEL PERFORMANCE")
# print("-------------------")
# print(f"Accuracy: {accuracy * 100:.2f}%")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# # -----------------------------
# # Report Generation
# # -----------------------------
# with open(REP_DIR / "sports_report.txt", "w", encoding="utf-8") as f:
#     f.write("WEEK 3 – SPORTS ANALYTICS REPORT\n")
#     f.write("=" * 35 + "\n\n")

#     f.write("Dataset Overview:\n")
#     f.write(f"Total Records: {len(df)}\n")
#     f.write(f"Teams Analyzed: {df['team'].nunique()}\n\n")

#     f.write("Top Performing Teams (by Win %):\n")
#     top_teams = team_stats.sort_values("win_percentage", ascending=False).head(5)
#     for _, row in top_teams.iterrows():
#         f.write(f"- {row['team']}: {row['win_percentage']:.2f}%\n")

#     f.write("\nPredictive Model:\n")
#     f.write(f"Top-4 Finish Prediction Accuracy: {accuracy * 100:.2f}%\n")

# print("\n📄 Report saved to reports/sports_report.txt")
# print("📊 Visualizations saved to visualizations/")
# print("\n✅ WEEK 3 PROJECT COMPLETED SUCCESSFULLY")

# =========================================================
# Project 3: Sports Analytics
# File: sports_analysis.py
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path("D:/data_analysis_portfolio/project3_sports_analytics")
DATA_PATH = BASE_DIR / "datasets" / "sports_data.xlsx"
VIS_DIR = BASE_DIR / "visualizations"
REPORT_PATH = BASE_DIR / "sports_report.txt"

VIS_DIR.mkdir(exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_excel(DATA_PATH)
print("Dataset Loaded Successfully")
print(df.head())
print("\nColumns:")
print(df.columns)

# -----------------------------
# Clean Column Names (CRITICAL FIX)
# -----------------------------
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(" ", "_")
      .str.replace("/", "_")
)

# -----------------------------
# Team-Level Statistics
# -----------------------------
team_stats = df.groupby("team").agg({
    "mat": "sum",
    "won": "sum",
    "lost": "sum",
    "points": "sum",
    "net_r_r": "mean"
}).reset_index()

# -----------------------------
# Visualization 1: Matches Won
# -----------------------------
plt.figure()
sns.barplot(data=team_stats, x="won", y="team")
plt.title("Matches Won by Team")
plt.tight_layout()
plt.savefig(VIS_DIR / "matches_won.png")
plt.close()

# -----------------------------
# Visualization 2: Net Run Rate
# -----------------------------
plt.figure()
sns.barplot(data=team_stats, x="net_r_r", y="team")
plt.title("Average Net Run Rate by Team")
plt.tight_layout()
plt.savefig(VIS_DIR / "net_run_rate.png")
plt.close()

# -----------------------------
# Machine Learning
# Predict Top-4 Finish
# -----------------------------
df["top_4"] = (df["position"] <= 4).astype(int)

features = ["mat", "won", "lost", "points", "net_r_r"]
X = df[features]
y = df["top_4"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# -----------------------------
# Save Report
# -----------------------------
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("SPORTS ANALYTICS REPORT\n")
    f.write("=======================\n\n")
    f.write(f"Model Accuracy: {accuracy:.2%}\n\n")
    f.write(report)

# -----------------------------
# Final Output
# -----------------------------
print("\n🤖 MODEL PERFORMANCE")
print("-------------------")
print(f"Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(report)

print("\n📄 Report saved to:", REPORT_PATH)
print("📊 Visualizations saved to:", VIS_DIR)
print("\n✅ WEEK 3 PROJECT COMPLETED SUCCESSFULLY")
