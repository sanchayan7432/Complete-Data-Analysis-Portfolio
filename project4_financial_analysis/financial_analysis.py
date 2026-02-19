# ============================================================
# 📈 PROJECT 4: FINANCIAL MARKET ANALYSIS (IMPROVED)
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# 1. PATH SETUP
# ============================================================

BASE_DIR = r"D:\data_analysis_portfolio\project4_financial_analysis"
DATA_PATH = os.path.join(BASE_DIR, "datasets", "AAPL.csv")
VIS_DIR = os.path.join(BASE_DIR, "visualizations")
REPORT_PATH = os.path.join(BASE_DIR, "financial_report.txt")

os.makedirs(VIS_DIR, exist_ok=True)

# ============================================================
# 2. LOAD & CLEAN DATA (Yahoo Finance Fix)
# ============================================================

df = pd.read_csv(DATA_PATH)

# Handle Yahoo Finance extra rows
if 'Price' in df.columns:
    df = pd.read_csv(DATA_PATH, skiprows=2)
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

df = df.dropna()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

print("\n📊 DATASET OVERVIEW")
print("-------------------")
print(df.head())
print("\nShape:", df.shape)

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

df['Daily Return'] = df['Close'].pct_change()

# Moving averages
df['MA_10'] = df['Close'].rolling(10).mean()
df['MA_30'] = df['Close'].rolling(30).mean()

# Volatility (rolling std)
df['Volatility'] = df['Daily Return'].rolling(10).std()

# Momentum
df['Momentum'] = df['Close'] - df['Close'].shift(5)

df = df.dropna()

# ============================================================
# 4. RISK METRICS
# ============================================================

annual_return = df['Daily Return'].mean() * 252
annual_volatility = df['Daily Return'].std() * np.sqrt(252)
sharpe_ratio = annual_return / annual_volatility

# Max Drawdown
df['Cumulative Return'] = (1 + df['Daily Return']).cumprod()
df['Running Max'] = df['Cumulative Return'].cummax()
df['Drawdown'] = df['Cumulative Return'] / df['Running Max'] - 1
max_drawdown = df['Drawdown'].min()

print("\n📊 RISK & PERFORMANCE")
print("-------------------")
print(f"Annual Return: {annual_return:.2%}")
print(f"Annual Volatility: {annual_volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")

# ============================================================
# 5. BUILD PREDICTION MODEL
# ============================================================

# Target: Will price go UP tomorrow?
df['Target'] = np.where(df['Daily Return'].shift(-1) > 0, 1, 0)
df = df.dropna()

features = ['MA_10', 'MA_30', 'Volatility', 'Momentum']
X = df[features]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n🤖 MODEL PERFORMANCE")
print("-------------------")
print(f"Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ============================================================
# 6. VISUALIZATIONS
# ============================================================

# Stock Price + Moving Averages
plt.figure()
plt.plot(df['Date'], df['Close'], label='Close Price')
plt.plot(df['Date'], df['MA_10'], label='MA 10')
plt.plot(df['Date'], df['MA_30'], label='MA 30')
plt.legend()
plt.title("Stock Price with Moving Averages")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "price_moving_avg.png"))
plt.close()

# Daily Returns
plt.figure()
plt.plot(df['Date'], df['Daily Return'])
plt.title("Daily Returns")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "daily_returns.png"))
plt.close()

# Drawdown
plt.figure()
plt.plot(df['Date'], df['Drawdown'])
plt.title("Drawdown")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "drawdown.png"))
plt.close()

# ============================================================
# 7. SAVE REPORT
# ============================================================

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("FINANCIAL MARKET ANALYSIS REPORT\n")
    f.write("================================\n\n")
    f.write(f"Annual Return: {annual_return:.2%}\n")
    f.write(f"Annual Volatility: {annual_volatility:.2%}\n")
    f.write(f"Sharpe Ratio: {sharpe_ratio:.2f}\n")
    f.write(f"Max Drawdown: {max_drawdown:.2%}\n\n")
    f.write(f"Model Accuracy: {accuracy:.2%}\n")

print(f"\n📄 Report saved to: {REPORT_PATH}")
print(f"📊 Visualizations saved to: {VIS_DIR}")
print("\n✅ WEEK 4 PROJECT COMPLETED SUCCESSFULLY (IMPROVED)")
