from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =====================================================
# PATH SETUP
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "datasets" / "sales_data.csv"
VIZ_DIR = BASE_DIR / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(DATA_PATH)

# =====================================================
# DATA CLEANING
# =====================================================
df.drop_duplicates(inplace=True)

# --- FIX 1: DATE PARSING ---
df["order_date"] = pd.to_datetime(
    df["order_date"],
    dayfirst=True,
    errors="coerce"
)

# --- FIX 2: SALES MUST BE NUMERIC ---
df["sales"] = (
    df["sales"]
    .astype(str)
    .str.replace(",", "", regex=False)
)

df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

# Remove rows where critical values are missing
df.dropna(subset=["order_date", "sales"], inplace=True)

# =====================================================
# 1. MONTHLY SALES TREND
# =====================================================
monthly_sales = (
    df.set_index("order_date")
      .resample("ME")["sales"]
      .sum()
)

plt.figure(figsize=(10, 5))
plt.plot(monthly_sales.index, monthly_sales.values)
plt.title("Monthly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig(VIZ_DIR / "monthly_sales.png")
plt.close()

# =====================================================
# 2. CATEGORY PERFORMANCE
# =====================================================
category_sales = (
    df.groupby("category")["sales"]
      .sum()
      .sort_values()
)

plt.figure(figsize=(8, 5))
category_sales.plot(kind="barh")
plt.title("Sales by Category")
plt.xlabel("Sales")
plt.tight_layout()
plt.savefig(VIZ_DIR / "category_performance.png")
plt.close()

# =====================================================
# 3. CUSTOMER SEGMENTATION (RFM)
# =====================================================
max_date = df["order_date"].max()

rfm = df.groupby("customer_name").agg({
    "order_date": lambda x: (max_date - x.max()).days,
    "order_id": "nunique",
    "sales": "sum"
})

rfm.columns = ["recency", "frequency", "monetary"]

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm["segment"] = kmeans.fit_predict(rfm_scaled)

plt.figure(figsize=(7, 5))
sns.scatterplot(
    x=rfm["recency"],
    y=rfm["monetary"],
    hue=rfm["segment"],
    palette="Set2"
)
plt.title("Customer Segmentation (RFM Analysis)")
plt.tight_layout()
plt.savefig(VIZ_DIR / "customer_segmentation.png")
plt.close()

# =====================================================
# BUSINESS INSIGHTS
# =====================================================
top_20_revenue_share = (
    rfm.sort_values("monetary", ascending=False)
       .head(int(0.2 * len(rfm)))["monetary"].sum()
    / df["sales"].sum()
)

print("\n📊 BUSINESS INSIGHTS")
print("--------------------")
print(f"Total Revenue: ₹{df['sales'].sum():,.2f}")
print(f"Best Category: {category_sales.idxmax()}")
print(f"Top 20% Customers Revenue Share: {top_20_revenue_share:.2%}")
