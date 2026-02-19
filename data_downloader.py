import os
import yfinance as yf

folder_path = r"D:\data_analysis_portfolio\project4_financial_analysis\datasets"
os.makedirs(folder_path, exist_ok=True)

data = yf.download("AAPL", start="2025-02-11", end="2026-02-11")

file_path = os.path.join(folder_path, "AAPL.csv")
data.to_csv(file_path)

print("Dataset saved successfully!")
