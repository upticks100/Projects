import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIG ---
# Use the exact tickers from your experiment
TARGET_TICKERS = [
    "NVDA","AAPL","MSFT","AMZN","GOOGL","AVGO","META","TSLA","BRK.B","LLY","WMT",
    "JPM","V","ORCL","JNJ","MA","XOM","NFLX","COST","ABBV","PLTR","BAC","HD","AMD",
    "PG","GE","KO","CSCO","CVX","UNH","IBM","MS","WFC","CAT","MU","MRK","AXP","GS",
    "PM","RTX","TMUS","ABT","MCD","TMO","CRM","PEP","ISRG","APP","AMAT"
]

def check_alignment():
    print("--- LOADING DATA ---")
    root = Path(__file__).resolve().parent
    df = pd.read_csv(root / "90-25_Q_Fundamentals.csv")
    mapping = pd.read_csv(root / "gvkeys_to_gics.csv")
    
    # 1. Filter for your 49 Target Tickers
    target_gvkeys = mapping[mapping["tic"].isin(TARGET_TICKERS)]["gvkey"].unique()
    df = df[df["gvkey"].isin(target_gvkeys)]
    
    # 2. Get All Unique Dates (The Tensor Time Axis)
    tensor_dates = sorted(df["datadate"].unique())
    n_dates = len(tensor_dates)
    
    print(f"\nTotal Tickers: {len(target_gvkeys)}")
    print(f"Total Unique Dates (Tensor Axis Length): {n_dates}")
    print(f"Expected Length (if perfectly aligned Quarterly): ~80 (20 years * 4)")
    
    # 3. Pick two heavyweights: Microsoft (Standard) vs Walmart (Offset)
    # If I am right, they report on different days.
    # If I am wrong, they report on the same day.
    
    # Get GVKEYS
    msft_key = mapping[mapping["tic"]=="MSFT"]["gvkey"].values[0]
    wmt_key = mapping[mapping["tic"]=="WMT"]["gvkey"].values[0] # Walmart usually ends Jan/Apr/Jul/Oct
    
    msft_dates = set(df[df["gvkey"]==msft_key]["datadate"])
    wmt_dates = set(df[df["gvkey"]==wmt_key]["datadate"])
    
    overlap = msft_dates.intersection(wmt_dates)
    
    print(f"\n--- ALIGNMENT CHECK ---")
    print(f"Microsoft (MSFT) Report Count: {len(msft_dates)}")
    print(f"Walmart   (WMT)  Report Count: {len(wmt_dates)}")
    print(f"Dates where they match EXACTLY: {len(overlap)}")
    
    if len(overlap) < len(msft_dates) * 0.5:
        print("\n[VERDICT] -> JAGGED GRID CONFIRMED.")
        print("Different companies report on different months.")
        print("This creates gaps (NaNs) in the tensor for specific months.")
    else:
        print("\n[VERDICT] -> PERFECT ALIGNMENT.")
        print("The problem is NOT the dates. Something else is wrong.")

if __name__ == "__main__":
    check_alignment()