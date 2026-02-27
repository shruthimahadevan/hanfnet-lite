"""
HANF-Net Lite - Data Cleaner
Robust cleaning for real-world messy insurance datasets
"""

import pandas as pd
import numpy as np
import os


# ==========================================================
# Utility: Normalize column names
# ==========================================================
def normalize_columns(df):
    """
    Make column names consistent:
    - strip spaces
    - remove special chars
    - lowercase
    """
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"[₹()]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
    )
    return df


# ==========================================================
# AGE CLEANING
# ==========================================================
def clean_age_column(df):
    print("🔄 Cleaning Age column...")

    if 'Age' not in df.columns:
        print("⚠️ Age column not found. Skipping age cleaning.")
        return df

    def extract_age(val):
        if pd.isna(val):
            return np.nan
        try:
            if isinstance(val, str):
                val = val.lower().replace("yrs", "").replace("years", "").strip()
            return int(float(val))
        except:
            return np.nan

    df['Age_clean'] = df['Age'].apply(extract_age)

    print(f"   → {df['Age_clean'].isna().sum()} missing/invalid ages")

    return df


# ==========================================================
# MONETARY CLEANING
# ==========================================================
def clean_monetary_columns(df):
    print("🔄 Cleaning monetary columns...")

    money_cols = ['Bill Amt', 'Approved Amt', 'Bill Amt ₹', 'Approved Amt ₹']

    for col in df.columns:
        if "Bill" in col or "Approved" in col:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("₹", "", regex=False)
                .str.replace(",", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ==========================================================
# COVERAGE CALCULATION
# ==========================================================
def calculate_coverage(df):
    print("🔄 Calculating coverage %...")

    bill_col = None
    approved_col = None

    # Automatically detect correct columns
    for col in df.columns:
        if "Bill" in col:
            bill_col = col
        if "Approved" in col:
            approved_col = col

    if bill_col is None or approved_col is None:
        print("⚠️ Bill or Approved column not found. Skipping coverage.")
        return df

    df['Coverage'] = df[approved_col] / df[bill_col]

    # Handle division by zero
    df['Coverage'] = df['Coverage'].replace([np.inf, -np.inf], np.nan)

    # Clip realistic bounds
    df['Coverage'] = df['Coverage'].clip(0, 1)

    anomalies = df[df['Coverage'] > 1]
    print(f"   → Found {len(anomalies)} >100% anomalies")

    return df


# ==========================================================
# UNKNOWN HANDLING
# ==========================================================
def handle_unknown_values(df):
    print("🔄 Handling Unknown values...")

    for col in df.columns:
        if df[col].dtype == "object":
            unknown_count = (df[col].astype(str).str.lower() == "unknown").sum()
            if unknown_count > 0:
                print(f"   → {col}: {unknown_count} 'Unknown' values")

    return df


# ==========================================================
# OUTLIER REMOVAL
# ==========================================================
def remove_outliers(df, column, threshold=3):
    print("🔄 Removing outliers...")

    if column not in df.columns:
        print("⚠️ Column not found for outlier removal.")
        return df

    df = df.copy()
    z = np.abs((df[column] - df[column].mean()) / df[column].std())
    original_len = len(df)
    df = df[z < threshold]

    print(f"   → Removed {original_len - len(df)} outliers")

    return df


# ==========================================================
# MASTER CLEANER
# ==========================================================
def clean_dataset(df):
    print("\n" + "="*50)
    print("🧹 CLEANING DATASET")
    print("="*50)

    df = df.copy()

    df = normalize_columns(df)
    df = clean_monetary_columns(df)
    df = clean_age_column(df)
    df = calculate_coverage(df)
    df = handle_unknown_values(df)

    if 'Coverage' in df.columns:
        df = remove_outliers(df, 'Coverage', threshold=3)

    print("\n✅ Cleaning complete!")
    return df


# ==========================================================
# SAVE FUNCTION
# ==========================================================
def save_clean_data(df, output_path="data/processed/cleaned_dataset.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"💾 Saved cleaned data to: {output_path}")


# ==========================================================
# TEST BLOCK
# ==========================================================
if __name__ == "__main__":
    from data_loader import load_raw_data

    df_raw = load_raw_data()
    df_clean = clean_dataset(df_raw)

    print(f"\nClean data shape: {df_clean.shape}")

    if 'Coverage' in df_clean.columns:
        print("\nCoverage stats:")
        print(df_clean['Coverage'].describe())