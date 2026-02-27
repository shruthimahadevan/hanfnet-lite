"""
Quick fix to clean and save the dataset
"""

import pandas as pd
import numpy as np
import os

print("🔧 Loading raw Excel file...")

# Load the Excel file
df_raw = pd.read_excel(
    "data/raw/oioooo.xlsx", 
    sheet_name="📋 Full Dataset (421)",
    header=3  # The actual data starts at row 4 (index 3)
)

print(f"✅ Loaded {len(df_raw)} rows")

# Rename columns properly
df_raw.columns = [
    'No.', 'Gender', 'Age', 'Payer Zone', 'Insurance Company',
    'Treatment Type', 'Disease Category', 'LOS (Days)', 'Bill Amt (₹)',
    'Approved Amt (₹)', 'Approval Rate', 'Original Status', 'Outcome', 'TPA Remarks'
]

print("✅ Columns renamed")

# Clean Age column
def extract_age(age_val):
    if pd.isna(age_val) or age_val == 'Unknown':
        return np.nan
    try:
        if isinstance(age_val, str):
            age_num = age_val.replace('yrs', '').strip()
            return int(age_num)
        return age_val
    except:
        return np.nan

df_raw['Age_clean'] = df_raw['Age'].apply(extract_age)

# Calculate Coverage
df_raw['Coverage'] = df_raw['Approved Amt (₹)'] / df_raw['Bill Amt (₹)']
df_raw['Coverage'] = df_raw['Coverage'].clip(0, 1)

# Remove extreme outliers (like claim 124 with 357%)
df_clean = df_raw[df_raw['Coverage'] <= 1].copy()

print(f"✅ Cleaned: {len(df_clean)} claims remain")

# Create processed folder if it doesn't exist
os.makedirs("data/processed", exist_ok=True)

# Save cleaned dataset
output_path = "data/processed/cleaned_dataset.csv"
df_clean.to_csv(output_path, index=False)
print(f"💾 Saved cleaned dataset to: {output_path}")

# Show first few rows
print("\n👁️ Preview:")
print(df_clean[['No.', 'Gender', 'Age_clean', 'Insurance Company', 'Coverage']].head())