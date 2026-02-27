"""
Prepare final dataset for neural network
Selects best 15 features
"""

import pandas as pd
import numpy as np
import os

# Load features
df = pd.read_csv("data/processed/final_features.csv")
print(f"📊 Loaded {len(df)} rows with {len(df.columns)} columns")

# Selected features (based on correlation analysis)
selected_features = [
    # High positive correlation
    'ID_Coverage',
    'Zone_Insurer',
    'Disease_Avg_Cov',
    'Insurer_Generosity',
    'Insurer_Abd_Gen',
    'ID_Deviation',
    'Disease_Avg_LOS',
    
    # High negative correlation
    'ID_Rejection',
    'Insurer_Rejection_Rate',
    'Disease_Risk',
    'Combined_Risk',
    'LOS_Risk_Short',
    
    # Financial (log transformed)
    'Log_Bill',
    'Cost_per_Day',
    'Bill_Pct_Insurer'
]

print(f"\n✅ Selected {len(selected_features)} features")

# Create final dataset
X = df[selected_features].copy()
y = df['Coverage'].copy()

# Handle any missing values
X = X.fillna(X.mean())

# Save
os.makedirs("data/model_ready", exist_ok=True)
X.to_csv("data/model_ready/X_features.csv", index=False)
y.to_csv("data/model_ready/y_target.csv", index=False)

# Also save with claim IDs for reference
df[['No.'] + selected_features + ['Coverage']].to_csv(
    "data/model_ready/full_model_data.csv", index=False
)

print("\n💾 Saved model-ready data to data/model_ready/")
print("\n📋 Feature list:")
for i, f in enumerate(selected_features, 1):
    print(f"  {i:2d}. {f}")

# Quick stats
print("\n📊 Data shape:", X.shape)
print("Target stats:")
print(y.describe())