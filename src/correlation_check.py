"""
Quick correlation check for feature selection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load features
df = pd.read_csv("data/processed/final_features.csv")
print(f"📊 Loaded {len(df)} rows with {len(df.columns)} columns")

# Features to check (exclude non-feature columns)
exclude = ['No.', 'Gender', 'Age', 'Payer Zone', 'Insurance Company', 
           'Treatment Type', 'Disease Category', 'Original Status', 
           'Outcome', 'TPA Remarks', 'Approval Rate', 'Approved Amt (₹)',
           'Age_clean', 'Insurer_Disease', 'Age_Group', 'Is_Rejected', 'Is_Partial']

feature_cols = [col for col in df.columns if col not in exclude and col != 'Coverage']
print(f"\n🔧 Checking {len(feature_cols)} features")

# Correlation with target
correlations = []
for col in feature_cols:
    corr = df[col].corr(df['Coverage'])
    correlations.append({'feature': col, 'correlation': corr})

corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)

print("\n📈 Top 15 features correlated with Coverage:")
print(corr_df.head(15).to_string(index=False))

print("\n📉 Bottom 15 features (negative correlation):")
print(corr_df.tail(15).to_string(index=False))

# Check for highly correlated features (redundancy)
corr_matrix = df[feature_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

high_corr = []
for col in upper.columns:
    for idx in upper.index:
        if upper.loc[idx, col] > 0.85:
            high_corr.append((idx, col, upper.loc[idx, col]))

print(f"\n🔴 Found {len(high_corr)} highly correlated pairs (>0.85):")
for i, (f1, f2, val) in enumerate(high_corr[:10]):  # Show first 10
    print(f"   {f1} ↔ {f2} : {val:.3f}")

# Save correlation matrix for tomorrow
corr_df.to_csv("data/processed/feature_correlations.csv", index=False)
print("\n💾 Saved correlations to data/processed/feature_correlations.csv")

# Quick visualization
plt.figure(figsize=(12, 8))
top_20 = corr_df.head(20)
plt.barh(top_20['feature'], top_20['correlation'], color='teal')
plt.xlabel('Correlation with Coverage')
plt.title('Top 20 Features by Correlation')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_correlations.png', dpi=150)
print("📊 Saved correlation plot to feature_correlations.png")