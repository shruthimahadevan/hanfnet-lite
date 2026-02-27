"""
HANF-Net Lite - Feature Engineering
Creates ALL 35 features for coverage prediction
"""

import pandas as pd
import numpy as np
import os

def load_cleaned_data():
    """Load the cleaned dataset"""
    df = pd.read_csv("data/processed/cleaned_dataset.csv")
    print(f"✅ Loaded {len(df)} cleaned claims")
    return df

def engineer_all_features(df):
    """
    Create ALL 35 features across 7 categories
    """
    print("\n" + "="*60)
    print("🔧 ENGINEERING ALL 35 FEATURES")
    print("="*60)
    
    data = df.copy()
    
    # ============================================
    # CATEGORY A: Financial Dynamics (8 features)
    # ============================================
    print("\n📊 Category A: Financial Dynamics")
    
    # A1: Bill Z-score per Insurer
    insurer_stats = data.groupby('Insurance Company')['Bill Amt (₹)'].agg(['mean', 'std']).fillna(0)
    data['Bill_Z_Insurer'] = data.apply(
        lambda row: (row['Bill Amt (₹)'] - insurer_stats.loc[row['Insurance Company'], 'mean']) / 
                   max(insurer_stats.loc[row['Insurance Company'], 'std'], 1), axis=1
    )
    data['Bill_Z_Insurer'] = data['Bill_Z_Insurer'].clip(-3, 3)
    print("  ✅ A1: Bill_Z_Insurer")
    
    # A2: Bill Z-score per Disease
    disease_stats = data.groupby('Disease Category')['Bill Amt (₹)'].agg(['mean', 'std']).fillna(0)
    data['Bill_Z_Disease'] = data.apply(
        lambda row: (row['Bill Amt (₹)'] - disease_stats.loc[row['Disease Category'], 'mean']) / 
                   max(disease_stats.loc[row['Disease Category'], 'std'], 1), axis=1
    )
    data['Bill_Z_Disease'] = data['Bill_Z_Disease'].clip(-3, 3)
    print("  ✅ A2: Bill_Z_Disease")
    
    # A3: LOS Z-score per Disease
    los_stats = data.groupby('Disease Category')['LOS (Days)'].agg(['mean', 'std']).fillna(0)
    data['LOS_Z_Disease'] = data.apply(
        lambda row: (row['LOS (Days)'] - los_stats.loc[row['Disease Category'], 'mean']) / 
                   max(los_stats.loc[row['Disease Category'], 'std'], 1), axis=1
    )
    data['LOS_Z_Disease'] = data['LOS_Z_Disease'].clip(-3, 3)
    print("  ✅ A3: LOS_Z_Disease")
    
    # A4: Cost-to-LOS Ratio
    data['Cost_per_Day'] = data['Bill Amt (₹)'] / (data['LOS (Days)'] + 1)  # +1 to avoid division by zero
    data['Cost_per_Day'] = data['Cost_per_Day'].clip(0, data['Cost_per_Day'].quantile(0.99))
    print("  ✅ A4: Cost_per_Day")
    
    # A5: Bill Percentile in Insurer
    data['Bill_Pct_Insurer'] = data.groupby('Insurance Company')['Bill Amt (₹)'].rank(pct=True)
    print("  ✅ A5: Bill_Pct_Insurer")
    
    # A6: Bill Percentile in Disease
    data['Bill_Pct_Disease'] = data.groupby('Disease Category')['Bill Amt (₹)'].rank(pct=True)
    print("  ✅ A6: Bill_Pct_Disease")
    
    # A7: Log Bill Amount
    data['Log_Bill'] = np.log1p(data['Bill Amt (₹)'])
    print("  ✅ A7: Log_Bill")
    
    # A8: Bill squared (normalized)
    data['Bill_Sq_Norm'] = (data['Bill Amt (₹)'] / 1e5) ** 2
    data['Bill_Sq_Norm'] = data['Bill_Sq_Norm'].clip(0, 10)
    print("  ✅ A8: Bill_Sq_Norm")
    
    # ============================================
    # CATEGORY B: Insurer Behavior (7 features)
    # ============================================
    print("\n📊 Category B: Insurer Behavior")
    
    # B1: Insurer Generosity Index
    insurer_coverage = data.groupby('Insurance Company')['Coverage'].mean()
    data['Insurer_Generosity'] = data['Insurance Company'].map(insurer_coverage)
    print("  ✅ B1: Insurer_Generosity")
    
    # B2: Insurer Consistency Score (lower std = more consistent)
    insurer_std = data.groupby('Insurance Company')['Coverage'].std()
    data['Insurer_Consistency'] = data['Insurance Company'].map(insurer_std)
    print("  ✅ B2: Insurer_Consistency")
    
    # B3: Insurer Claim Volume
    insurer_count = data['Insurance Company'].value_counts()
    data['Insurer_Volume'] = data['Insurance Company'].map(insurer_count)
    print("  ✅ B3: Insurer_Volume")
    
    # B4: Insurer Rejection Rate
    data['Is_Rejected'] = (data['Outcome'] == '❌ Rejected').astype(int)
    insurer_rejection = data.groupby('Insurance Company')['Is_Rejected'].mean()
    data['Insurer_Rejection_Rate'] = data['Insurance Company'].map(insurer_rejection)
    print("  ✅ B4: Insurer_Rejection_Rate")
    
    # B5: Insurer Partial Rate
    data['Is_Partial'] = (data['Outcome'] == '◑  Partial').astype(int)
    insurer_partial = data.groupby('Insurance Company')['Is_Partial'].mean()
    data['Insurer_Partial_Rate'] = data['Insurance Company'].map(insurer_partial)
    print("  ✅ B5: Insurer_Partial_Rate")
    
    # B6: Insurer CKD Generosity
    ckd_data = data[data['Disease Category'].str.contains('CKD|Chronic Kidney', na=False)]
    if len(ckd_data) > 0:
        ckd_insurer_coverage = ckd_data.groupby('Insurance Company')['Coverage'].mean()
        data['Insurer_CKD_Gen'] = data['Insurance Company'].map(ckd_insurer_coverage)
    else:
        data['Insurer_CKD_Gen'] = data['Insurer_Generosity']
    print("  ✅ B6: Insurer_CKD_Gen")
    
    # B7: Insurer Abdominal Generosity
    abd_data = data[data['Disease Category'].str.contains('Abdominal', na=False)]
    if len(abd_data) > 0:
        abd_insurer_coverage = abd_data.groupby('Insurance Company')['Coverage'].mean()
        data['Insurer_Abd_Gen'] = data['Insurance Company'].map(abd_insurer_coverage)
    else:
        data['Insurer_Abd_Gen'] = data['Insurer_Generosity']
    print("  ✅ B7: Insurer_Abd_Gen")
    
    # ============================================
    # CATEGORY C: Disease Dynamics (6 features)
    # ============================================
    print("\n📊 Category C: Disease Dynamics")
    
    # C1: Disease Risk Index
    disease_rejection = data.groupby('Disease Category')['Is_Rejected'].mean()
    data['Disease_Risk'] = data['Disease Category'].map(disease_rejection)
    print("  ✅ C1: Disease_Risk")
    
    # C2: Disease Avg Coverage
    disease_coverage = data.groupby('Disease Category')['Coverage'].mean()
    data['Disease_Avg_Cov'] = data['Disease Category'].map(disease_coverage)
    print("  ✅ C2: Disease_Avg_Cov")
    
    # C3: Disease Avg LOS
    disease_los = data.groupby('Disease Category')['LOS (Days)'].mean()
    data['Disease_Avg_LOS'] = data['Disease Category'].map(disease_los)
    print("  ✅ C3: Disease_Avg_LOS")
    
    # C4: Disease Avg Bill
    disease_bill = data.groupby('Disease Category')['Bill Amt (₹)'].mean()
    data['Disease_Avg_Bill'] = data['Disease Category'].map(disease_bill)
    print("  ✅ C4: Disease_Avg_Bill")
    
    # C5: Disease Variability
    disease_coverage_std = data.groupby('Disease Category')['Coverage'].std()
    data['Disease_Variability'] = data['Disease Category'].map(disease_coverage_std)
    print("  ✅ C5: Disease_Variability")
    
    # C6: Disease Sample Size
    disease_count = data['Disease Category'].value_counts()
    data['Disease_Volume'] = data['Disease Category'].map(disease_count)
    print("  ✅ C6: Disease_Volume")
    
    # ============================================
    # CATEGORY D: Insurer-Disease Interactions (4 features)
    # ============================================
    print("\n📊 Category D: Insurer-Disease Interactions")
    
    # Create combo key
    data['Insurer_Disease'] = data['Insurance Company'] + "_" + data['Disease Category']
    
    # D1: Insurer-Disease Coverage
    combo_coverage = data.groupby('Insurer_Disease')['Coverage'].mean()
    data['ID_Coverage'] = data['Insurer_Disease'].map(combo_coverage)
    print("  ✅ D1: ID_Coverage")
    
    # D2: Insurer-Disease Rejection
    combo_rejection = data.groupby('Insurer_Disease')['Is_Rejected'].mean()
    data['ID_Rejection'] = data['Insurer_Disease'].map(combo_rejection)
    print("  ✅ D2: ID_Rejection")
    
    # D3: Insurer-Disease Volume
    combo_count = data['Insurer_Disease'].value_counts()
    data['ID_Volume'] = data['Insurer_Disease'].map(combo_count)
    print("  ✅ D3: ID_Volume")
    
    # D4: Insurer-Disease Deviation
    data['ID_Deviation'] = data['ID_Coverage'] - data['Insurer_Generosity']
    print("  ✅ D4: ID_Deviation")
    
    # ============================================
    # CATEGORY E: Demographic & Zone (4 features)
    # ============================================
    print("\n📊 Category E: Demographic & Zone")
    
    # E1: Zone Approval Bias
    zone_coverage = data.groupby('Payer Zone')['Coverage'].mean()
    data['Zone_Bias'] = data['Payer Zone'].map(zone_coverage)
    print("  ✅ E1: Zone_Bias")
    
    # E2: Age Group
    bins = [0, 30, 60, 120]
    labels = ['Young', 'Middle', 'Senior']
    data['Age_Group'] = pd.cut(data['Age_clean'], bins=bins, labels=labels)
    age_group_map = {'Young': 0, 'Middle': 1, 'Senior': 2}
    data['Age_Group_Encoded'] = data['Age_Group'].map(age_group_map).fillna(1)
    print("  ✅ E2: Age_Group_Encoded")
    
    # E3: Gender Encoded
    gender_map = {'Male': 0, 'Female': 1, 'Unknown': 0.5}
    data['Gender_Encoded'] = data['Gender'].map(gender_map).fillna(0.5)
    print("  ✅ E3: Gender_Encoded")
    
    # E4: Zone × Insurer Interaction
    zone_insurer = data.groupby(['Payer Zone', 'Insurance Company'])['Coverage'].mean()
    data['Zone_Insurer'] = data.apply(
        lambda row: zone_insurer.get((row['Payer Zone'], row['Insurance Company']), row['Zone_Bias']), 
        axis=1
    )
    print("  ✅ E4: Zone_Insurer")
    
    # ============================================
    # CATEGORY F: Fuzzy-Inspired Risk Bands (4 features)
    # ============================================
    print("\n📊 Category F: Fuzzy Risk Bands")
    
    # F1: Cost Risk Band (Low/Medium/High)
    bill_mean = data['Bill Amt (₹)'].mean()
    bill_std = data['Bill Amt (₹)'].std()
    
    data['Cost_Risk_Low'] = np.exp(-0.5 * ((data['Bill Amt (₹)'] - bill_mean) / (bill_std/2)) ** 2)
    data['Cost_Risk_Medium'] = np.exp(-0.5 * ((data['Bill Amt (₹)'] - bill_mean) / bill_std) ** 2)
    data['Cost_Risk_High'] = 1 - data['Cost_Risk_Low']
    print("  ✅ F1: Cost Risk Bands (3 features)")
    
    # F2: LOS Risk Band
    los_mean = data['LOS (Days)'].mean()
    los_std = data['LOS (Days)'].std()
    
    data['LOS_Risk_Short'] = np.exp(-0.5 * ((data['LOS (Days)'] - 0) / (los_std/2)) ** 2)
    data['LOS_Risk_Optimal'] = np.exp(-0.5 * ((data['LOS (Days)'] - los_mean) / los_std) ** 2)
    data['LOS_Risk_Long'] = 1 - data['LOS_Risk_Optimal']
    print("  ✅ F2: LOS Risk Bands (3 features)")
    
    # F3: Combined Risk Score
    data['Combined_Risk'] = (data['Cost_Risk_High'] + data['LOS_Risk_Long'] + 
                             (1 - data['Insurer_Generosity']) + data['Disease_Risk']) / 4
    print("  ✅ F3: Combined_Risk")
    
    # ============================================
    # FINAL COUNT
    # ============================================
    print("\n" + "="*60)
    
    # Get all original columns plus the ones we want to exclude
    original_cols = list(df.columns)
    exclude_cols = original_cols + ['Insurer_Disease', 'Age_Group']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    print(f"✅ TOTAL FEATURES CREATED: {len(feature_cols)}")
    print("="*60)
    
    return data, feature_cols

def save_features(data, output_path="data/processed/final_features.csv"):
    """Save the feature-engineered dataset"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"\n💾 Saved feature-engineered data to: {output_path}")

if __name__ == "__main__":
    # Load cleaned data
    df = pd.read_csv("data/processed/cleaned_dataset.csv")
    
    # Engineer all features
    df_features, feature_list = engineer_all_features(df)
    
    # Save
    save_features(df_features)
    
    print("\n📋 Feature List:")
    for i, feat in enumerate(feature_list, 1):
        print(f"  {i:2d}. {feat}")