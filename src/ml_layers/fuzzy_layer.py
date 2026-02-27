"""
HANF-Net Lite - Layer 2: Neuro-Fuzzy Layer
Converts continuous features to fuzzy membership values
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NeuroFuzzyLayer:
    """
    Neuro-Fuzzy Layer - converts continuous values to membership degrees
    Uses Gaussian membership functions: exp(-(x - c)² / (2σ²))
    """
    
    def __init__(self):
        self.membership_params = {}  # Store parameters for each feature
        self.feature_ranges = {}     # Store min/max for visualization
        
    def add_feature(self, feature_name, centers, sigmas):
        """
        Add a feature with its fuzzy membership functions
        
        Parameters:
        - feature_name: name of the feature (e.g., 'Log_Bill')
        - centers: list of center values for each membership (e.g., [low, medium, high])
        - sigmas: list of spread values for each membership
        """
        self.membership_params[feature_name] = {
            'centers': centers,
            'sigmas': sigmas,
            'n_memberships': len(centers)
        }
        print(f"  ✅ Added fuzzy feature: {feature_name} ({len(centers)} memberships)")
    
    def fit_ranges(self, X):
        """Store min/max of each feature for visualization"""
        for feature in self.membership_params.keys():
            if feature in X.columns:
                self.feature_ranges[feature] = {
                    'min': X[feature].min(),
                    'max': X[feature].max()
                }
    
    def transform(self, X):
        """
        Convert features to fuzzy membership values
        
        Returns:
        - fuzzy_outputs: DataFrame with membership values
        - membership_names: list of column names
        """
        fuzzy_outputs = pd.DataFrame(index=X.index)
        membership_names = []
        
        for feature, params in self.membership_params.items():
            if feature not in X.columns:
                print(f"  ⚠️ Warning: {feature} not found in data")
                continue
                
            values = X[feature].values
            centers = params['centers']
            sigmas = params['sigmas']
            
            # For each membership function
            for i, (c, s) in enumerate(zip(centers, sigmas)):
                # Gaussian membership: exp(-(x - c)² / (2σ²))
                membership = np.exp(-((values - c) ** 2) / (2 * s ** 2))
                
                # Name the column
                if i == 0:
                    suffix = "Low"
                elif i == 1:
                    suffix = "Medium"
                elif i == 2:
                    suffix = "High"
                else:
                    suffix = f"_{i}"
                    
                col_name = f"{feature}_Fuzzy_{suffix}"
                fuzzy_outputs[col_name] = membership
                membership_names.append(col_name)
        
        return fuzzy_outputs, membership_names
    
    def visualize_membership(self, feature_name, figsize=(10, 4)):
        """
        Plot the membership functions for a feature
        """
        if feature_name not in self.membership_params:
            print(f"❌ Feature {feature_name} not found")
            return
        
        params = self.membership_params[feature_name]
        centers = params['centers']
        sigmas = params['sigmas']
        
        # Create x values for plotting
        if feature_name in self.feature_ranges:
            x_min = self.feature_ranges[feature_name]['min']
            x_max = self.feature_ranges[feature_name]['max']
        else:
            x_min = min(centers) - 2 * max(sigmas)
            x_max = max(centers) + 2 * max(sigmas)
        
        x = np.linspace(x_min, x_max, 200)
        
        # Plot each membership
        plt.figure(figsize=figsize)
        colors = ['#2E7D32', '#ED6C02', '#C62828']  # Green, Orange, Red
        
        for i, (c, s) in enumerate(zip(centers, sigmas)):
            membership = np.exp(-((x - c) ** 2) / (2 * s ** 2))
            
            if i == 0:
                label = "Low"
            elif i == 1:
                label = "Medium"
            elif i == 2:
                label = "High"
            else:
                label = f"MF {i}"
                
            plt.plot(x, membership, color=colors[i % len(colors)], 
                    linewidth=2.5, label=label)
            plt.axvline(c, color=colors[i % len(colors)], 
                       linestyle='--', alpha=0.3)
        
        plt.xlabel(feature_name, fontsize=12)
        plt.ylabel('Membership Degree', fontsize=12)
        plt.title(f'Fuzzy Membership Functions - {feature_name}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        
        return plt.gcf()
    
    def check_output(self, fuzzy_outputs):
        """
        Check if fuzzy outputs are behaving correctly
        """
        print("\n" + "="*50)
        print("🔍 FUZZY LAYER OUTPUT CHECK")
        print("="*50)
        
        # Check 1: Values should be between 0 and 1
        print("\n📊 Check 1: Value ranges (should be 0-1)")
        for col in fuzzy_outputs.columns:
            min_val = fuzzy_outputs[col].min()
            max_val = fuzzy_outputs[col].max()
            status = "✅" if 0 <= min_val <= 1 and 0 <= max_val <= 1 else "❌"
            print(f"  {status} {col}: min={min_val:.3f}, max={max_val:.3f}")
        
        # Check 2: For each original feature, membership sum should be ≈1
        print("\n📊 Check 2: Membership sum per feature (should be ≈1)")
        features = set([col.split('_Fuzzy_')[0] for col in fuzzy_outputs.columns])
        
        for feature in features:
            feature_cols = [col for col in fuzzy_outputs.columns if col.startswith(feature)]
            if len(feature_cols) > 0:
                membership_sum = fuzzy_outputs[feature_cols].sum(axis=1)
                mean_sum = membership_sum.mean()
                std_sum = membership_sum.std()
                print(f"  {feature}: mean sum = {mean_sum:.3f} ± {std_sum:.3f}")
        
        # Check 3: Sample a few rows
        print("\n📊 Check 3: Sample rows (first 3)")
        sample = fuzzy_outputs.head(3)
        for idx, row in sample.iterrows():
            print(f"\n  Row {idx}:")
            for col in fuzzy_outputs.columns:
                if row[col] > 0.1:  # Only show significant memberships
                    print(f"    {col}: {row[col]:.3f}")
        
        return fuzzy_outputs.describe()


# ============================================
# FIXED: create_default_fuzzy_layer (SINGLE FUNCTION)
# ============================================

def create_default_fuzzy_layer(X=None):
    """
    Create fuzzy layer with parameters optimized for your data
    If X is provided, uses data-driven centers
    """
    fuzzy_layer = NeuroFuzzyLayer()
    
    # Feature 1: Log_Bill - Use data-driven centers if available
    if X is not None and 'Log_Bill' in X.columns:
        low_center = X['Log_Bill'].quantile(0.2)
        med_center = X['Log_Bill'].quantile(0.5)
        high_center = X['Log_Bill'].quantile(0.8)
        print(f"\n📊 Log_Bill centers from data: {low_center:.2f}, {med_center:.2f}, {high_center:.2f}")
    else:
        low_center, med_center, high_center = 10.5, 11.8, 13.0
    
    fuzzy_layer.add_feature(
        feature_name='Log_Bill',
        centers=[low_center, med_center, high_center],
        sigmas=[0.4, 0.6, 0.4]  # 🔥 CHANGED
    )
    
    # Feature 2: LOS_Risk_Short (0-1 scale)
    fuzzy_layer.add_feature(
        feature_name='LOS_Risk_Short',
        centers=[0.1, 0.5, 0.9],
        sigmas=[0.15, 0.2, 0.15]  # ✅ Keep same (was good)
    )
    
    # Feature 3: Combined_Risk - Use data-driven if available
    if X is not None and 'Combined_Risk' in X.columns:
        low_center = X['Combined_Risk'].quantile(0.2)
        med_center = X['Combined_Risk'].quantile(0.5)
        high_center = X['Combined_Risk'].quantile(0.8)
        print(f"📊 Combined_Risk centers from data: {low_center:.2f}, {med_center:.2f}, {high_center:.2f}")
    else:
        low_center, med_center, high_center = 0.2, 0.5, 0.8
    
    fuzzy_layer.add_feature(
        feature_name='Combined_Risk',
        centers=[low_center, med_center, high_center],
        sigmas=[0.12, 0.15, 0.12]  # 🔥 CHANGED
    )
    
    # Feature 4: Insurer_Generosity - Use data-driven if available
    if X is not None and 'Insurer_Generosity' in X.columns:
        low_center = X['Insurer_Generosity'].quantile(0.2)
        med_center = X['Insurer_Generosity'].quantile(0.5)
        high_center = X['Insurer_Generosity'].quantile(0.8)
        print(f"📊 Insurer_Generosity centers from data: {low_center:.2f}, {med_center:.2f}, {high_center:.2f}")
    else:
        low_center, med_center, high_center = 0.4, 0.6, 0.8
    
    fuzzy_layer.add_feature(
        feature_name='Insurer_Generosity',
        centers=[low_center, med_center, high_center],
        sigmas=[0.08, 0.12, 0.08]  # 🔥 CHANGED
    )
    
    return fuzzy_layer


# ============================================
# TESTING CODE
# ============================================

if __name__ == "__main__":
    print("🧪 Testing Neuro-Fuzzy Layer...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    sample_data = pd.DataFrame({
        'Log_Bill': np.random.normal(11.8, 1.0, n_samples),
        'LOS_Risk_Short': np.random.uniform(0, 1, n_samples),
        'Combined_Risk': np.random.uniform(0, 1, n_samples),
        'Insurer_Generosity': np.random.uniform(0.3, 0.9, n_samples)
    })
    
    # Create fuzzy layer WITH data-driven centers
    fuzzy = create_default_fuzzy_layer(sample_data)
    fuzzy.fit_ranges(sample_data)
    
    # Transform
    fuzzy_outputs, names = fuzzy.transform(sample_data)
    
    print(f"\n✅ Created {len(names)} fuzzy features:")
    for name in names:
        print(f"  - {name}")
    
    # Check outputs
    stats = fuzzy.check_output(fuzzy_outputs)
    
    # Visualize features
    for feature in ['Log_Bill', 'LOS_Risk_Short', 'Combined_Risk', 'Insurer_Generosity']:
        fuzzy.visualize_membership(feature)
        plt.savefig(f'fuzzy_{feature}.png', dpi=150)
        print(f"📊 Saved plot: fuzzy_{feature}.png")
    
    plt.show()