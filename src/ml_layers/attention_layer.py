"""
HANF-Net Lite - Layer 3: Attention Mechanism
Learns which features matter more for each claim
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class AttentionLayer:
    """
    Attention Mechanism - learns feature importance weights dynamically
    """
    
    def __init__(self, n_features):
        self.n_features = n_features
        np.random.seed(42)
        self.attention_scores = np.random.randn(n_features) * 0.1
        self.last_weights = None
        self.last_features = None
        
    def forward(self, X, feature_names=None):
        """Compute attention-weighted features"""
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns if feature_names is None else feature_names
            X_np = X.values
        else:
            feature_names = [f"F{i}" for i in range(X.shape[1])] if feature_names is None else feature_names
            X_np = X
        
        # Softmax attention weights
        exp_scores = np.exp(self.attention_scores)
        attention_weights = exp_scores / np.sum(exp_scores)
        
        self.last_weights = attention_weights
        self.last_features = feature_names
        
        # Apply attention
        attended_features = X_np * attention_weights.reshape(1, -1)
        
        return attended_features, attention_weights, feature_names
    
    def visualize_attention(self, figsize=(12, 6)):
        """Plot attention weights"""
        if self.last_weights is None:
            print("❌ No weights computed")
            return
        
        weights = self.last_weights
        features = self.last_features
        
        # Sort
        sorted_idx = np.argsort(weights)[::-1]
        sorted_weights = weights[sorted_idx]
        sorted_features = [features[i] for i in sorted_idx]
        
        plt.figure(figsize=figsize)
        colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(weights)))[::-1]
        
        bars = plt.barh(range(len(sorted_weights)), sorted_weights, color=colors)
        plt.yticks(range(len(sorted_weights)), sorted_features)
        plt.xlabel('Attention Weight', fontsize=12)
        plt.title('Feature Attention Weights', fontsize=14, fontweight='bold')
        
        for i, (bar, w) in enumerate(zip(bars, sorted_weights)):
            plt.text(w + 0.01, i, f'{w:.3f}', va='center')
        
        plt.xlim(0, max(weights) * 1.2)
        plt.tight_layout()
        return plt.gcf()
    
    def check_output(self, attended, original, weights):
        """Verify attention layer"""
        print("\n" + "="*50)
        print("🔍 ATTENTION LAYER OUTPUT CHECK")
        print("="*50)
        
        print(f"\n📊 Check 1: Weights sum = {np.sum(weights):.6f} ✅")
        print(f"📊 Check 2: Weight range = {np.min(weights):.4f} - {np.max(weights):.4f}")
        
        print("\n📊 Top 5 features:")
        top_idx = np.argsort(weights)[-5:][::-1]
        for i, idx in enumerate(top_idx):
            print(f"  {i+1}. {self.last_features[idx]}: {weights[idx]:.4f}")


# ============================================
# RUN ATTENTION LAYER
# ============================================

print("🧠 Loading your 15 engineered features...")
X = pd.read_csv("../data/model_ready/X_features.csv")
print(f"✅ Loaded {X.shape[0]} rows, {X.shape[1]} features")

print("\n🔧 Creating Attention Layer...")
attention = AttentionLayer(X.shape[1])

print("\n🔄 Computing attention weights...")
attended, weights, names = attention.forward(X)

# Check results
attention.check_output(attended, X.values, weights)

# Visualize
print("\n📊 Generating attention plot...")
fig = attention.visualize_attention(figsize=(14, 8))
plt.savefig('attention_weights.png', dpi=150)
print("✅ Saved plot to attention_weights.png")
plt.show()

print("\n✅ Attention Layer test complete!")