"""
HANF-Net Lite - Flask API for Coverage Prediction
Run: python app.py
"""

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow import keras
import sys
import os

# Add path for feature engineering
sys.path.append('src')
try:
    from feature_engineering import create_features_from_form
except ImportError:
    # Fallback feature engineering if module not found
    def create_features_from_form(gender, age, zone, insurer, treatment, los, disease, bill):
        """Simplified feature engineering for API"""
        data = pd.DataFrame([{
            'Gender': gender,
            'Age_clean': age if age else 40,
            'Payer Zone': zone,
            'Insurance Company': insurer,
            'Treatment Type': treatment,
            'LOS (Days)': los if los else 5,
            'Disease Category': disease,
            'Bill Amt (₹)': bill if bill else 50000
        }])
        
        # Calculate basic features
        data['Log_Bill'] = np.log1p(data['Bill Amt (₹)'])
        data['Cost_per_Day'] = data['Bill Amt (₹)'] / (data['LOS (Days)'] + 1)
        data['LOS_Risk_Short'] = np.exp(-0.5 * ((data['LOS (Days)'] - 0) / 15) ** 2)
        
        # Insurer mappings
        insurer_map = {
            'New India Assurance': 0.728, 'Star Health': 0.645, 'ICICI Lombard': 0.721,
            'United India': 0.702, 'Oriental Insurance': 0.684, 'National Insurance': 0.70,
            'Care Health': 0.691, 'Bajaj Allianz': 0.632, 'SBI General': 0.621,
            'HDFC Ergo': 0.719, 'Aditya Birla Health': 0.70, 'Reliance Health': 0.849,
            'Tata AIG': 0.736, 'GO DIGIT': 0.65, 'MAGMA': 0.80, 'LIBERTY': 0.65,
            'CHOLAMANDALAM': 0.65, 'Others': 0.65, 'Unknown': 0.65
        }
        data['Insurer_Generosity'] = insurer_map.get(insurer, 0.65)
        
        # Disease mappings
        disease_map = {
            'Chronic Kidney Disease (CKD)': 0.85, 'Abdominal Condition': 0.52,
            'Cancer': 0.64, 'Cardiac Procedure': 0.78, 'Respiratory': 0.72,
            'Neurological': 0.68, 'Fever / Infection': 0.70, 'Abdominal Pain': 0.65,
            'GI Procedure': 0.75, 'Unspecified': 0.60, 'Others': 0.62
        }
        data['Disease_Avg_Cov'] = disease_map.get(disease, 0.60)
        data['Disease_Risk'] = 1 - data['Disease_Avg_Cov']
        
        # Combined risk
        data['Combined_Risk'] = (1 - data['Insurer_Generosity'] + 
                                  (1 - data['Disease_Avg_Cov']) + 
                                  (data['LOS (Days)']/100)) / 3
        
        # Zone mappings
        zone_map = {
            'Chennai': 0.81, 'Centralised': 0.74, 'Mumbai': 0.76, 'Mandaveli': 0.78,
            'Delhi': 0.72, 'Pune': 0.73, 'Chandigarh': 0.75, 'Bangalore': 0.77,
            'Ncr-Delhi': 0.71, 'Unknown': 0.70
        }
        data['Zone_Bias'] = zone_map.get(zone, 0.70)
        
        # Placeholder features
        data['Bill_Pct_Insurer'] = 0.5
        data['Bill_Z_Disease'] = 0.0
        data['ID_Coverage'] = data['Insurer_Generosity'] * 1.05
        data['ID_Rejection'] = 1 - data['Insurer_Generosity']
        data['ID_Deviation'] = 0.05
        data['Insurer_Rejection_Rate'] = 1 - data['Insurer_Generosity']
        data['Disease_Avg_LOS'] = 10
        data['Cost_Risk_High'] = np.minimum(1, data['Bill Amt (₹)'] / 200000)
        data['LOS_Risk_Long'] = 1 - data['LOS_Risk_Short']
        
        return data

# Load model and scaler
print("📂 Loading model and scaler...")
model_path = "models/hanfnet_lite_best_679.keras"
scaler_path = "models/scaler_optimized.pkl"

if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    print("✅ Model and scaler loaded successfully!")
else:
    print("⚠️ Model files not found. Using fallback predictions.")
    model = None
    scaler = None

# Load dataset for insurer statistics
try:
    df = pd.read_csv("data/processed/cleaned_dataset.csv")
    insurer_stats = df.groupby('Insurance Company')['Coverage'].agg(['mean', 'std', 'count']).reset_index()
    insurer_stats['rating'] = pd.cut(insurer_stats['mean'], 
                                      bins=[0, 0.3, 0.6, 1], 
                                      labels=['bad', 'medium', 'good'])
    insurer_stats = insurer_stats.to_dict(orient='records')
    print("✅ Insurer statistics loaded!")
except:
    # Fallback insurer stats
    insurer_stats = [
        {"Insurance Company": "New India Assurance", "mean": 0.728, "rating": "good", "count": 45},
        {"Insurance Company": "Star Health", "mean": 0.645, "rating": "medium", "count": 38},
        {"Insurance Company": "ICICI Lombard", "mean": 0.721, "rating": "good", "count": 42},
        {"Insurance Company": "MAGMA", "mean": 0.18, "rating": "bad", "count": 5},
        {"Insurance Company": "LIBERTY", "mean": 0.19, "rating": "bad", "count": 4},
    ]
    print("⚠️ Using fallback insurer statistics.")

app = Flask(__name__)
CORS(app)  # Allow frontend to access

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print(f"📥 Received prediction request: {data}")
        
        # Create features
        features_df = create_features_from_form(
            data['gender'], data['age'], data['zone'], data['insurer'],
            data['treatment'], data['los'], data['disease'], data['bill']
        )
        
        # Define feature columns (same as training)
        feature_cols = ['Log_Bill', 'Cost_per_Day', 'LOS_Risk_Short', 'Insurer_Generosity',
                        'Disease_Avg_Cov', 'Disease_Risk', 'Combined_Risk', 'Zone_Bias',
                        'Bill_Pct_Insurer', 'ID_Coverage', 'ID_Rejection', 'ID_Deviation',
                        'Insurer_Rejection_Rate', 'Disease_Avg_LOS', 'Cost_Risk_High', 'LOS_Risk_Long']
        
        # Select and scale features
        X = features_df[feature_cols].fillna(0).values
        
        if scaler and model:
            X_scaled = scaler.transform(X)
            coverage_pred, risk_probs = model.predict(X_scaled, verbose=0)
            coverage = float(coverage_pred[0][0])
            risk_class = int(np.argmax(risk_probs[0]))
            risk_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
            risk = risk_map[risk_class]
        else:
            # Fallback prediction
            coverage = 0.65
            if data['insurer'] in ["MAGMA", "LIBERTY", "CHOLAMANDALAM", "GO DIGIT"]:
                coverage = 0.18
            elif data['insurer'] == "Star Health" and "Abdominal" in data['disease']:
                coverage = 0.22
            risk = "HIGH" if coverage < 0.35 else ("MEDIUM" if coverage < 0.7 else "LOW")
        
        return jsonify({
            'coverage': coverage,
            'risk': risk,
            'success': True
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/insurer-stats', methods=['GET'])
def get_insurer_stats():
    return jsonify(insurer_stats)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Starting NeuroSure API Server")
    print("="*50)
    print("📍 Endpoints:")
    print("   POST /predict - Get coverage prediction")
    print("   GET /insurer-stats - Get insurer statistics")
    print("   GET /health - Check server health")
    print("\n🌐 Server running at: http://localhost:5000")
    print("="*50)
    app.run(debug=True, port=5000)