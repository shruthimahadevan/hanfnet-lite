"""
HANF-Net Lite - Data Loader
Load raw dataset from Excel/CSV
"""

import pandas as pd
import os

def load_raw_data(file_path=r"C:\Users\Shruthi B M\OneDrive\Desktop\HANF-Net-Lite\data\raw\oioooo.xlsx"):
    """
    Load the raw dataset
    Handles both CSV and Excel files
    """
    print(f"📂 Loading data from: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    # Detect file type
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, encoding="utf-8")

    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        df = pd.read_excel(file_path)   # no sheet name needed unless multiple sheets

    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")

    print(f"✅ Loaded {len(df)} claims")
    print(f"📊 Columns: {list(df.columns)}")
    return df

def preview_data(df, n=5):
    """Show first n rows"""
    print("\n👁️ Preview:")
    print(df.head(n))
    
def get_basic_info(df):
    """Get basic dataset info"""
    print("\n📊 Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    print(f"\nData Types:")
    print(df.dtypes)
    
    return {
        'shape': df.shape,
        'missing': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.to_dict()
    }

if __name__ == "__main__":
    # Test the loader
    df = load_raw_data()
    preview_data(df)
    get_basic_info(df)