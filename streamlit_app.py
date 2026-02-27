import streamlit as st
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras
import os

st.set_page_config(page_title="NeuroSure", page_icon="🧠", layout="centered")

# Load your HTML file
with open('index.html', 'r', encoding='utf-8') as f:
    html_content = f.read()

# Load model and scaler
@st.cache_resource
def load_model():
    model = keras.models.load_model("models/hanfnet_lite_best_679.keras")
    scaler = joblib.load("models/scaler_optimized.pkl")
    return model, scaler

model, scaler = load_model()

# Custom CSS to hide Streamlit branding
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stApp {background: transparent;}
</style>
""", unsafe_allow_html=True)

# Display your HTML
st.components.v1.html(html_content, height=1200, scrolling=True)