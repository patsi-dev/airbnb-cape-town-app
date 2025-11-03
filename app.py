import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Cape Town Airbnb", layout="centered")
st.title("Cape Town Airbnb Price Predictor")
st.markdown("**XGBoost — The Best Model — Live Predictions**")

# --- LOAD YOUR XGBoost MODEL & FEATURES ---
@st.cache_resource
def load_model():
    model = joblib.load("cape_town_model.pkl")  # Your trained XGBoost
    features = joblib.load("listings_features.pkl")  # Feature list
    return model, features

model, feature_cols = load_model()

# --- LOAD REAL DATA FOR UI (dropdowns) ---
@st.cache_data
def load_ui_data():
    df = pd.read_csv("data/sample_listings.csv")
    return df

df = load_ui_data()

# --- USER INPUT ---
st.sidebar.header("Enter Listing Details")

guests = st.sidebar.slider("Guests", 1, 16, 4)
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 2)
bathrooms = st.sidebar.slider("Bathrooms", 0.5, 6.0, 1.0, 0.5)
room_type = st.sidebar.selectbox("Room Type", df['room_type'].unique())
area = st.sidebar.selectbox("Area", df['neighbourhood_cleansed'].unique())

# --- BUILD INPUT DATAFRAME ---
input_data = {
    'room_type': room_type,
    'accommodates': guests,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'neighbourhood_cleansed': area
}

# Fill missing features (required by model) with 0
for col in feature_cols:
    if col not in input_data:
        input_data[col] = 0

# Match exact column order
inp = pd.DataFrame([input_data])[feature_cols]

# --- PREDICT ---
if st.sidebar.button("Predict Price"):
    pred_log = model.predict(inp)
    price = np.expm1(pred_log)[0]
    st.success(f"**R {price:,.0f} per night**")
    st.balloons()

# --- FOOTER ---
st.caption("**Model: XGBoost (Tuned)** | Data: Inside Airbnb | Live on Streamlit")