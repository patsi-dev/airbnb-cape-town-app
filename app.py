# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os

# Load model & data
@st.cache_resource
def load_model():
    path = 'cape_town_model.pkl'
    if not os.path.exists(path):
        st.error("❌ Model file missing! Run training cell first or place 'cape_town_model.pkl' in this folder.")
        st.stop()
    return joblib.load(path)

@st.cache_data
def load_data():
    path = 'listings_features.pkl'
    if not os.path.exists(path):
        st.error("❌ Data file missing! Run training cell first or place 'listings_features.pkl' in this folder.")
        st.stop()
    return joblib.load(path)

model = load_model()
df = load_data()

st.set_page_config(page_title="Cape Town Airbnb", layout="wide")
st.title("🏠 Cape Town Airbnb Price Predictor")

# Sidebar navigation
page = st.sidebar.radio("Go to", ["Predict Price", "Insights", "SHAP"])

# ---------------- PREDICT PAGE ---------------- #
if page == "Predict Price":
    st.markdown("### Enter Listing Details")
    with st.form("form"):
        c1, c2 = st.columns(2)
        with c1:
            acc = st.slider("Guests", 1, 16, 4)
            bed = st.slider("Bedrooms", 1, 10, 2)
            bath = st.slider("Bathrooms", 0.5, 6.0, 1.5, 0.5)
            amen = st.slider("Amenities", 0, 50, 12)
            dist = st.slider("Distance to Waterfront (km)", 0.0, 30.0, 5.0)
        with c2:
            room = st.selectbox("Room Type", df['room_type'].unique())
            nb = st.selectbox("Area (Neighbourhood)", df['neighbourhood_cleansed'].unique())
            superhost = st.checkbox("Superhost")
            season = st.checkbox("High Season (Dec–Feb)")
            pool = st.checkbox("Has Pool")

        if st.form_submit_button("🔮 Predict"):
            inp = pd.DataFrame([{
                'accommodates': acc,
                'bedrooms': bed,
                'bathrooms': bath,
                'amenity_count': amen,
                'dist_waterfront': dist,
                'room_type': room,
                'neighbourhood_cleansed': nb,
                'is_superhost': int(superhost),
                'high_season': int(season),
                'has_pool': int(pool)
            }])

            pred = np.expm1(model.predict(inp)[0])
            st.success(f"💰 **Estimated Price: R {pred:,.0f} per night**")

# ---------------- INSIGHTS PAGE ---------------- #
elif page == "Insights":
    st.markdown("### Market Insights")
    fig, ax = plt.subplots()
    sns.histplot(df['price'], bins=50, ax=ax, color='#2f5597')
    ax.set_title("Distribution of Airbnb Prices in Cape Town")
    ax.set_xlabel("Price (R)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# ---------------- SHAP PAGE ---------------- #
elif page == "SHAP":
    st.markdown("### 🔍 Model Explainability (SHAP Values)")
    X_sample = model.named_steps['prep'].transform(df.sample(100, random_state=42))
    explainer = shap.TreeExplainer(model.named_steps['model'])
    shap_vals = explainer.shap_values(X_sample)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_vals, X_sample, show=False)
    st.pyplot(fig)

st.caption("📊 Data: Inside Airbnb | 🧠 Model: XGBoost | 🌍 Project: Cape Town Airbnb Analysis")
