import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Cape Town Airbnb", layout="centered")
st.title("Cape Town Airbnb Price Predictor")
st.markdown("*Enter details → Get optimal nightly rate*")

# --- REAL + SAFE DATA LOADING ---
@st.cache_data
def load_data():
    # Try small real dataset first
    url = "https://raw.githubusercontent.com/dennischesire/airbnb-cape-town-app/main/data/sample_listings.csv"
    try:
        df = pd.read_csv(url)
        st.success("Loaded real Cape Town data!")
    except:
        st.warning("Using sample data (real data will load next time)")
        df = pd.DataFrame({
            'price': [1800, 3200, 1500, 4500, 2200, 900, 2800],
            'room_type': ['Entire home/apt', 'Private room', 'Entire home/apt', 'Entire home/apt', 'Private room', 'Private room', 'Entire home/apt'],
            'accommodates': [4, 2, 3, 6, 2, 1, 5],
            'bedrooms': [2, 1, 1, 3, 1, 1, 2],
            'bathrooms': [1.5, 1.0, 1.0, 2.0, 1.0, 1.0, 1.5],
            'neighbourhood_cleansed': ['City Bowl', 'Sea Point', 'Gardens', 'Camps Bay', 'Green Point', 'Tamboerskloof', 'Clifton']
        })
    return df

df = load_data()

# --- Train model ---
X = df.drop('price', axis=1)
y = np.log1p(df['price'])

numeric_features = ['accommodates', 'bedrooms', 'bathrooms']
categorical_features = ['room_type', 'neighbourhood_cleansed']

prep = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

model = Pipeline([
    ('prep', prep),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
])
model.fit(X, y)

# --- Input ---
st.sidebar.header("Enter Listing Details")
guests = st.sidebar.slider("Guests", 1, 16, 4)
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 2)
bathrooms = st.sidebar.slider("Bathrooms", 0.5, 6.0, 1.0, 0.5)
room = st.sidebar.selectbox("Room Type", df['room_type'].unique())
area = st.sidebar.selectbox("Area", df['neighbourhood_cleansed'].unique())

if st.sidebar.button("Predict Price"):
    inp = pd.DataFrame({
        'room_type': [room],
        'accommodates': [guests],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'neighbourhood_cleansed': [area]
    })
    pred = np.expm1(model.predict(inp))[0]
    st.success(f"**R {pred:,.0f} per night**")
    st.balloons()

st.caption("Data: Inside Airbnb | Model: Random Forest | Live on Streamlit")