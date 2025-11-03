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


# --- CORRECT, WORKING DATA URL (FULL LISTINGS) ---
@st.cache_data
def load_data():
    url = "http://data.insideairbnb.com/south-africa/wc/cape-town/2024-09-28/data/listings.csv.gz"
    df = pd.read_csv(url, compression='gzip', low_memory=False)

    # Select and clean
    cols = ['price', 'room_type', 'accommodates', 'bedrooms', 'bathrooms_text', 'neighbourhood_cleansed']
    df = df[cols].copy()
    df = df.dropna(subset=cols)

    # Clean price
    df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)

    # Extract bathrooms
    df['bathrooms'] = df['bathrooms_text'].str.extract(r'(\d+\.?\d*)').astype(float).fillna(1.0)
    df = df.drop('bathrooms_text', axis=1)

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

st.caption("Data: Inside Airbnb (Sep 2024) | Model: Random Forest | Live on Streamlit")