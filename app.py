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

# --- Load data ---
@st.cache_data
def load_data():
    url = "http://data.insideairbnb.com/south-africa/wc/cape-town/2024-09-28/data/listings.csv.gz"
    df = pd.read_csv(url, compression='gzip', low_memory=False)
    df = df[['price','room_type','accommodates','bedrooms','bathrooms','neighbourhood_cleansed']].dropna()
    df['price'] = df['price'].replace('[\$,]','',regex=True).astype(float)
    return df

df = load_data()

# --- Train model ---
X = df.drop('price', axis=1)
y = np.log1p(df['price'])

numeric_features = ['accommodates','bedrooms','bathrooms']
categorical_features = ['room_type','neighbourhood_cleansed']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

model = Pipeline([
    ('prep', preprocessor),
    ('rf', RandomForestRegressor(n_estimators=200, random_state=42))
])
model.fit(X, y)

# --- Input ---
st.sidebar.header("Enter Listing Details")
guests = st.sidebar.slider("Guests", 1, 16, 4)
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 2)
bathrooms = st.sidebar.slider("Bathrooms", 0.5, 6.0, 1.0, 0.5)
room_type = st.sidebar.selectbox("Room Type", df['room_type'].unique())
area = st.sidebar.selectbox("Area", df['neighbourhood_cleansed'].unique())

if st.sidebar.button("Predict Price"):
    input_df = pd.DataFrame({
        'room_type': [room_type],
        'accommodates': [guests],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'neighbourhood_cleansed': [area]
    })
    pred_log = model.predict(input_df)
    price = np.expm1(pred_log)[0]
    st.metric("Estimated Price", f"R {price:,.0f} per night")
    st.success("Prediction complete!")

st.caption("Data: Inside Airbnb | Model: Random Forest")