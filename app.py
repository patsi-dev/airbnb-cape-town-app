# app.py
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ========= Compatibility shim for old scikit-learn pickles =========
try:
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(list): pass
        _ct._RemainderColsList = _RemainderColsList
    if not hasattr(_ct, "_RemainderCols"):
        class _RemainderCols:
            def __init__(self, cols): self.cols = cols
        _ct._RemainderCols = _RemainderCols
except Exception:
    pass
# ===================================================================

st.set_page_config(page_title="Cape Town Airbnb", layout="wide")
st.title("🏠 Cape Town Airbnb Price Predictor")
st.markdown("**Tuned XGBoost pipeline — Live predictions (log→price).**")

MODEL_PATH = os.getenv("CT_MODEL_PATH", "cape_town_model.pkl")
SAMPLE_PATH = os.getenv("CT_SAMPLE_PATH", "data/sample_listings.csv")

DEFAULT_PROPERTY_TYPES = ['Apartment', 'House', 'Guesthouse', 'Condominium', 'Villa']
DEFAULT_ROOM_TYPES = ['Entire home/apt', 'Private room', 'Shared room', 'Hotel room']
DEFAULT_NEIGHBOURHOODS = ['City Bowl', 'Atlantic Seaboard', 'Southern Suburbs', 'False Bay', 'Northern Suburbs']

# ✨ UPDATED: Include the two missing features below
UI_FEATURES = [
    'property_type', 'room_type', 'neighbourhood_cleansed',           # categorical
    'accommodates', 'bedrooms', 'beds', 'bathrooms',                   # numeric
    'latitude', 'longitude',                                            # numeric
    'host_listings_count', 'host_acceptance_rate', 'hosting_years',    # numeric
    'host_is_superhost',                                               # NEW (0/1)
    'review_scores_rating', 'review_scores_location',                   # numeric
    'number_of_reviews',                                               # NEW (int)
    'amenities_count', 'has_pool', 'has_bbq_grill', 'has_ocean_view', 'has_hot_tub',
    'instant_bookable'
]

@st.cache_resource
def load_model(path: str):
    try:
        model = joblib.load(path)
        return model, None
    except Exception as e:
        return None, str(e)

model, load_err = load_model(MODEL_PATH)

@st.cache_data
def load_ui_data(path: str):
    try:
        df = pd.read_csv(path)
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)

df_sample, sample_err = load_ui_data(SAMPLE_PATH)

try:
    import sklearn, xgboost
    st.caption(f"scikit-learn: {sklearn.__version__} | xgboost: {xgboost.__version__}")
except Exception:
    pass

if model is None:
    st.error(f"❌ Error loading model: {load_err or 'Unknown error'}")
else:
    st.success("✅ Model loaded")

if not df_sample.empty:
    st.info(f"📄 Sample data available: {len(df_sample)} rows")
else:
    st.warning(f"⚠️ Could not load sample data ({sample_err or 'missing file'}). UI will use defaults.")

def get_unique_values(column_name, defaults):
    if column_name in df_sample.columns:
        vals = df_sample[column_name].dropna().unique()
        if len(vals) > 0:
            return list(vals)
    return defaults

def as_float_array(x):
    return np.array([x], dtype=np.float64)

def as_int_array(x):
    return np.array([x], dtype=np.int64)

def safe_expected_features(model_obj):
    if hasattr(model_obj, "feature_names_in_"):
        return list(model_obj.feature_names_in_)
    return UI_FEATURES

# ============== Sidebar: Inputs ==============
st.sidebar.header("🏡 Property Details")

# Property basics
c1, c2 = st.sidebar.columns(2)
with c1:
    property_type = st.selectbox("Property Type", get_unique_values('property_type', DEFAULT_PROPERTY_TYPES))
    room_type = st.selectbox("Room Type", get_unique_values('room_type', DEFAULT_ROOM_TYPES))
    accommodates = st.slider("Accommodates", 1, 16, 4)
with c2:
    bedrooms = st.slider("Bedrooms", 0, 10, 2)
    beds = st.slider("Beds", 1, 16, 3)
    bathrooms = st.slider("Bathrooms", 0.0, 6.0, 1.0, 0.5)

# Location
st.sidebar.subheader("📍 Location")
neighbourhood = st.sidebar.selectbox("Neighbourhood", get_unique_values('neighbourhood_cleansed', DEFAULT_NEIGHBOURHOODS))
c3, c4 = st.sidebar.columns(2)
with c3:
    latitude = st.number_input("Latitude", value=-33.9258, format="%.6f")
with c4:
    longitude = st.number_input("Longitude", value=18.4232, format="%.6f")

# Host
st.sidebar.subheader("👤 Host")
c5, c6 = st.sidebar.columns(2)
with c5:
    host_listings_count = st.slider("Host Listings Count", 0, 200, 2)
    host_acceptance_rate = st.slider("Host Acceptance Rate %", 0, 100, 90)
with c6:
    hosting_years = st.slider("Hosting Years", 0.0, 30.0, 3.0, 0.5)
    host_is_superhost = st.selectbox("Superhost", ['No', 'Yes'])  # NEW

# Reviews
st.sidebar.subheader("⭐ Reviews")
c7, c8 = st.sidebar.columns(2)
with c7:
    review_scores_rating = st.slider("Review Rating (0–5)", 0.0, 5.0, 4.5, 0.1)
with c8:
    review_scores_location = st.slider("Location Rating (0–5)", 0.0, 5.0, 4.5, 0.1)
number_of_reviews = st.sidebar.slider("Number of Reviews", 0, 1000, 25)  # NEW

# Amenities & Booking
st.sidebar.subheader("🧺 Amenities & Booking")
instant_bookable = st.sidebar.selectbox("Instant Bookable", ['No', 'Yes'])
c9, c10 = st.sidebar.columns(2)
with c9:
    has_pool = st.checkbox("Has Pool", value=False)
    has_bbq_grill = st.checkbox("Has BBQ Grill", value=False)
with c10:
    has_ocean_view = st.checkbox("Has Ocean View", value=False)
    has_hot_tub = st.checkbox("Has Hot Tub", value=False)

amenities_count = st.sidebar.slider("Amenities Count", 0, 60, 15)

# ============== Input builder ==============
def build_input_row(expected_columns):
    input_data = {
        # Categorical
        'property_type': [property_type],
        'room_type': [room_type],
        'neighbourhood_cleansed': [neighbourhood],

        # Numeric
        'accommodates': as_float_array(accommodates),
        'bedrooms': as_float_array(bedrooms),
        'beds': as_float_array(beds),
        'bathrooms': as_float_array(bathrooms),

        'latitude': as_float_array(latitude),
        'longitude': as_float_array(longitude),

        'host_listings_count': as_float_array(host_listings_count),
        'host_acceptance_rate': as_float_array(float(host_acceptance_rate)),
        'hosting_years': as_float_array(hosting_years),

        # NEW: model expects 0/1
        'host_is_superhost': as_int_array(1 if host_is_superhost == 'Yes' else 0),

        'review_scores_rating': as_float_array(review_scores_rating),
        'review_scores_location': as_float_array(review_scores_location),

        # NEW: total reviews as integer
        'number_of_reviews': as_int_array(number_of_reviews),

        'amenities_count': as_float_array(amenities_count),
        'has_pool': as_int_array(1 if has_pool else 0),
        'has_bbq_grill': as_int_array(1 if has_bbq_grill else 0),
        'has_ocean_view': as_int_array(1 if has_ocean_view else 0),
        'has_hot_tub': as_int_array(1 if has_hot_tub else 0),

        'instant_bookable': as_int_array(1 if instant_bookable == 'Yes' else 0),
    }

    missing = [c for c in expected_columns if c not in input_data]
    if missing:
        raise ValueError(f"Input missing required features (expected by model): {missing}")

    return pd.DataFrame({c: input_data[c] for c in expected_columns})

def predict_price(model_obj, row_df):
    pred_log = model_obj.predict(row_df)
    pred_log = float(np.ravel(pred_log)[0])
    price = float(np.expm1(pred_log))
    return price, {"pred_log": pred_log}

# ============== Main: Input summary ==============
st.write("### Enter property details in the sidebar and click **Predict Price**")

with st.expander("📋 Current Input Summary", expanded=True):
    colA, colB, colC = st.columns(3)
    with colA:
        st.write("**Property**")
        st.write(f"- Type: {property_type}")
        st.write(f"- Room: {room_type}")
        st.write(f"- Guests: {accommodates}")
    with colB:
        st.write("**Rooms**")
        st.write(f"- Bedrooms: {bedrooms}")
        st.write(f"- Beds: {beds}")
        st.write(f"- Bathrooms: {bathrooms}")
    with colC:
        st.write("**Location & Host**")
        st.write(f"- Area: {neighbourhood}")
        st.write(f"- Host Listings: {host_listings_count}")
        st.write(f"- Superhost: {host_is_superhost}")  # NEW
        st.write(f"- Acceptance: {host_acceptance_rate}%")
    st.write(f"**Reviews:** {number_of_reviews} reviews | Rating {review_scores_rating}/5 | Location {review_scores_location}/5")  # NEW

# ============== Prediction ==============
if st.sidebar.button("🚀 Predict Price", type="primary", use_container_width=True):
    if model is None:
        st.error("Model not loaded. Check the .pkl path and scikit-learn version.")
    else:
        with st.spinner("Calculating price prediction…"):
            try:
                expected = safe_expected_features(model)
                inp = build_input_row(expected)

                with st.expander("🔧 Debug Information", expanded=False):
                    st.write("**Expected Features (order):**", expected)
                    st.write("**Input dtypes:**")
                    st.write(inp.dtypes)
                    st.write("**Input sample:**")
                    st.dataframe(inp)

                price, details = predict_price(model, inp)

                st.success("### 🎯 Price Prediction")
                st.metric("Predicted Price", f"R {price:,.0f}", delta="XGBoost (tuned)")
                st.balloons()

                st.write("---")
                st.write("#### 💰 Price Insights")
                pp_bedroom = price / max(1, bedrooms)
                pp_guest = price / max(1, accommodates)
                c11, c12 = st.columns(2)
                with c11:
                    st.metric("Price per Bedroom", f"R {pp_bedroom:,.0f}")
                with c12:
                    st.metric("Price per Guest", f"R {pp_guest:,.0f}")

            except Exception as e:
                st.error(f"❌ Model prediction failed: {e}")

                # Deterministic fallback
                base_price = (
                    800
                    + (accommodates * 150)
                    + (bedrooms * 250)
                    + (bathrooms * 200)
                    + (review_scores_rating * 100)
                    + (50 if host_is_superhost == 'Yes' else 0)
                )
                if room_type == 'Entire home/apt':
                    base_price *= 1.5
                elif room_type == 'Private room':
                    base_price *= 1.0
                else:
                    base_price *= 0.7

                st.metric("Estimated Price (Fallback)", f"R {base_price:,.0f}")
                st.info("ℹ️ Using fallback estimate because the model prediction failed.")

# ============== Model info ==============
with st.expander("ℹ️ Model Information"):
    if model is not None:
        st.write(f"- Object type: `{type(model).__name__}`")
        if hasattr(model, "feature_names_in_"):
            st.write(f"- Number of raw input features expected by estimator: **{len(model.feature_names_in_)}**")
            for i, f in enumerate(model.feature_names_in_, 1):
                st.write(f"  {i}. {f}")
        else:
            st.write("- Estimator does not expose `feature_names_in_`; app used the app's UI feature set.")
        try:
            from sklearn.pipeline import Pipeline
            if isinstance(model, Pipeline):
                st.write("- Detected scikit-learn Pipeline steps:")
                for i, (name, step) in enumerate(model.steps):
                    st.write(f"  {i+1}. **{name}**: `{type(step).__name__}`")
        except Exception:
            pass
    else:
        st.write("No model loaded.")

st.write("---")
st.caption("Cape Town Airbnb Price Prediction | Tuned XGBoost Pipeline | 22 UI Features (incl. superhost & review count)")
