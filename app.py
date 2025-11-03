import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Cape Town Airbnb", layout="wide")
st.title("🏠 Cape Town Airbnb Price Predictor")
st.markdown("**XGBoost — The Best Model — Live Predictions**")

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("cape_town_model.pkl")
        st.success("✅ Model loaded successfully")
        
        # Show model info
        if hasattr(model, 'feature_names_in_'):
            st.info(f"Model expects {len(model.feature_names_in_)} features")
            
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# Load UI data
@st.cache_data
def load_ui_data():
    try:
        df = pd.read_csv("data/sample_listings.csv")
        st.success(f"✅ Sample data loaded: {len(df)} rows")
        return df
    except Exception as e:
        st.error(f"❌ Error loading sample data: {e}")
        return pd.DataFrame()

df = load_ui_data()

# Safe function to get unique values
def get_unique_values(column_name, default_options):
    try:
        if column_name in df.columns and len(df[column_name].dropna()) > 0:
            unique_vals = df[column_name].dropna().unique()
            return unique_vals if len(unique_vals) > 0 else default_options
        else:
            return default_options
    except:
        return default_options

# Default options
DEFAULT_PROPERTY_TYPES = ['Apartment', 'House', 'Guesthouse', 'Condominium', 'Villa']
DEFAULT_ROOM_TYPES = ['Entire home/apt', 'Private room', 'Shared room', 'Hotel room']
DEFAULT_NEIGHBOURHOODS = ['City Bowl', 'Atlantic Seaboard', 'Southern Suburbs', 'False Bay', 'Northern Suburbs']

# User input - ONLY THE 11 FEATURES THE MODEL EXPECTS
st.sidebar.header("🏡 Property Details")

# Property Basics
col1, col2 = st.sidebar.columns(2)

with col1:
    property_type = st.selectbox(
        "Property Type", 
        get_unique_values('property_type', DEFAULT_PROPERTY_TYPES)
    )
    room_type = st.selectbox(
        "Room Type", 
        get_unique_values('room_type', DEFAULT_ROOM_TYPES)
    )
    accommodates = st.slider("Accommodates", 1, 16, 4)
    
with col2:
    bedrooms = st.slider("Bedrooms", 1, 10, 2)
    bathrooms = st.slider("Bathrooms", 0.5, 6.0, 1.0, 0.5)

# Location
st.sidebar.subheader("📍 Location")
neighbourhood = st.sidebar.selectbox(
    "Neighbourhood", 
    get_unique_values('neighbourhood_cleansed', DEFAULT_NEIGHBOURHOODS)
)

col1, col2 = st.sidebar.columns(2)
with col1:
    latitude = st.number_input("Latitude", value=-33.9258, format="%.4f")
with col2:
    longitude = st.number_input("Longitude", value=18.4232, format="%.4f")

# Host Information
st.sidebar.subheader("👤 Host Details")
host_is_superhost = st.sidebar.selectbox("Superhost", ['No', 'Yes'])

# Reviews
st.sidebar.subheader("⭐ Reviews")
review_scores_rating = st.sidebar.slider("Review Rating", 0.0, 5.0, 4.5, 0.1)
number_of_reviews = st.sidebar.slider("Number of Reviews", 0, 500, 25)

def create_input_with_exact_features():
    """Create input with EXACTLY the 11 features the model expects"""
    
    # Get the expected feature names from the model
    expected_features = model.feature_names_in_
    
    # Create data with ONLY the expected features
    input_data = {
        # Property basics - categorical
        'property_type': [property_type],
        'room_type': [room_type],
        'neighbourhood_cleansed': [neighbourhood],
        
        # Property basics - numeric
        'accommodates': np.array([accommodates], dtype=np.float64),
        'bedrooms': np.array([bedrooms], dtype=np.float64),
        'bathrooms': np.array([bathrooms], dtype=np.float64),
        
        # Location - numeric
        'latitude': np.array([latitude], dtype=np.float64),
        'longitude': np.array([longitude], dtype=np.float64),
        
        # Host information - numeric
        'host_is_superhost': np.array([1 if host_is_superhost == 'Yes' else 0], dtype=np.float64),
        
        # Reviews - numeric
        'review_scores_rating': np.array([review_scores_rating], dtype=np.float64),
        'number_of_reviews': np.array([number_of_reviews], dtype=np.float64),
    }
    
    # Create DataFrame with ONLY the expected features in the correct order
    df = pd.DataFrame({feature: input_data[feature] for feature in expected_features})
    
    return df

def predict_price_safe(inp):
    """Safe prediction with exact feature matching"""
    try:
        # Ensure we have exactly the right features in the right order
        expected_features = model.feature_names_in_
        
        # Check if we have all expected features
        missing_features = set(expected_features) - set(inp.columns)
        if missing_features:
            st.error(f"❌ Missing features: {missing_features}")
            return None, "error"
        
        # Reorder columns to match model expectation exactly
        inp = inp[expected_features]
        
        # Make prediction
        pred_log = model.predict(inp)
        price = np.expm1(pred_log)[0]
        
        return price, "success"
        
    except Exception as e:
        st.error(f"❌ Model prediction failed: {str(e)}")
        
        # Fallback calculation
        base_price = (800 + 
                     (accommodates * 150) + 
                     (bedrooms * 250) + 
                     (bathrooms * 200) +
                     (review_scores_rating * 100))
        
        if room_type == 'Entire home/apt':
            base_price *= 1.5
        elif room_type == 'Private room':
            base_price *= 1.0
        else:
            base_price *= 0.7
            
        return base_price, "fallback"

# Main app
st.write("### Enter property details in the sidebar and click 'Predict Price'")

# Display current inputs
with st.expander("📋 Current Input Summary", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Property**")
        st.write(f"- Type: {property_type}")
        st.write(f"- Room: {room_type}")
        st.write(f"- Guests: {accommodates}")
    with col2:
        st.write("**Rooms**")
        st.write(f"- Bedrooms: {bedrooms}")
        st.write(f"- Bathrooms: {bathrooms}")
        st.write(f"- Area: {neighbourhood}")
    with col3:
        st.write("**Host & Reviews**")
        st.write(f"- Superhost: {host_is_superhost}")
        st.write(f"- Rating: {review_scores_rating}/5")
        st.write(f"- Reviews: {number_of_reviews}")

# Prediction button
if st.sidebar.button("🚀 Predict Price", type="primary", use_container_width=True):
    if model is not None:
        with st.spinner("Calculating price prediction..."):
            # Create input with exact features
            inp = create_input_with_exact_features()
            
            # Debug information
            with st.expander("🔧 Debug Information", expanded=False):
                st.write("**Input DataFrame:**")
                st.dataframe(inp)
                st.write("**Data Types:**")
                st.write(inp.dtypes)
                st.write("**Input Shape:**", inp.shape)
                st.write("**Expected Features:**", len(model.feature_names_in_))
                st.write("**Actual Features:**", len(inp.columns))
                st.write("**Features:**", list(inp.columns))
            
            # Make prediction
            price, status = predict_price_safe(inp)
            
            if price is not None:
                # Display result
                st.success("### 🎯 Price Prediction")
                
                if status == "success":
                    st.metric("Predicted Price", f"R {price:,.0f}", delta="XGBoost Model")
                    st.balloons()
                else:
                    st.metric("Estimated Price", f"R {price:,.0f}", delta="Fallback Calculation")
                    st.info("ℹ️ Using fallback calculation - model prediction failed")
                
                # Price insights
                st.write("---")
                st.write("#### 💰 Price Insights")
                
                price_per_bedroom = price / bedrooms if bedrooms > 0 else price
                price_per_guest = price / accommodates if accommodates > 0 else price
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Price per Bedroom", f"R {price_per_bedroom:,.0f}")
                with col2:
                    st.metric("Price per Guest", f"R {price_per_guest:,.0f}")
                
    else:
        st.error("Please ensure the model file 'cape_town_model.pkl' exists and is valid.")

# Model information
with st.expander("ℹ️ Model Information"):
    if model is not None:
        st.write("**Model Details:**")
        st.write(f"- Model type: {type(model).__name__}")
        st.write(f"- Number of features: {len(model.feature_names_in_)}")
        st.write("**Expected Features:**")
        for i, feature in enumerate(model.feature_names_in_, 1):
            st.write(f"  {i}. {feature}")
        
        if isinstance(model, Pipeline):
            st.write("- Model is a scikit-learn Pipeline")
            for i, (name, step) in enumerate(model.steps):
                st.write(f"  {i+1}. {name}: {type(step).__name__}")
    else:
        st.write("No model loaded")

# Footer
st.write("---")
st.caption("Cape Town Airbnb Price Prediction | XGBoost Model | 11 Features")