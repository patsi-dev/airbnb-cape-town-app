# test_prediction.py
import pandas as pd
import numpy as np
import joblib

def test_prediction():
    """Test prediction with exact features"""
    
    # Load model
    model = joblib.load("cape_town_model.pkl")
    print("✅ Model loaded")
    print(f"Expected features: {model.feature_names_in_}")
    print(f"Number of features: {len(model.feature_names_in_)}")
    
    # Create test input with EXACT features
    test_data = {
        'property_type': ['Apartment'],
        'room_type': ['Entire home/apt'],
        'accommodates': np.array([4], dtype=np.float64),
        'bedrooms': np.array([2], dtype=np.float64),
        'bathrooms': np.array([1.0], dtype=np.float64),
        'neighbourhood_cleansed': ['City Bowl'],
        'latitude': np.array([-33.9258], dtype=np.float64),
        'longitude': np.array([18.4232], dtype=np.float64),
        'host_is_superhost': np.array([1], dtype=np.float64),
        'review_scores_rating': np.array([4.5], dtype=np.float64),
        'number_of_reviews': np.array([25], dtype=np.float64),
    }
    
    # Create DataFrame with exact features in correct order
    inp = pd.DataFrame({feature: test_data[feature] for feature in model.feature_names_in_})
    
    print(f"📊 Input shape: {inp.shape}")
    print(f"📊 Input columns: {list(inp.columns)}")
    print(f"📊 Input dtypes: {inp.dtypes.tolist()}")
    
    # Try prediction
    try:
        pred_log = model.predict(inp)
        price = np.expm1(pred_log)[0]
        print(f"🎯 Prediction successful: R {price:,.0f}")
        return True
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False

if __name__ == "__main__":
    test_prediction()