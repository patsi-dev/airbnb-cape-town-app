import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline

def debug_model_prediction():
    """Debug the model prediction to identify data type issues"""
    
    # Load the model
    try:
        model = joblib.load("cape_town_model.pkl")
        print("✅ Model loaded successfully")
        print(f"Model type: {type(model)}")
        
        # Check if it's a pipeline
        if isinstance(model, Pipeline):
            print("Model is a Pipeline")
            for step_name, step in model.steps:
                print(f"Step: {step_name} -> {type(step)}")
        
        # Check feature names if available
        if hasattr(model, 'feature_names_in_'):
            print(f"Model expects {len(model.feature_names_in_)} features:")
            print(model.feature_names_in_)
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Create a test input with proper data types
    test_input = create_test_input()
    print(f"\n📊 Test input shape: {test_input.shape}")
    print(f"📊 Test input dtypes:")
    print(test_input.dtypes)
    
    # Check for any object/string columns that should be numeric
    object_columns = test_input.select_dtypes(include=['object']).columns
    if len(object_columns) > 0:
        print(f"\n⚠️ Object columns found: {list(object_columns)}")
    
    # Try prediction
    try:
        print("\n🎯 Attempting prediction...")
        prediction = model.predict(test_input)
        print(f"✅ Prediction successful: {prediction}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        print(f"Error type: {type(e)}")

def create_test_input():
    """Create a test input with proper data types"""
    
    # Landmark coordinates
    LANDMARKS = {
        'table_mountain': (-33.9628, 18.4099),
        'v_a_waterfront': (-33.9056, 18.4218),
        'camps_bay': (-33.9508, 18.3786),
        'clifton_beach': (-33.9375, 18.3800),
        'city_center': (-33.9258, 18.4232)
    }
    
    # Haversine function
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    # Create test data with EXPLICIT data types
    test_data = {
        # Property basics
        'property_type': ['Apartment'],  # Will be one-hot encoded
        'room_type': ['Entire home/apt'],  # Will be one-hot encoded
        'accommodates': np.array([4], dtype=np.float64),
        'bedrooms': np.array([2], dtype=np.float64),
        'beds': np.array([3], dtype=np.float64),
        'bathrooms': np.array([1.0], dtype=np.float64),
        
        # Location
        'neighbourhood_cleansed': ['City Bowl'],  # Will be one-hot encoded
        'latitude': np.array([-33.9258], dtype=np.float64),
        'longitude': np.array([18.4232], dtype=np.float64),
        
        # Host information
        'host_is_superhost': np.array([1], dtype=np.float64),
        'host_listings_count': np.array([2], dtype=np.float64),
        'host_acceptance_rate': np.array([90.0], dtype=np.float64),
        'hosting_years': np.array([3.0], dtype=np.float64),
        
        # Reviews
        'review_scores_rating': np.array([4.5], dtype=np.float64),
        'review_scores_location': np.array([4.5], dtype=np.float64),
        'number_of_reviews': np.array([25], dtype=np.float64),
        
        # Availability
        'availability_365': np.array([120], dtype=np.float64),
        'instant_bookable': np.array([1], dtype=np.float64),
        
        # Amenities
        'amenities_count': np.array([15], dtype=np.float64),
        'has_pool': np.array([0], dtype=np.float64),
        'has_bbq_grill': np.array([0], dtype=np.float64),
        'has_ocean_view': np.array([0], dtype=np.float64),
        'has_hot_tub': np.array([0], dtype=np.float64),
    }
    
    # Add distances
    latitude = -33.9258
    longitude = 18.4232
    for landmark, (lat, lon) in LANDMARKS.items():
        distance = haversine(latitude, longitude, lat, lon)
        test_data[f'distance_from_{landmark}'] = np.array([distance], dtype=np.float64)
    
    # Create DataFrame
    df = pd.DataFrame(test_data)
    
    return df

if __name__ == "__main__":
    debug_model_prediction()