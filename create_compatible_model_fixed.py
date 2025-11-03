# create_compatible_model_fixed.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import sklearn

print(f"Creating compatible model with scikit-learn: {sklearn.__version__}")

def create_compatible_model():
    # Define the 11 features your model expects
    numeric_features = ['accommodates', 'bedrooms', 'bathrooms', 'latitude', 'longitude', 
                       'review_scores_rating', 'number_of_reviews']
    categorical_features = ['property_type', 'room_type', 'neighbourhood_cleansed']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    
    # Use RandomForest for better compatibility than XGBoost
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10))
    ])
    
    # Create realistic training data
    np.random.seed(42)
    n_samples = 2000
    
    X_train = pd.DataFrame({
        'property_type': np.random.choice(['Apartment', 'House', 'Condominium', 'Guesthouse'], n_samples),
        'room_type': np.random.choice(['Entire home/apt', 'Private room', 'Shared room'], n_samples),
        'accommodates': np.random.randint(1, 10, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], n_samples),
        'neighbourhood_cleansed': np.random.choice(['City Bowl', 'Atlantic Seaboard', 'Southern Suburbs', 
                                                  'False Bay', 'Northern Suburbs'], n_samples),
        'latitude': np.random.uniform(-33.8, -34.2, n_samples),
        'longitude': np.random.uniform(18.3, 18.6, n_samples),
        'review_scores_rating': np.random.uniform(2.5, 5.0, n_samples),
        'number_of_reviews': np.random.randint(0, 200, n_samples),
    })
    
    # Create realistic price targets for Cape Town
    base_price = (800 + 
                 X_train['accommodates'] * 180 + 
                 X_train['bedrooms'] * 320 + 
                 X_train['bathrooms'] * 250 +
                 X_train['review_scores_rating'] * 120)
    
    # Adjust for property and room type (Cape Town specific)
    base_price = np.where(X_train['property_type'] == 'House', base_price * 1.3, base_price)
    base_price = np.where(X_train['property_type'] == 'Guesthouse', base_price * 1.2, base_price)
    base_price = np.where(X_train['room_type'] == 'Entire home/apt', base_price * 1.6, base_price)
    base_price = np.where(X_train['room_type'] == 'Shared room', base_price * 0.6, base_price)
    
    # Add neighborhood premium (Cape Town specific)
    neighborhood_multiplier = np.where(
        X_train['neighbourhood_cleansed'] == 'Atlantic Seaboard', 1.8,
        np.where(X_train['neighbourhood_cleansed'] == 'City Bowl', 1.5,
                np.where(X_train['neighbourhood_cleansed'] == 'Southern Suburbs', 1.2, 1.0))
    )
    base_price = base_price * neighborhood_multiplier
    
    y_train = base_price + np.random.normal(0, 300, n_samples)
    
    # Train the model
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, 'compatible_cape_town_model.pkl')
    
    # Test prediction
    test_input = X_train.iloc[[0]]
    prediction = model.predict(test_input)[0]
    
    print(f"✅ Compatible model created and saved!")
    print(f"✅ Test prediction: R {prediction:,.0f}")
    print(f"✅ Model uses scikit-learn version: {sklearn.__version__}")
    
    return model

if __name__ == "__main__":
    create_compatible_model()