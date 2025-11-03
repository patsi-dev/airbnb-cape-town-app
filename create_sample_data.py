import pandas as pd
import numpy as np

def create_sample_listings():
    """Create a sample listings CSV with all required columns"""
    
    # Define all the columns your app expects
    columns = [
        'property_type', 'room_type', 'accommodates', 'bedrooms', 'beds', 'bathrooms',
        'neighbourhood_cleansed', 'latitude', 'longitude', 'host_is_superhost',
        'host_listings_count', 'host_acceptance_rate', 'review_scores_rating',
        'review_scores_location', 'number_of_reviews', 'availability_365', 'instant_bookable'
    ]
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'property_type': np.random.choice(['Apartment', 'House', 'Guesthouse', 'Condominium'], n_samples),
        'room_type': np.random.choice(['Entire home/apt', 'Private room', 'Shared room'], n_samples),
        'accommodates': np.random.randint(1, 10, n_samples),
        'bedrooms': np.random.randint(1, 5, n_samples),
        'beds': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0], n_samples),
        'neighbourhood_cleansed': np.random.choice(['City Bowl', 'Atlantic Seaboard', 'Southern Suburbs', 'False Bay', 'Northern Suburbs'], n_samples),
        'latitude': np.random.uniform(-33.8, -34.2, n_samples),
        'longitude': np.random.uniform(18.3, 18.6, n_samples),
        'host_is_superhost': np.random.choice([0, 1], n_samples),
        'host_listings_count': np.random.randint(1, 10, n_samples),
        'host_acceptance_rate': np.random.uniform(50, 100, n_samples),
        'review_scores_rating': np.random.uniform(3.5, 5.0, n_samples),
        'review_scores_location': np.random.uniform(3.5, 5.0, n_samples),
        'number_of_reviews': np.random.randint(0, 100, n_samples),
        'availability_365': np.random.randint(0, 365, n_samples),
        'instant_bookable': np.random.choice([0, 1], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv('data/sample_listings.csv', index=False)
    print(f"✅ Created sample_listings.csv with {len(df)} rows and {len(columns)} columns")
    print("Columns:", list(df.columns))
    
    return df

if __name__ == "__main__":
    create_sample_listings()