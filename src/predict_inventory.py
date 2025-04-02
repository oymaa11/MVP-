import os
import pandas as pd
import joblib

def preprocess_data(df):
    # Convert 'date' and 'delivery_time' to datetime, coercing errors
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['delivery_time'] = pd.to_datetime(df['delivery_time'], errors='coerce')
    
    # Drop rows with invalid dates
    if df['date'].isnull().any() or df['delivery_time'].isnull().any():
        print("Warning: Some dates could not be parsed. Dropping rows with invalid dates.")
        df = df.dropna(subset=['date', 'delivery_time'])
    
    # Create numeric features
    df['date_ordinal'] = df['date'].map(pd.Timestamp.toordinal)
    df['delivery_time_ordinal'] = df['delivery_time'].map(pd.Timestamp.toordinal)
    df['delivery_diff'] = (df['delivery_time'] - df['date']).dt.days
    
    # One-hot encode the 'item' column
    df = pd.get_dummies(df, columns=['item'])
    
    # Drop original date columns
    df = df.drop(['date', 'delivery_time'], axis=1)
    return df

def load_model():
    # Load the pre-trained model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, '../models/demand_forecast.pkl')
    model = joblib.load(model_path)
    return model

def load_feature_columns():
    # Load the feature columns saved during training
    base_dir = os.path.dirname(os.path.abspath(__file__))
    feature_columns_path = os.path.join(base_dir, '../models/feature_columns.pkl')
    feature_columns = joblib.load(feature_columns_path)
    return feature_columns

def predict_inventory(input_data):
    """
    Predict inventory levels using the pre-trained Linear Regression model.
    
    Args:
      input_data: A pandas DataFrame containing raw input data.
      
    Returns:
      An array of predicted inventory levels.
    """
    model = load_model()
    feature_columns = load_feature_columns()
    
    # Preprocess the input data
    df_processed = preprocess_data(input_data.copy())
    
    # Remove target column if present
    if 'stock' in df_processed.columns:
        df_processed = df_processed.drop('stock', axis=1)
    
    # Ensure the processed DataFrame has all the feature columns used during training,
    # adding any missing ones with a value of 0.
    df_processed = df_processed.reindex(columns=feature_columns, fill_value=0)
    
    predictions = model.predict(df_processed)
    return predictions

if __name__ == "__main__":
    # For testing: load a sample row from the cleaned data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '../data/cleaned_data.csv')
    df = pd.read_csv(data_path)
    
    print("Sample Raw Input:")
    print(df.iloc[[0]])
    
    # Use the first row as a sample input for prediction
    sample_input = df.iloc[[0]].copy()
    prediction = predict_inventory(sample_input)
    print("\nPredicted Inventory Level:")
    print(prediction)
