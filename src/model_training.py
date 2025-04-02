import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def preprocess_data(df):
    # Convert 'date' and 'delivery_time' to datetime, coercing errors
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['delivery_time'] = pd.to_datetime(df['delivery_time'], errors='coerce')
    
    # Drop rows with invalid dates
    if df['date'].isnull().any() or df['delivery_time'].isnull().any():
        print("Warning: Some dates could not be parsed. Dropping rows with invalid dates.")
        df = df.dropna(subset=['date', 'delivery_time'])
    
    # Convert dates to numeric values
    df['date_ordinal'] = df['date'].map(pd.Timestamp.toordinal)
    df['delivery_time_ordinal'] = df['delivery_time'].map(pd.Timestamp.toordinal)
    
    # Create a new feature for the difference in days between delivery and date
    df['delivery_diff'] = (df['delivery_time'] - df['date']).dt.days
    
    # One-hot encode the 'item' column
    df = pd.get_dummies(df, columns=['item'])
    
    # Drop the original date columns
    df = df.drop(['date', 'delivery_time'], axis=1)
    return df

def main():
    # Define path to cleaned data file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '../data/cleaned_data.csv')
    
    # Load the cleaned dataset
    df = pd.read_csv(data_path)
    print("Data Head:")
    print(df.head())
    print("\nData Info:")
    print(df.info())
    
    # Preprocess the data
    df = preprocess_data(df)
    print("\nData after preprocessing:")
    print(df.head())
    print("\nData types after preprocessing:")
    print(df.dtypes)
    
    # Define target column; here, 'stock' is the inventory level to predict
    target_col = 'stock'
    if target_col not in df.columns:
        raise ValueError(f"The dataset must contain a '{target_col}' column as the target variable.")
    
    # Separate features (X) and target (y)
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Save the list of feature names for later use during prediction
    feature_columns = X.columns.tolist()
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"\nModel Evaluation:\nMSE: {mse}\nR² Score: {r2}")
    
    # Save the trained model and feature columns
    models_dir = os.path.join(base_dir, '../models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    model_path = os.path.join(models_dir, 'demand_forecast.pkl')
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save the feature columns used in training
    feature_columns_path = os.path.join(models_dir, 'feature_columns.pkl')
    joblib.dump(feature_columns, feature_columns_path)
    print(f"Feature columns saved to: {feature_columns_path}")

if __name__ == "__main__":
    main()
