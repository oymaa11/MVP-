import joblib
from src.backend.config import MODEL_PATH, FEATURE_COLUMNS_PATH
import numpy as np

try:
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
except Exception as e:
    model = None
    feature_columns = None
    print(f"Error loading model: {e}")

def predict_inventory(day):
    if model is None or feature_columns is None:
        raise Exception("Model not loaded")
    if day <= 0:
        raise ValueError("Day must be positive")
    
    X = np.array([[day]])
    prediction = model.predict(X)
    return float(prediction[0])
