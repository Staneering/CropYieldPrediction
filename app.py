import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import json
import os
from flask import Flask, request, jsonify

# Load the saved model and preprocessing objects
MODEL_PATH = 'crop_yield_model.pkl'
ENCODER_PATH = 'label_encoders.pkl'
SCALER_PATH = 'scaler.pkl'

# Load model and preprocessing objects
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    with open(ENCODER_PATH, 'rb') as f:
        encoders = pickle.load(f)
        
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    model = None
    encoders = None
    scaler = None

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for making yield predictions"""
    if not model or not encoders or not scaler:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get input data
        data = request.get_json()
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Preprocess the input data (same as training preprocessing)
        processed_df = preprocess_input(input_df)
        
        # Make prediction
        prediction = model.predict(processed_df)
        
        # Return prediction
        return jsonify({
            'predicted_yield': float(prediction[0]),
            'units': 'hg/ha'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def preprocess_input(input_df):
    """Preprocess input data in the same way as training data"""
    # Make a copy to avoid modifying original
    df = input_df.copy()
    
    # Convert to numeric where needed
    numeric_cols = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Encode categorical variables
    categorical_cols = ['Area', 'Item']
    for col in categorical_cols:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])
        else:
            # If new category not seen during training, use most common
            df[col] = 0
    
    # Scale numerical features
    features_to_scale = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
    df[features_to_scale] = scaler.transform(df[features_to_scale])
    
    return df

def train_model():
    """Function to train and save the model (would be called separately)"""
    # Load and preprocess data
    df = pd.read_csv('yield_df.csv')
    
    # Clean data (as shown in notebook)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df.drop_duplicates(inplace=True)
    
    # Remove non-numeric rain values
    def isStr(obj):
        try:
            float(obj)
            return False
        except:
            return True
    
    to_drop = df[df['average_rain_fall_mm_per_year'].apply(isStr)].index
    df = df.drop(to_drop)
    df['average_rain_fall_mm_per_year'] = df['average_rain_fall_mm_per_year'].astype(np.float64)
    
    # Encode categorical variables
    encoders = {}
    categorical_cols = ['Area', 'Item']
    for col in categorical_cols:
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col])
    
    # Scale numerical features
    scaler = StandardScaler()
    features_to_scale = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    # Split data (assuming hg/ha_yield is the target)
    X = df.drop('hg/ha_yield', axis=1)
    y = df['hg/ha_yield']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model (example with Random Forest - replace with your actual model)
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model and preprocessing objects
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(encoders, f)
        
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Model training complete and artifacts saved")

if __name__ == '__main__':
    # For production, just run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
    
    # To train the model (run separately):
    # train_model()