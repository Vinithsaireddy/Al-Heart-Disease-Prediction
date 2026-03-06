import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# STEP 1: KAGGLE API CONFIGURATION
# ==========================================
def setup_kaggle():
    """Manually creates the kaggle.json file for authentication."""
    # REPLACE THESE with your actual Kaggle details
    kaggle_credentials = {
        "username": "YOUR_KAGGLE_USERNAME", 
        "key": "YOUR_KAGGLE_API_KEY"
    }
    
    dot_kaggle = os.path.expanduser("~/.kaggle")
    os.makedirs(dot_kaggle, exist_ok=True)
    
    with open(os.path.join(dot_kaggle, "kaggle.json"), "w") as f:
        json.dump(kaggle_credentials, f)
    
    # Secure the file (required by Kaggle API)
    os.chmod(os.path.join(dot_kaggle, "kaggle.json"), 0o600)
    print("✅ Kaggle API configured.")

# ==========================================
# STEP 2: DATA ACQUISITION
# ==========================================
def load_data():
    """Downloads and loads the heart disease dataset."""
    dataset_path = 'heart.csv'
    
    # Download if file doesn't exist
    if not os.path.exists(dataset_path):
        print("📥 Downloading dataset from Kaggle...")
        os.system("kaggle datasets download -d johnsmith88/heart-disease-dataset")
        os.system("unzip -o heart-disease-dataset.zip")
    
    df = pd.read_csv(dataset_path)
    print(f"📊 Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# ==========================================
# STEP 3: TRAINING PIPELINE
# ==========================================
def train_model(df):
    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 80/20 Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    predictions = model.predict(X_test_scaled)
    print(f"\n✨ Model Accuracy: {accuracy_score(y_test, predictions)*100:.2f}%")
    print("\n📋 Classification Report:\n", classification_report(y_test, predictions))
    
    # Save artifacts
    joblib.dump(model, 'heart_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("💾 Model and Scaler saved locally.")
    
    return model, scaler

# ==========================================
# STEP 4: USER INPUT PREDICTION
# ==========================================
def manual_prediction(model, scaler):
    print("\n--- Clinical Input for Prediction ---")
    # Example input: [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    # Here we use a sample, but you can replace this with input() calls
    sample_data = [52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3] 
    
    input_scaled = scaler.transform([sample_data])
    result = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0][1]
    
    status = "Heart Disease Detected" if result[0] == 1 else "No Heart Disease"
    print(f"Prediction: {status} (Probability: {prob*100:.2f}%)")

if __name__ == "__main__":
    setup_kaggle()
    data = load_data()
    trained_model, trained_scaler = train_model(data)
    manual_prediction(trained_model, trained_scaler)
