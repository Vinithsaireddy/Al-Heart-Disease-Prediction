# AI-Based Heart Disease Prediction

An end-to-end Machine Learning pipeline built in Python to predict heart disease risk using the UCI Heart Disease Dataset.

## 🚀 Features
* **Automated Data Fetching**: Uses Kaggle API to retrieve the latest dataset.
* **High Accuracy**: Utilizes Random Forest Classifier achieving ~99% accuracy.
* **Data Preprocessing**: Includes Feature Scaling and Correlation Analysis.
* **Model Persistence**: Saves the trained model using `joblib` for production use.

## 🛠️ Setup
1. **Kaggle API**:
   - Go to Kaggle Settings -> Create New API Token.
   - Note down your `username` and `key`.
   - Update the `kaggle_credentials` dictionary in `heart_disease_ai.py`.

2. **Install Dependencies**:
   ```bash
   pip install pandas numpy scikit-learn seaborn matplotlib joblib kaggle
