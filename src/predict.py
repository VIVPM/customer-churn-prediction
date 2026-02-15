"""
Prediction Module

Makes churn predictions for new customers:
- Single customer prediction
- Batch prediction from CSV
- Risk level classification

"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from config import MODELS_DIR, DATA_PROCESSED
from src.utils import load_model, load_dataframe


def load_prediction_artifacts():
    """Load model and feature information for prediction."""
    # Load best model (XGBoost by default, fall back to random_forest)
    model_path = MODELS_DIR / "xgboost.joblib"
    if not model_path.exists():
        model_path = MODELS_DIR / "random_forest.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError("No trained model found. Run train.py first.")
    
    model = load_model(model_path)
    print(f"Loaded model: {model_path.name}")
    
    # Load feature names
    try:
        features_df = load_dataframe(DATA_PROCESSED / "feature_names_selected.csv")
    except FileNotFoundError:
        features_df = load_dataframe(DATA_PROCESSED / "feature_names.csv")
    
    feature_names = features_df['feature'].tolist()
    
    return model, feature_names


def get_risk_level(probability):
    """
    Convert churn probability to risk level.
    
    Args:
        probability: Churn probability (0-1)
        
    Returns:
        str: Risk level (Low, Medium, High)
    """
    if probability >= 0.7:
        return "High"
    elif probability >= 0.4:
        return "Medium"
    else:
        return "Low"


def predict_single(customer_data: dict):
    """
    Predict churn for a single customer.
    
    Args:
        customer_data: Dictionary with customer features
        
    Returns:
        dict: Prediction results including probability and risk level
    """
    model, feature_names = load_prediction_artifacts()
    
    # Create DataFrame from input
    df = pd.DataFrame([customer_data])
    
    # Ensure all required columns exist
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match training data
    df = df[feature_names]
    
    # Make prediction
    prob = model.predict_proba(df)[0][1]
    pred = int(prob >= 0.5)
    risk_level = get_risk_level(prob)
    
    return {
        'churn_prediction': pred,
        'churn_probability': round(prob, 4),
        'risk_level': risk_level,
        'recommendation': get_recommendation(risk_level, prob)
    }


def get_recommendation(risk_level, probability):
    """
    Get retention recommendation based on risk level.
    
    Args:
        risk_level: Customer risk level
        probability: Churn probability
        
    Returns:
        str: Recommendation text
    """
    if risk_level == "High":
        return "Immediate action required! Contact customer with special retention offer."
    elif risk_level == "Medium":
        return "Monitor closely. Consider proactive engagement and loyalty rewards."
    else:
        return "Low risk. Continue regular engagement and satisfaction monitoring."


def predict_batch(filepath: str, output_filepath: str = None):
    """
    Predict churn for multiple customers from CSV.
    
    Args:
        filepath: Path to input CSV file
        output_filepath: Path to save results (optional)
        
    Returns:
        pd.DataFrame: DataFrame with predictions
    """
    model, feature_names = load_prediction_artifacts()
    
    # Load data
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} customers for prediction")
    
    # Store original columns
    original_cols = df.columns.tolist()
    
    # Ensure all required columns exist
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    # Select and reorder features
    X = df[feature_names]
    
    # Make predictions
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    
    # Add predictions to dataframe
    df['churn_probability'] = probs.round(4)
    df['churn_prediction'] = preds
    df['risk_level'] = df['churn_probability'].apply(get_risk_level)
    df['recommendation'] = df.apply(
        lambda row: get_recommendation(row['risk_level'], row['churn_probability']), 
        axis=1
    )
    
    # Summary statistics
    print("\n=== Prediction Summary ===")
    print(f"Total customers: {len(df)}")
    print(f"Predicted to churn: {preds.sum()} ({preds.mean()*100:.1f}%)")
    print(f"\nRisk Level Distribution:")
    print(df['risk_level'].value_counts())
    
    # Save results if output path provided
    if output_filepath:
        df.to_csv(output_filepath, index=False)
        print(f"\nResults saved to: {output_filepath}")
    
    return df


def interactive_prediction():
    """Interactive prediction mode for testing."""
    print("\n" + "=" * 50)
    print("CUSTOMER CHURN PREDICTION - INTERACTIVE MODE")
    print("=" * 50)
    
    model, feature_names = load_prediction_artifacts()
    print(f"\nRequired features: {len(feature_names)}")
    
    # Create a sample customer with default values
    print("\nCreating sample customer prediction...")
    
    sample_customer = {
        'CustomerID': 1001,
        'TransactionID': 5000,
        'AmountSpent': 250.0,
        'InteractionID': 3000,
        'LoginFrequency': 15,
        'TransactionYear': 2022,
    }
    
    # Add encoded categorical features (set to 0 by default)
    for feature in feature_names:
        if feature not in sample_customer:
            sample_customer[feature] = 0
    
    result = predict_single(sample_customer)
    
    print("\n=== Prediction Result ===")
    print(f"Churn Prediction: {'Yes' if result['churn_prediction'] else 'No'}")
    print(f"Churn Probability: {result['churn_probability']:.2%}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Recommendation: {result['recommendation']}")


if __name__ == "__main__":
    interactive_prediction()
