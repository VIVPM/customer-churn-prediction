import os
import joblib

def save_model(model, filepath):
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    return joblib.load(filepath)

def create_directories(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)