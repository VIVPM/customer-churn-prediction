import pandas as pd
from config import DATA_RAW, RAW_DATA_FILE

def load_data():
    filepath = DATA_RAW / RAW_DATA_FILE
    df = pd.read_csv(filepath)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(df.info())