import pandas as pd
import joblib

def load_data(path="data/heart.csv"):
    return pd.read_csv(path)

def save_model(model, path):
    joblib.dump(model, path)