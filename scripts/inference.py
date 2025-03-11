import joblib
import pandas as pd

def predict_new_data(new_data: pd.DataFrame):
    """Load model and make predictions."""
    model = joblib.load('../models/fraud_model.pkl')
    return model.predict(new_data)
