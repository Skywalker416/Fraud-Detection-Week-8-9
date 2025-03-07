import pandas as pd

def predict_fraud(model, new_data: pd.DataFrame):
    """Use trained model to predict fraud."""
    return model.predict(new_data)
