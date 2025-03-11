import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    return pd.read_csv(filepath)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the dataset."""
    df = df.drop_duplicates()
    df.fillna(method='ffill', inplace=True)
    return df
