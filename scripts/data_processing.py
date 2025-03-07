import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """Load transaction data from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform necessary data cleaning and transformations."""
    df = df.drop_duplicates()
    df.fillna(method="ffill", inplace=True)
    return df
