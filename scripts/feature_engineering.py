import pandas as pd
from sklearn.preprocessing import StandardScaler

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate additional features for fraud detection."""
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['amount_log'] = df['amount'].apply(lambda x: np.log1p(x))
    return df

def scale_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Scale numerical features."""
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df
