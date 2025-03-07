from sklearn.preprocessing import StandardScaler

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate new features for fraud detection."""
    df['amount_log'] = np.log1p(df['amount'])
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    return df

def scale_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Scale numeric features for better model performance."""
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df
