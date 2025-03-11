from src.model_training import train_model
import pandas as pd

def test_train_model():
    """Test model training with sample data."""
    data = pd.DataFrame({
        'amount': [100, 200, 300],
        'timestamp': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'label': [0, 1, 0]
    })
    X = data[['amount']]
    y = data['label']
    model = train_model(X, y)
    assert model is not None
