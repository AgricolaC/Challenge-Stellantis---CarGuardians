import pandas as pd

def load_data(train_path, test_path):
    "Load datasets and replace 'na' strings with np.nan."
    train = pd.read_csv(train_path, na_values='na')
    test = pd.read_csv(test_path, na_values='na')
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    return train, test
