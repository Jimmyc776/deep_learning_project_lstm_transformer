import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

def create_stock_dataloader(stock_csv: str, metadata_csv: str, seq_len: int=100, batch_size: int=32, seed: int=42,
                            stocks_per_bucket: int=13, train_per_bucket: int=10) -> dict:
    """
    Elite 15 -> 10 train + 3 eval split -> LSTM/Transformer ready tensors
    """

    np.random.seed(seed) # Reproducibility

    # 1. Load CSVs
    print("Loading stock data...")
    stock_data = pd.read_csv(stock_csv, index_col=0, parse_dates=True)
    metadata = pd.read_csv(metadata_csv)

    # 2. Top 13/buck -> random 10 train +. 3 eval
    train_tickers = []
    eval_tickers = []
    eval_details = []

    print(f"\nSplitting top {stocks_per_bucket} stocks per category...")
    for category in metadata['category'].unique():
        cat_data = metadata[metadata['category'] == category].sort_values('quality', ascending=False).head(stocks_per_bucket)

        train_idx = np.random.choice(stocks_per_bucket, train_per_bucket, replace=False)
        eval_idx = np.setdiff1d(np.arange(stocks_per_bucket), train_idx)

        train_tickers.extend(cat_data.iloc[train_idx]['ticker'].tolist())
        eval_tickers.extend(cat_data.iloc[eval_idx]['ticker'].tolist())
        eval_details.extend([f"{category}: {cat_data.iloc[i]['ticker']} (q={cat_data.iloc[i]['quality']:.4f})" for i in eval_idx])

    print(f"✅ Train: {len(train_tickers)} tickers | Eval: {len(eval_tickers)} tickers")
    print("Eval stocks:", eval_details)

    # 3. Train tensors [N, seq_len, 1]
    scalers = {}
    train_sequences = []
    train_targets = []
    print("\nCreating training sequences...")
    for ticker in train_tickers:
        series = stock_data[ticker].dropna().values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series)
        scalers[ticker] = scaler

        for i in range(seq_len, len(scaled)):
            train_sequences.append(scaled[i-seq_len:i])
            train_targets.append(scaled[i])

    train_X = torch.FloatTensor(np.array(train_sequences))  # [N, seq_len, 1]
    train_y = torch.FloatTensor(np.array(train_targets))    # [N, 1]

    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 4. Eval dataset (no loader - use for final metrics)
    eval_sequences = []
    eval_targets = []
    eval_scalers = {}

    for ticker in eval_tickers:
        series = stock_data[ticker].dropna().values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series)
        eval_scalers[ticker] = scaler

        for i in range(seq_len, len(scaled)):
            eval_sequences.append(scaled[i-seq_len:i])
            eval_targets.append(scaled[i])

    eval_X = torch.FloatTensor(np.array(eval_sequences))  # [N, seq_len, 1]
    eval_y = torch.FloatTensor(np.array(eval_targets))    # [N, 1]
    eval_dataset = TensorDataset(eval_X, eval_y)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    print(f"\n✅ Data loaded: {len(train_tickers)} train tickers, {len(eval_tickers)} eval tickers")

    return {
        'train_loader': train_loader,
        'train_dataset': train_dataset,
        'train_tickers': train_tickers,
        'train_scalers' : scalers,
        'eval_loader': eval_loader,
        'eval_dataset': eval_dataset,
        'eval_tickers': eval_tickers,
        'eval_scalers' : eval_scalers,
        'eval_details': eval_details,
        'seq_len' : seq_len
    }
