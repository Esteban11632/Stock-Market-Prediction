import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class StockMarketDataset(Dataset):
    """Windows: each timestep [return, body, range, vol_change]. Targets: next return, body, range."""

    def __init__(self, df, seq_length):
        super().__init__()
        close = df["Close"].squeeze()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        open_ = df["Open"].squeeze()
        if isinstance(open_, pd.DataFrame):
            open_ = open_.iloc[:, 0]
        
        high = df["High"].squeeze()
        if isinstance(high, pd.DataFrame):
            high = high.iloc[:, 0]
        
        low = df["Low"].squeeze()
        if isinstance(low, pd.DataFrame):
            low = low.iloc[:, 0]
        
        volume = df["Volume"].squeeze()
        if isinstance(volume, pd.DataFrame):
            volume = volume.iloc[:, 0]

        self.seq_length = seq_length
        # Daily return (target / univariate X); first row NaN → dropped
        self.returns = close.pct_change().dropna()
        # Intraday body: close minus open, same calendar rows as close/open
        body = close - open_
        # Align with returns so windows share the same dates as self.returns
        self.body = body.reindex(self.returns.index)
        # Intraday range: high minus low, same calendar rows as close/open
        range = high - low
        self.range = range.reindex(self.returns.index)
        # Volume change: Volume_t / Volume_{t-1} (same calendar rows as returns)
        vol_chg = volume / volume.shift(1)
        vol_chg = vol_chg.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        self.vol_chg = vol_chg.reindex(self.returns.index)
        # Close position: close price minus open price, same calendar rows as close/open
        close_pos = (close - open_) / (high - low)
        self.close_pos = close_pos.reindex(self.returns.index)

    def __len__(self):
        return len(self.returns) - self.seq_length

    def as_numpy(self):
        """Stack all windows into arrays for scaling / train_test_split.

        Returns
        -------
        X : np.ndarray, shape (N, seq_length, 5)
        y : np.ndarray, shape (N, 5)
        """
        n = len(self)
        X = np.stack([self[i][0].numpy() for i in range(n)], axis=0)
        y = np.stack([self[i][1].numpy() for i in range(n)], axis=0)
        return X, y

    def __getitem__(self, idx):
        X_ret = self.returns.iloc[idx : idx + self.seq_length].values
        X_bod = self.body.iloc[idx : idx + self.seq_length].values
        X_ran = self.range.iloc[idx : idx + self.seq_length].values
        X_vol = self.vol_chg.iloc[idx : idx + self.seq_length].values
        X_pos = self.close_pos.iloc[idx : idx + self.seq_length].values

        # Shape of X: (seq_length, 5)
        X = np.stack([X_ret, X_bod, X_ran, X_vol, X_pos], axis=-1)
        y_ret = self.returns.iloc[idx + self.seq_length]
        y_bod = self.body.iloc[idx + self.seq_length]
        y_ran = self.range.iloc[idx + self.seq_length]
        y_vol = self.vol_chg.iloc[idx + self.seq_length]
        y_pos = self.close_pos.iloc[idx + self.seq_length]

        # Shape of y: (5,)
        y = np.stack([y_ret, y_bod, y_ran, y_vol, y_pos], axis=-1)

        # Shape of X_t: (seq_length, 5)
        X_t = torch.tensor(X, dtype=torch.float32)
        # Shape of y_t: (5,)
        y_t = torch.tensor(y, dtype=torch.float32)
        return X_t, y_t

def split_data(dataset, train_size=0.8, val_size=0.1, test_size=0.1):
    """Chronological split. Test segment has length ``n - train - val`` (no dropped rows)."""
    _ = test_size # third fraction ignored; test size is the remainder after train/val
    X, y = dataset.as_numpy()
    n = len(dataset)
    train_size = int(train_size * n)
    val_size = int(val_size * n)
    test_size = n - train_size - val_size
    if test_size < 0:
        raise ValueError("train_size + val_size must not exceed dataset length")
    train_idx = slice(0, train_size)
    val_idx = slice(train_size, train_size+val_size)
    test_idx = slice(train_size + val_size, train_size + val_size + test_size)
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    return X_train, X_val, X_test, y_train, y_val, y_test