from __future__ import annotations

import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class StockMarketDataset(Dataset):
    """Windows of Close returns (raw). After fitting scaler on train indices, use `build_scaled_tensors`."""

    def __init__(self, df, seq_length):
        super().__init__()
        close = df["Close"].squeeze()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        self.seq_length = seq_length
        self.r = close.pct_change().dropna()

    def __len__(self):
        return len(self.r) - self.seq_length

    def __getitem__(self, idx):
        X = self.r.iloc[idx : idx + self.seq_length].values
        y = self.r.iloc[idx + self.seq_length]
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        return X_t, y_t


def fit_scaler_returns(base: StockMarketDataset, train_indices) -> StandardScaler:
    """Fit StandardScaler on training windows only (flattened returns)."""
    X_stack = np.stack([base[i][0].numpy() for i in train_indices])
    scaler = StandardScaler()
    scaler.fit(X_stack.reshape(-1, 1))
    return scaler


def build_scaled_tensors(
    base: StockMarketDataset, indices, scaler: StandardScaler
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Transform raw windows once → float tensors for TensorDataset / DataLoader.
    Avoids calling scaler.transform inside __getitem__ every epoch.
    """
    X_raw = np.stack([base[i][0].numpy() for i in indices])
    y_raw = np.stack([base[i][1].numpy() for i in indices]).squeeze(-1)
    X_s = scaler.transform(X_raw.reshape(-1, 1)).reshape(X_raw.shape).astype(np.float32)
    y_s = scaler.transform(y_raw.reshape(-1, 1)).astype(np.float32).ravel()
    X_t = torch.from_numpy(X_s)
    y_t = torch.from_numpy(y_s).unsqueeze(-1)
    return X_t, y_t
