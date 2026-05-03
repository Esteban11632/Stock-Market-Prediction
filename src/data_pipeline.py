import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class StockMarketDataset(Dataset):
    """Sliding windows: X[t] stacks input channels; y is next-bar targets only.

    Input (`cols_x`) can be wider than output (`cols_y`): e.g. add MAs to X while
    still predicting only the core 5-vector for the next day.
    Rows with incomplete rolling-window features are skipped via ``_warmup``.
    """

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

        returns_s = close.pct_change().dropna()
        idx = returns_s.index
        body = (close - open_).reindex(idx)
        range_ = (high - low).reindex(idx)
        vol_chg = (
            (volume / volume.shift(1))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(1.0)
            .reindex(idx)
        )
        close_pos = ((close - open_) / (high - low)).reindex(idx)
        self.returns = returns_s
        self.body = body
        self.range = range_
        self.vol_chg = vol_chg
        self.close_pos = close_pos

        # Targets: next timestep of these channels only
        self.cols_y = [
            self.returns,
            self.body,
            self.range,
            self.vol_chg,
            self.close_pos,
        ]

        # Moving averages
        self.ma_windows = (5, 10, 20)
        self._warmup = max(self.ma_windows) - 1
        # Closing-price SMA (textbook Ma_N = mean of last N closes); aligned to idx like other cols_y
        self.moving_average_5 = self.get_moving_average(5, close).reindex(idx)
        self.moving_average_10 = self.get_moving_average(10, close).reindex(idx)
        self.moving_average_20 = self.get_moving_average(20, close).reindex(idx)

        # Rolling volatility
        self.rolling_volatility_5 = self.returns.rolling(window=5, min_periods=5).std()
        self.rolling_volatility_10 = self.returns.rolling(window=10, min_periods=10).std()

        # Bollinger Bands on close (20, ±2σ); align to idx for iloc stacking
        bb_mid = close.rolling(window=20, min_periods=20).mean()
        bb_std = close.rolling(window=20, min_periods=20).std()
        self.bollinger_bands_20 = bb_mid.reindex(idx)
        self.bollinger_bands_20_std = bb_std.reindex(idx)
        self.bollinger_bands_20_upper = (
            self.bollinger_bands_20 + 2 * self.bollinger_bands_20_std
        )
        self.bollinger_bands_20_lower = (
            self.bollinger_bands_20 - 2 * self.bollinger_bands_20_std
        )
        self.cols_x = list(self.cols_y) + [
            self.moving_average_5,
            self.moving_average_10,
            self.moving_average_20,
            self.rolling_volatility_5,
            self.rolling_volatility_10,
            self.bollinger_bands_20,
            self.bollinger_bands_20_upper,
            self.bollinger_bands_20_lower
        ]

    @property
    def n_features_in(self) -> int:
        return len(self.cols_x)

    @property
    def n_features_out(self) -> int:
        return len(self.cols_y)

    def __len__(self):
        usable = len(self.returns) - self.seq_length - self._warmup
        if usable < 0:
            raise ValueError(
                "Not enough rows after warmup for this seq_length; "
                "increase history or shorten seq_length / MA horizon."
            )
        return usable

    def get_moving_average(self, window_size, col):
        return col.rolling(window=window_size, min_periods=window_size).mean()

    def as_numpy(self):
        """Stack all windows into arrays for scaling / train_test_split.

        Returns
        -------
        X : np.ndarray, shape (N, seq_length, n_features_in)
        y : np.ndarray, shape (N, n_features_out)
        """
        n = len(self)
        X = np.stack([self[i][0].numpy() for i in range(n)], axis=0)
        y = np.stack([self[i][1].numpy() for i in range(n)], axis=0)
        return X, y

    def __getitem__(self, idx):
        base = idx + self._warmup
        X = np.stack(
            [col.iloc[base : base + self.seq_length].values for col in self.cols_x],
            axis=-1,
        )
        y = np.array(
            [col.iloc[base + self.seq_length] for col in self.cols_y],
            dtype=np.float64,
        )
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        return X_t, y_t


def split_data(dataset, train_size=0.8, val_size=0.1, test_size=0.1):
    """Chronological split. Test segment has length ``n - train - val`` (no dropped rows)."""
    _ = test_size  # third fraction ignored; test size is the remainder after train/val
    X, y = dataset.as_numpy()
    n = len(dataset)
    train_size = int(train_size * n)
    val_size = int(val_size * n)
    test_size = n - train_size - val_size
    if test_size < 0:
        raise ValueError("train_size + val_size must not exceed dataset length")
    train_idx = slice(0, train_size)
    val_idx = slice(train_size, train_size + val_size)
    test_idx = slice(train_size + val_size, train_size + val_size + test_size)
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    return X_train, X_val, X_test, y_train, y_val, y_test
