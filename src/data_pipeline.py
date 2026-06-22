import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime, date, timedelta
from dotenv import load_dotenv
from transformers import pipeline
import requests
import os

load_dotenv()

class StockMarketDataset(Dataset):
    """Sliding windows: X stacks past inputs; targets are forward cumulative **log** returns.

    Daily series ``self.returns`` is **natural log return** :math:`\\ln(P_t/P_{t-1})`.

    For dataset index ending with bar ``t``, each label ``y[..., j]`` at row ``lbl`` is the **sum**
    of the next ``H`` daily log returns starting that day,
    :math:`\\sum_{k=0}^{H-1} L_{\\text{lbl}+k} = \\ln(P_{\\text{lbl}+H-1}/P_{\\text{lbl}-1})`.

    Rows with incomplete **past** features are skipped via ``_warmup``. Rows lacking enough
    **future** returns for long horizons drop off the dataset via ``len`` (tail trim).

    Use ``input_feature_names`` / ``target_column_names`` for labels aligned with tensor columns.
    """

    def __init__(self, df, wanted_features, engineering_features,start_date, end_date=date.today().strftime("%Y-%m-%d"), seq_length=5, stride=1, ticker="VOO"):
        super().__init__()

        self.pipe = pipeline("text-classification", model="ProsusAI/finbert")
        self.headers = {"User-Agent": os.getenv("SEC_USER_AGENT")}

        self.stride = stride
        self.ticker = ticker
        self.wanted_features = wanted_features
        self.engineering_features = engineering_features

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

        close_f = close.astype(np.float64)
        open_f = open_.astype(np.float64)
        high_f = high.astype(np.float64)
        low_f = low.astype(np.float64)
        returns_close = (
            np.log(close_f).diff().replace([np.inf, -np.inf], np.nan).dropna()
        )
        returns_open = (
            np.log(open_f).diff().replace([np.inf, -np.inf], np.nan).dropna()
        )
        returns_high = (
            np.log(high_f).diff().replace([np.inf, -np.inf], np.nan).dropna()
        )
        returns_low = (
            np.log(low_f).diff().replace([np.inf, -np.inf], np.nan).dropna()
        )
        idx = returns_close.index
        body = (close - open_).reindex(idx)
        range_ = (high - low).reindex(idx)
        vol_chg = (
            (volume / volume.shift(1))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(1.0)
            .reindex(idx)
        )
        close_pos = ((close - open_) / (high - low)).reindex(idx)
        self.returns = returns_close
        self.returns_open = returns_open
        self.returns_high = returns_high
        self.returns_low = returns_low
        self.body = body
        self.range = range_
        self.vol_chg = vol_chg
        self.close_pos = close_pos

        # Forward cumulative log-return targets from prediction bar onward (see class docstring).
        self._fwd_target_specs = (
            ("fwd_tot_next_bar", 1),
            ("fwd_tot_1w", 5),
            ("fwd_tot_1m", 20)
        )
        self._fwd_horizon_max = max(h for _, h in self._fwd_target_specs)
        _fwd_totals = StockMarketDataset.forward_cumulative_log_returns_from_here(
            returns_close,
            tuple(h for _, h in self._fwd_target_specs),
        )
        self.cols_y = [_fwd_totals[h] for _, h in self._fwd_target_specs]
        self.target_column_names = tuple(name for name, _ in self._fwd_target_specs)
        assert len(self.target_column_names) == len(self.cols_y)

        self.ma_windows = (5, 10, 20)
        self._warmup = max(w - 1 for w in self.ma_windows)
        # Closing-price SMA (textbook Ma_N = mean of last N closes); aligned to idx like other cols_y
        self.moving_average_5 = self.get_moving_average(5, close).reindex(idx)
        self.moving_average_10 = self.get_moving_average(10, close).reindex(idx)
        self.moving_average_20 = self.get_moving_average(20, close).reindex(idx)
        self.exponential_moving_average_5 = self.get_exponential_moving_average(5, close).reindex(idx)
        self.exponential_moving_average_10 = self.get_exponential_moving_average(10, close).reindex(idx)
        self.exponential_moving_average_20 = self.get_exponential_moving_average(20, close).reindex(idx)

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

        # RSI
        self.rsi_14 = self.get_rsi(close, 14).reindex(idx)
        self.rsi_20 = self.get_rsi(close, 20).reindex(idx)
        self.rsi_30 = self.get_rsi(close, 30).reindex(idx)

        # MACD (tuple of Series — reindex each to returns timeline idx)
        self.macd, self.macd_signal, self.macd_hist = (
            s.reindex(idx) for s in self.get_macd(close)
        )

        # Momentum
        self.momentum_5d = self.get_momentum(close, 5).reindex(idx)
        self.momentum_20d = self.get_momentum(close, 20).reindex(idx)

        # Overnight gap: today's open minus prior session's close (aligned to returns idx).
        self.overnight_gap = (open_ - close.shift(1)).reindex(idx)

        # SEC data
        self.cik = self.cik_matching_ticker(self.ticker, self.headers)
        self.facts = self.get_facts(self.cik, self.headers)
        self.feature_data = self.get_wanted_features(
            self.facts,
            self.wanted_features,
            start_date,
            end_date
        )
        self.feature_data = self.get_engineering_features(
            self.feature_data,
            self.wanted_features,
            self.engineering_features
        )
        self.sec_daily_features = self.build_sec_daily_features(
            self.feature_data,
            self.engineering_features,
            idx
        )

        _x_extra_specs = (
            ("returns", self.returns),
            ("returns_open", self.returns_open),
            ("returns_high", self.returns_high),
            ("returns_low", self.returns_low),
            ("body", self.body),
            ("range", self.range),
            ("vol_chg", self.vol_chg),
            ("close_pos", self.close_pos),
            #("ma_5", self.moving_average_5),
            #("ma_10", self.moving_average_10),
            #("ma_20", self.moving_average_20),
            #("ema_5", self.exponential_moving_average_5),
            #("ema_10", self.exponential_moving_average_10),
            ("ema_20", self.exponential_moving_average_20),
            ("roll_vol_5", self.rolling_volatility_5),
            ("roll_vol_10", self.rolling_volatility_10),
            #("bb_mid", self.bollinger_bands_20),
            #("bb_upper", self.bollinger_bands_20_upper),
            #("bb_lower", self.bollinger_bands_20_lower),
            ("rsi_14", self.rsi_14),
            #("rsi_20", self.rsi_20),
            #("rsi_30", self.rsi_30),
            ("macd", self.macd),
            ("macd_signal", self.macd_signal),
            ("macd_hist", self.macd_hist),
            #("momentum_5d", self.momentum_5d),
            #("momentum_20d", self.momentum_20d),
            #("overnight_gap", self.overnight_gap),
            ("asset_growth_yoy", self.sec_daily_features["asset_growth_yoy"]),
            ("liability_growth_yoy", self.sec_daily_features["liability_growth_yoy"]),
            ("cash_growth_yoy", self.sec_daily_features["cash_growth_yoy"]),
            ("net_income_growth_yoy", self.sec_daily_features["net_income_growth_yoy"]),
            ("eps_growth_yoy", self.sec_daily_features["eps_growth_yoy"]),
            ("shares_growth_yoy", self.sec_daily_features["shares_growth_yoy"]),
            ("operating_cf_growth_yoy", self.sec_daily_features["operating_cf_growth_yoy"]),
        )
        self.cols_x = [series for _, series in _x_extra_specs]
        self.input_feature_names = tuple(
            name for name, _ in _x_extra_specs
        )
        assert len(self.input_feature_names) == len(self.cols_x)

    @property
    def n_features_in(self) -> int:
        return len(self.cols_x)

    @property
    def n_features_out(self) -> int:
        return len(self.cols_y)

    def __len__(self):
        usable = (
            len(self.returns)
            - self.seq_length
            - self._warmup
            - (self._fwd_horizon_max - 1)
        )
        if usable < 0:
            raise ValueError(
                "Not enough rows after warmup / forward horizons for seq_length; "
                "increase history, shorten horizons, or reduce seq_length / warmup."
            )
        return usable // self.stride

    @staticmethod
    def forward_cumulative_log_returns_from_here(daily_log_returns: pd.Series, horizons: tuple[int, ...]):
        """Cumulative **log** return over the **next H** bars: sum of daily log returns.

        At index ``t`` the value is :math:`\\sum_{k=0}^{H-1} L[t+k]` where ``L`` are daily log
        returns. That equals :math:`\\ln(P_{t+H-1}/P_{t-1})` for positive prices. For ``H=1`` this
        is a single-day log return. NaN where the forward window is incomplete.
        """
        horizons_u = tuple(sorted(set(int(h) for h in horizons)))
        ix = daily_log_returns.index
        v = np.asarray(daily_log_returns.to_numpy(dtype=float), dtype=np.float64)
        n = len(v)
        out = {}
        for H in horizons_u:
            if H < 1:
                raise ValueError("Horizon must be >= 1")
            cs = np.zeros(n + 1, dtype=np.float64)
            cs[1:] = np.cumsum(v)
            if H <= n:
                sums = cs[H:] - cs[:-H]
                arr = np.full(n, np.nan, dtype=np.float64)
                arr[: n - H + 1] = sums
                out[H] = pd.Series(arr, index=ix)
            else:
                out[H] = pd.Series(np.nan, index=ix)
        return out

    def get_moving_average(self, window_size, col):
        return col.rolling(window=window_size, min_periods=window_size).mean()

    def get_exponential_moving_average(self, span, col):
        """EMA on raw ``col`` (e.g. close), not on the SMA. Uses ``span=N`` (same N as MA window)."""
        return col.ewm(span=int(span), min_periods=int(span), adjust=False).mean()

    def get_rsi(self, close, period=14):
        delta = close.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(
            alpha=1/period,
            min_periods=period,
            adjust=False
        ).mean()

        avg_loss = loss.ewm(
            alpha=1/period,
            min_periods=period,
            adjust=False
        ).mean()

        rs = avg_gain / avg_loss

        rsi = 100 - (100 / (1 + rs))

        return rsi

    def get_macd(self, close, fast=12, slow=26, signal=9):

        ema_fast = close.ewm(
            span=fast,
            adjust=False
        ).mean()

        ema_slow = close.ewm(
            span=slow,
            adjust=False
        ).mean()

        macd = ema_fast - ema_slow

        macd_signal = macd.ewm(
            span=signal,
            adjust=False
        ).mean()

        macd_hist = macd - macd_signal

        return macd, macd_signal, macd_hist

    def get_momentum(self, close, period):
        return close.pct_change(period)

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

    def get_samples_info_sets(self):
        """
        Required for mlfinlab PurgedKFold.
        
        Returns
        -------
        pd.Series
            index = label-bar timestamp (when the forward return window starts)
            value = timestamp when the longest forward horizon is fully realized
        """
        starts = []
        ends = []

        for idx in range(len(self)):
            base = self._warmup + idx * self.stride
            lbl = base + self.seq_length

            start_time = self.returns.index[lbl]

            end_pos = (
                base
                + self.seq_length
                + self._fwd_horizon_max
                - 1
            )

            end_time = self.returns.index[end_pos]

            starts.append(start_time)
            ends.append(end_time)

        return pd.Series(ends, index=starts)

    def cik_matching_ticker(self, ticker, headers):
        ticker = ticker.upper().replace(".", "-")
        ticker_json = requests.get(
            "https://www.sec.gov/files/company_tickers.json", headers=headers
        ).json()

        for company in ticker_json.values():
            if company["ticker"] == ticker:
                cik = str(company["cik_str"]).zfill(10)
                return cik
        raise ValueError(f"Ticker {ticker} not found in SEC database")

    def get_facts(self, cik, headers):
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    
    def get_wanted_features(self, facts, wanted_features, start_date, end_date):
        start_date = pd.Timestamp(start_date) - pd.DateOffset(years=2)
        end_date = pd.Timestamp(end_date)

        feature_data = {}

        for feature in wanted_features:

            if feature not in facts["facts"]["us-gaap"]:
                continue

            concept = facts["facts"]["us-gaap"][feature]

            units = concept["units"]

            # Grab the first unit available
            unit_name = next(iter(units))

            df = pd.DataFrame(units[unit_name])

            cols = [
                c for c in [
                    "end",
                    "filed",
                    "form",
                    "val",
                    "frame",
                    "fy",
                    "fp"
                ]
                if c in df.columns
            ]

            feature_data[feature] = df[cols]

        for name, df in feature_data.items():
            tmp = df.copy()
            tmp["filed"] = pd.to_datetime(tmp["filed"])
            feature_data[name] = tmp[
                tmp["filed"].between(start_date, end_date)
            ]

        return feature_data

    def get_engineering_features(self, feature_data, wanted_features, engineering_features):
        for raw_feature, engineered_feature in zip(wanted_features, engineering_features):
            if raw_feature not in feature_data:
                continue

            df = feature_data[raw_feature].copy()

            df = df[df["form"].isin(["10-K", "10-Q"])]

            df["end"] = pd.to_datetime(df["end"])
            df["filed"] = pd.to_datetime(df["filed"])

            df = df.sort_values("filed")

            # keep one value per accounting period
            df = df.groupby("end", as_index=False).last()

            # sort by accounting period before pct_change
            df = df.sort_values("end")

            # YoY growth: 4 quarters back (works for quarterly 10-Q series)
            df[engineered_feature] = df["val"].pct_change(4)

            df = df.dropna(subset=[engineered_feature])

            feature_data[engineered_feature] = df[
                [
                    "end",
                    "filed",
                    engineered_feature
                ]
            ]

        return feature_data

    def build_sec_daily_features(self, feature_data, engineering_features, idx):
        sec_frames = []

        for feature in engineering_features:
            if feature not in feature_data:
                continue

            df = feature_data[feature].copy()
            df["filed"] = pd.to_datetime(df["filed"])

            df = df[["filed", feature]]
            sec_frames.append(df)

        if not sec_frames:
            return pd.DataFrame(index=idx)

        sec = sec_frames[0]

        for other in sec_frames[1:]:
            sec = pd.merge(sec, other, on="filed", how="outer")

        sec = sec.sort_values("filed")
        sec = sec.set_index("filed")

        sec = sec.groupby(sec.index).last()

        sec_daily = (
            sec
            .reindex(idx.union(sec.index))
            .sort_index()
            .ffill()
            .reindex(idx)
        )

        for feature in engineering_features:
            if feature not in sec_daily.columns:
                sec_daily[feature] = np.nan

        sec_daily = sec_daily[list(engineering_features)]

        sec_daily = sec_daily.fillna(0.0)

        return sec_daily

    def __getitem__(self, idx):
        base = self._warmup + idx * self.stride

        # Get the window of features
        X = np.stack(
            [col.iloc[base : base + self.seq_length].values for col in self.cols_x],
            axis=-1,
        )

        # Get the target (first value after the window)
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
