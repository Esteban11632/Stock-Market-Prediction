import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mplfinance as mpf

from data_pipeline import StockMarketDataset

ticker = "VOO"

df = yf.download(ticker, start="2020-01-01")
print(df.head())

def s(df, name):
    s = df[name]
    return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s

def ohlcv_frame(df):
    ohlcv = pd.DataFrame(
        {
            "Open": s(df, "Open").astype(float),
            "High": s(df, "High").astype(float),
            "Low": s(df, "Low").astype(float),
            "Close": s(df, "Close").astype(float),
            "Volume": s(df, "Volume").astype(float),
        }
    )
    ohlcv.index = pd.to_datetime(ohlcv.index)
    return ohlcv

ohlcv = ohlcv_frame(df)

mpf.plot(
    ohlcv,
    type="candle",
    volume=True,
    style="yahoo",
    title=f"{ticker} — daily candlesticks + volume",
    figsize=(12, 6),
    warn_too_much_data=10_000,
)

# --- Derived features (same definitions as data_pipeline.StockMarketDataset) ---
seq_length = 10
ds = StockMarketDataset(df, seq_length)
t = ds.returns.index

fig2, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
rows = [
    (ds.returns, "Daily return (pct change)", "return"),
    (ds.body, "Intraday body (Close - Open)", "body"),
    (ds.range, "Intraday range (High - Low)", "range"),
    (ds.vol_chg, r"Volume change ($V_t / V_{t-1}$)", "vol. change"),
    (ds.close_pos, "Close position in range: (C-O) / (H-L)", "close pos."),
]
for ax, (ser, title, ylbl) in zip(axes, rows):
    ax.plot(t, ser, color="tab:blue", linewidth=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylbl)
    ax.grid(True, alpha=0.3)
    if ylbl in ("return", "body", "close pos."):
        ax.axhline(0.0, color="gray", linewidth=0.4, alpha=0.6)
    if ylbl == "vol. change":
        ax.axhline(1.0, color="gray", linewidth=0.4, alpha=0.6)

axes[-1].set_xlabel("Date")
fig2.suptitle(f"{ticker} — five model features aligned to return dates", y=1.01, fontsize=12)
fig2.tight_layout()
plt.show()

# --- Separate chart: raw returns vs rolling mean of returns (training MAs) ---
fig_ma, ax_ma = plt.subplots(figsize=(12, 4))
ax_ma.plot(t, ds.returns, color="black", linewidth=0.6, alpha=0.85, label="return")
for ser, w, c in [
    (ds.moving_average_5, 5, "tab:blue"),
    (ds.moving_average_10, 10, "tab:orange"),
    (ds.moving_average_20, 20, "tab:green")
]:
    ax_ma.plot(t, ser, color=c, linewidth=0.95, alpha=0.9, label=f"MA({w})")
ax_ma.axhline(0.0, color="gray", linewidth=0.5, alpha=0.65)
ax_ma.set_title(f"{ticker} — daily return vs rolling means of returns (model inputs)")
ax_ma.set_xlabel("Date")
ax_ma.set_ylabel("pct return / smoothed pct return")
ax_ma.legend(loc="upper left", fontsize=8, ncol=3, framealpha=0.92)
ax_ma.grid(True, alpha=0.35)
fig_ma.tight_layout()
plt.show()