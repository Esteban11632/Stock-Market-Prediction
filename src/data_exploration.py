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

# --- Close price with SMA(close) overlays (same defs as data_pipeline) ---
close_t_ma = ohlcv["Close"].astype(float).reindex(t)
fig_ma, ax_ma = plt.subplots(figsize=(12, 4.5))
ax_ma.plot(t, close_t_ma, color="black", linewidth=0.85, alpha=0.92, label="close")
for ser, w, c in [
    (ds.moving_average_5, 5, "tab:blue"),
    (ds.moving_average_10, 10, "tab:orange"),
    (ds.moving_average_20, 20, "tab:green"),
]:
    ax_ma.plot(t, ser, color=c, linewidth=0.95, alpha=0.9, label=f"SMA(close, {w})")
ax_ma.set_xlabel("Date")
ax_ma.set_ylabel("price (USD)")
ax_ma.set_title(f"{ticker} — closing price vs simple moving averages of close")
ax_ma.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.92)
ax_ma.grid(True, alpha=0.35)
fig_ma.suptitle(
    "Price above/below shorter MAs hints at short-term trend vs longer average.",
    y=1.03,
    fontsize=10,
)
fig_ma.tight_layout()
plt.show()

# --- Rolling volatility of returns (same as data_pipeline) vs returns ---
def _dataset_vol_specs(ds_instance):
    _attrs = [
        ("rolling_volatility_5", 5, "tab:blue"),
        ("rolling_volatility_10", 10, "tab:orange"),
        ("rolling_volatility_20", 20, "tab:red"),
    ]
    out = [(getattr(ds_instance, n, None), w, c) for n, w, c in _attrs if getattr(ds_instance, n, None) is not None]
    return out


vol_series = _dataset_vol_specs(ds)

if vol_series:
    fig_rv, axes_rv = plt.subplots(
        2, 1, figsize=(12, 5.5), sharex=True, height_ratios=[1, 1]
    )
    axes_rv[0].plot(t, ds.returns, color="black", linewidth=0.55, alpha=0.85, label="return")
    axes_rv[0].axhline(0.0, color="gray", linewidth=0.45, alpha=0.6)
    axes_rv[0].set_ylabel("daily return")
    axes_rv[0].set_title(f"{ticker} — returns (top) vs rolling std of returns (bottom)")
    axes_rv[0].grid(True, alpha=0.35)
    axes_rv[0].legend(loc="upper left", fontsize=8)

    ax_v = axes_rv[1]
    for ser, w, c in vol_series:
        ax_v.plot(t, ser, color=c, linewidth=1.0, alpha=0.88, label=f"σ({w}) rolling")
    ax_v.set_xlabel("Date")
    ax_v.set_ylabel(r"rolling $\sigma$ (pct ret.)")
    ax_v.set_title("Recent volatility rises after volatile patches (vol clustering)")
    ax_v.legend(loc="upper left", fontsize=8)
    ax_v.grid(True, alpha=0.35)
    fig_rv.tight_layout()
    plt.show()

    # Scatters: pick longest available vol window for readability
    ser_vol, w_vol, col_vol = vol_series[-1]
    for prefer in (20, 10, 5):
        hit = next(((s, w, c) for s, w, c in vol_series if w == prefer), None)
        if hit is not None:
            ser_vol, w_vol, col_vol = hit
            break

    fig_sc, (ax_s, ax_n) = plt.subplots(1, 2, figsize=(12, 4.2))
    vr = ds.returns
    next_r = vr.shift(-1)
    m0 = ser_vol.notna() & vr.notna()
    ax_s.scatter(
        ser_vol[m0],
        vr[m0],
        s=8,
        alpha=0.3,
        c=col_vol,
        edgecolors="none",
    )
    ax_s.axhline(0, color="gray", lw=0.5)
    ax_s.set_xlabel(f"rolling std of returns at t (window={w_vol})")
    ax_s.set_ylabel(r"return at $t$")
    ax_s.set_title("Same day: big |r| and σ move together (σ uses past r including t)")
    ax_s.grid(True, alpha=0.3)

    m1 = ser_vol.notna() & next_r.notna()
    ax_n.scatter(
        ser_vol[m1],
        next_r[m1],
        s=8,
        alpha=0.3,
        c="tab:green",
        edgecolors="none",
    )
    ax_n.axhline(0, color="gray", lw=0.5)
    ax_n.set_xlabel(f"rolling std of returns at t (window={w_vol})")
    ax_n.set_ylabel(r"return at $t+1$")
    ax_n.set_title("Next day: high σ sometimes precedes large moves (weak on many names)")
    ax_n.grid(True, alpha=0.3)

    fig_sc.suptitle(
        f"{ticker} — rolling volatility vs returns (window={w_vol})",
        y=1.02,
        fontsize=12,
    )
    fig_sc.tight_layout()
    plt.show()

# --- Close + Bollinger Bands + rolling volatility (aligned dates) ---
# Bands describe price level vs local mean; rolling σ is daily return std — both rise in turbulent periods.
if getattr(ds, "bollinger_bands_20", None) is not None:
    close_t = ohlcv["Close"].astype(float).reindex(t)
    mid = ds.bollinger_bands_20
    up = ds.bollinger_bands_20_upper
    lo = ds.bollinger_bands_20_lower
    denom = (up - lo).replace(0, np.nan)
    pct_b = (close_t - lo) / denom

    combo_vol = _dataset_vol_specs(ds)

    fig_combo, axes_combo = plt.subplots(
        2,
        1,
        figsize=(12, 6.8),
        sharex=True,
        height_ratios=[1.35, 1],
    )
    ax_top = axes_combo[0]
    ax_top.plot(t, close_t, color="black", linewidth=0.8, label="close", alpha=0.92)
    ax_top.plot(t, mid, color="tab:blue", linewidth=0.9, label="BB middle (SMA 20)")
    ax_top.fill_between(t, lo, up, color="steelblue", alpha=0.18, label="±2σ band")
    ax_top.plot(t, up, color="tab:red", linewidth=0.65, linestyle="--", label="upper")
    ax_top.plot(t, lo, color="tab:green", linewidth=0.65, linestyle="--", label="lower")
    ax_top.set_ylabel("price (USD)")
    ax_top.set_title(f"{ticker} — closing price vs Bollinger Bands (data_pipeline defs)")
    ax_top.legend(loc="upper left", fontsize=7, ncol=3, framealpha=0.92)
    ax_top.grid(True, alpha=0.3)

    ax_bot = axes_combo[1]
    if combo_vol:
        for ser, w, c in combo_vol:
            ax_bot.plot(t, ser, color=c, linewidth=1.0, alpha=0.88, label=f"rolling σ(ret), {w}d")
        ax_bot.legend(loc="upper left", fontsize=8)
    else:
        ax_bot.text(0.5, 0.5, "(no rolling_volatility_* on dataset)", transform=ax_bot.transAxes, ha="center")
    ax_bot.set_xlabel("Date")
    ax_bot.set_ylabel(r"rolling $\sigma$ of daily returns")
    ax_bot.set_title(
        "Higher σ often aligns with wider bands / faster price swings (not causal, same regime)."
    )
    ax_bot.grid(True, alpha=0.3)

    fig_combo.suptitle(
        f"{ticker} — how Bollinger width and return volatility relate to the close path",
        y=1.02,
        fontsize=11,
    )
    fig_combo.tight_layout()
    plt.show()

    ret = ds.returns
    next_ret = ret.shift(-1)
    valid_b = pct_b.notna()

    fig_bbs, (ax_bb_s, ax_bb_n) = plt.subplots(1, 2, figsize=(12, 4.2))
    m0 = valid_b & ret.notna()
    ax_bb_s.scatter(pct_b[m0], ret[m0], s=10, alpha=0.35, c="tab:blue", edgecolors="none")
    ax_bb_s.axhline(0, color="gray", lw=0.5)
    ax_bb_s.axvline(0.5, color="gray", linestyle=":", lw=0.5)
    ax_bb_s.set_xlabel("%B at close of day t")
    ax_bb_s.set_ylabel(r"return at $t$")
    ax_bb_s.set_title("Same day: extremes of %B line up with large |return|")
    ax_bb_s.grid(True, alpha=0.3)

    m1 = valid_b & next_ret.notna()
    ax_bb_n.scatter(pct_b[m1], next_ret[m1], s=10, alpha=0.35, c="tab:green", edgecolors="none")
    ax_bb_n.axhline(0, color="gray", lw=0.5)
    ax_bb_n.axvline(0.5, color="gray", linestyle=":", lw=0.5)
    ax_bb_n.set_xlabel("%B at close of day t")
    ax_bb_n.set_ylabel(r"return at $t+1$")
    ax_bb_n.set_title("Next day: mild mean-reversion vs momentum varies by asset")
    ax_bb_n.grid(True, alpha=0.3)

    fig_bbs.suptitle(f"{ticker} — Bollinger %B vs returns", y=1.02, fontsize=12)
    fig_bbs.tight_layout()
    plt.show()