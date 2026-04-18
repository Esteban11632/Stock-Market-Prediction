import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import StockMarketModel
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import utils

from data_pipeline1 import (
    StockMarketDataset,
    fit_scaler_returns,
    build_scaled_tensors,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

ticker = "MSFT"

df = yf.download(ticker, start="2000-01-01")

close = utils.get_column_normalized_to_1d(df, "Close")

seq_length = 30
full_ds = StockMarketDataset(df, seq_length)
r = full_ds.r

n_samples = len(full_ds)
train_size = int(0.8 * n_samples)
# Time-ordered split: first 80% train, last 20% test
train_indices = list(range(train_size))
test_indices = list(range(train_size, n_samples))

scaler = fit_scaler_returns(full_ds, train_indices)

# Scale once → TensorDataset (fast); avoid scaler.transform in __getitem__ every step
X_train, y_train = build_scaled_tensors(full_ds, train_indices, scaler)
X_test, y_test = build_scaled_tensors(full_ds, test_indices, scaler)

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)

# Full-batch: one batch = entire train / test set per epoch (no mini-batches)
_bs_train = len(train_ds)
_bs_test = len(test_ds)
train_loader = DataLoader(train_ds, batch_size=_bs_train, shuffle=True, drop_last=False)
test_loader = DataLoader(test_ds, batch_size=_bs_test, shuffle=False)
train_eval_loader = DataLoader(train_ds, batch_size=_bs_train, shuffle=False)
test_eval_loader = DataLoader(test_ds, batch_size=_bs_test, shuffle=False)

torch.manual_seed(42)

model = StockMarketModel(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100

def run_loader_loss(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    with torch.inference_mode():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            pred = model(Xb)
            total += criterion(pred, yb).item() * Xb.size(0)
            n += Xb.size(0)
    return total / max(n, 1)


def predict_loader(model, loader, device):
    model.eval()
    preds, ys = [], []
    with torch.inference_mode():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            preds.append(model(Xb).cpu())
            ys.append(yb)
    return torch.cat(preds, dim=0), torch.cat(ys, dim=0)


for epoch in tqdm(range(num_epochs)):

    model.train()
    train_loss_sum = 0.0
    n_train = 0
    for Xb, yb in train_loader:
        Xb = Xb.to(device)
        yb = yb.to(device)
        yb_pred = model(Xb)
        loss = criterion(yb_pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item() * Xb.size(0)
        n_train += Xb.size(0)
    train_loss = train_loss_sum / max(n_train, 1)

    test_loss = run_loader_loss(model, test_loader, device)

    if epoch % 25 == 0:
        print(f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

# Predictions (scaled space → real returns)
y_train_pred, y_train_scaled = predict_loader(model, train_eval_loader, device)
y_test_pred, y_test_scaled = predict_loader(model, test_eval_loader, device)

y_train_preds = scaler.inverse_transform(y_train_pred.numpy())
y_train_inv = scaler.inverse_transform(y_train_scaled.numpy())
y_test_preds = scaler.inverse_transform(y_test_pred.numpy())
y_test_inv = scaler.inverse_transform(y_test_scaled.numpy())

train_rmse_ret = root_mean_squared_error(y_train_inv, y_train_preds)
test_rmse_ret = root_mean_squared_error(y_test_inv, y_test_preds)

print(
    f"Train RMSE (returns): {train_rmse_ret:.6f} | Test RMSE (returns): {test_rmse_ret:.6f}"
)

# --- One-step-ahead price from predicted returns ---
len_test = len(test_indices)
pred_ret_flat = np.asarray(y_test_preds).ravel()

pred_prices = []
actual_prices = []
test_dates = []
for j in range(len_test):
    k = train_size + j
    prev_c = float(close.loc[r.index[k + seq_length - 1]])
    pred_prices.append(prev_c * (1.0 + float(pred_ret_flat[j])))
    actual_prices.append(float(close.loc[r.index[k + seq_length]]))
    test_dates.append(r.index[k + seq_length])

pred_prices = np.array(pred_prices)
actual_prices = np.array(actual_prices)
test_dates = pd.DatetimeIndex(test_dates)

# --- Recursive 5-day ahead (return-only): roll window of raw returns, not raw prices ---
# Model expects (batch, seq_length, 1) scaled returns; each step appends predicted return to window.
horizon = 35
window = r.iloc[-seq_length:].values.astype(np.float64).copy()
price_curr = float(close.loc[r.index[-1]])
recursive_prices = [price_curr]
recursive_returns = []
model.eval()
with torch.inference_mode():
    for _ in range(horizon):

        # Shape (seq_length,) -> (seq_length, 1) since scaler expects 2D array
        X_scaled = scaler.transform(window.reshape(seq_length, 1)).astype(np.float32)

        # (seq_length, 1) -> (1, seq_length, 1) for LSTM input with batch dimension
        xb = torch.tensor(X_scaled, device=device).unsqueeze(0)

        # (1, seq_length, 1) -> (1, 1)
        pred_scaled = model(xb)

        # (1, 1) -> float
        r_pred = float(scaler.inverse_transform(pred_scaled.cpu().numpy())[0, 0])
        recursive_returns.append(r_pred)
        price_curr = price_curr * (1.0 + r_pred)
        recursive_prices.append(price_curr)
        window = np.roll(window, -1)
        window[-1] = r_pred

recursive_prices = np.array(recursive_prices, dtype=np.float64)
print(
    f"Recursive {horizon}-day forecast (closes from last bar {r.index[-1]}): "
    f"{recursive_prices.round(4)}"
)

# Append multi-day forecast to pred series for plotting (future business days; no actuals yet)
future_dates = pd.date_range(
    start=r.index[-1] + BDay(1), periods=horizon, freq="B"
)
pred_prices_plot = np.concatenate([pred_prices, recursive_prices[1:]])
dates_plot = pd.DatetimeIndex(
    np.concatenate([test_dates.to_numpy(), future_dates.to_numpy()])
)
actual_prices_plot = np.concatenate([actual_prices, np.full(horizon, np.nan)])

test_rmse_price = root_mean_squared_error(actual_prices, pred_prices)
print(f"Test RMSE (price, one-step from pred. returns): {test_rmse_price:.4f}")

fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(4, 1)
ax1 = fig.add_subplot(gs[:3, 0])
ax2 = fig.add_subplot(gs[3, 0])

ax1.plot(dates_plot, actual_prices_plot, color="blue", label="Actual close")
ax1.plot(
    dates_plot,
    pred_prices_plot,
    color="green",
    label="Pred. close (test) + recursive forecast",
)
ax1.set_title(f"{ticker} test — one-step + {horizon}-day recursive forecast")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price")
ax1.grid(True)
ax1.legend()

err = np.abs(actual_prices - pred_prices)
ax2.axhline(test_rmse_price, color="blue", linestyle="--", label="Test RMSE (price)")
ax2.plot(test_dates, err, "r", label="Absolute price error (test only)")
ax2.set_title(f"{ticker} prediction error (price space)")
ax2.set_xlabel("Date")
ax2.set_ylabel("Error")
ax2.grid(True)
ax2.legend()

fig.tight_layout()
plt.show()

model_dir = Path(__file__).resolve().parent.parent / "models"
model_dir.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), model_dir / "stock_market_model.pth")
