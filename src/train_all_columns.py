import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import BDay
from sklearn.metrics import root_mean_squared_error
import torch
import torch.optim as optim
from model import StockMarketModel
from tqdm import tqdm
from pathlib import Path
from data_pipeline import StockMarketDataset, split_data
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import utils

device = "cuda" if torch.cuda.is_available() else "cpu"

ticker = "MSFT"

df = yf.download(ticker, start="2000-01-01")

seq_length = 30

full_ds = StockMarketDataset(df, seq_length)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(full_ds)

start_test = len(X_train) + len(X_val)
test_indices = list(range(start_test, start_test + len(X_test)))

# Initialize scalers
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Last dimension is the number of features
nf = X_train.shape[-1]

# Reshape to (N, nf) for scaling
scaler_X.fit(X_train.reshape(-1, nf))
X_train_scaled = scaler_X.transform(X_train.reshape(-1, nf)).reshape(X_train.shape)
X_val_scaled = scaler_X.transform(X_val.reshape(-1, nf)).reshape(X_val.shape)
X_test_scaled = scaler_X.transform(X_test.reshape(-1, nf)).reshape(X_test.shape)

# Fit scaler on training targets
scaler_y.fit(y_train)
# Transform training, validation and test targets
y_train_scaled = scaler_y.transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)

# Convert to PyTorch tensors
train_ds = TensorDataset(
    torch.from_numpy(X_train_scaled).float(),
    torch.from_numpy(y_train_scaled).float(),
)

val_ds = TensorDataset(
    torch.from_numpy(X_val_scaled).float(),
    torch.from_numpy(y_val_scaled).float(),
)

test_ds = TensorDataset(
    torch.from_numpy(X_test_scaled).float(),
    torch.from_numpy(y_test_scaled).float(),
)

# Mini-batches add gradient noise and usually generalize better than full-batch GD.
batch_size = min(256, max(32, len(train_ds) // 16))
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=min(512, len(val_ds)), shuffle=False)
test_loader = DataLoader(test_ds, batch_size=min(512, len(test_ds)), shuffle=False)

torch.manual_seed(42)

def mse_mean_over_batches(loader, model, criterion, device, out_dim):
    """MSELoss(reduction='mean') is over all elements; aggregate correctly across batches."""
    total = 0.0
    n_elem = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            loss = criterion(model(X_batch), y_batch)
            batch_elems = X_batch.size(0) * out_dim
            total += loss.item() * batch_elems
            n_elem += batch_elems
    return total / max(n_elem, 1)

model = StockMarketModel(
    input_dim=nf, hidden_dim=16, num_layers=1, output_dim=nf, dropout=0.35
).to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

num_epochs = 130
patience = 60
best_val = float("inf")
epochs_no_improve = 0
best_state = None

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
    n_elem_train = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch)
        loss = criterion(output, y_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batch_elems = X_batch.size(0) * nf
        train_loss_sum += loss.item() * batch_elems
        n_elem_train += batch_elems
    train_loss = train_loss_sum / max(n_elem_train, 1)

    model.eval()
    val_loss = mse_mean_over_batches(val_loader, model, criterion, device, nf)

    if val_loss < best_val:
        best_val = val_loss
        epochs_no_improve = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        epochs_no_improve += 1

    if epoch % 25 == 0:
        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f} (best: {best_val:.4f})"
        )

    if epochs_no_improve >= patience:
        print(f"Early stopping at epoch {epoch+1} (no val improvement for {patience} epochs).")
        break

if best_state is not None:
    model.load_state_dict(best_state)

eval_train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=False)
y_train_pred, y_train_scaled = predict_loader(model, eval_train_loader, device)
y_test_pred, y_test_scaled = predict_loader(model, test_loader, device)

y_train_preds = scaler_y.inverse_transform(y_train_pred.numpy())
y_train_inv = scaler_y.inverse_transform(y_train_scaled.numpy())
y_test_preds = scaler_y.inverse_transform(y_test_pred.numpy())
y_test_inv = scaler_y.inverse_transform(y_test_scaled.numpy())

# Multi-output (5 targets); sklearn averages RMSE across outputs by default.
train_rmse = root_mean_squared_error(y_train_inv, y_train_preds)
test_rmse = root_mean_squared_error(y_test_inv, y_test_preds)

print(
    f"Train RMSE (avg over 5 targets): {train_rmse:.6f} | "
    f"Test RMSE (avg over 5 targets): {test_rmse:.6f}"
)

pred_ret_flat = np.asarray(y_test_preds[:, 0]).ravel()
assert len(pred_ret_flat) == len(test_indices)

pred_prices = []
actual_prices = []
test_dates = []

close = utils.get_column_normalized_to_1d(df, "Close")

r = full_ds.returns

# Align rows with test_loader order: sample j corresponds to dataset index test_indices[j].
for j, k in enumerate(test_indices):
    p = k + seq_length
    prev_c = float(close.loc[r.index[p - 1]])
    pred_prices.append(prev_c * (1.0 + float(pred_ret_flat[j])))
    actual_prices.append(float(close.loc[r.index[p]]))
    test_dates.append(r.index[p])

pred_prices = np.array(pred_prices)
actual_prices = np.array(actual_prices)
test_dates = pd.DatetimeIndex(test_dates)

# --- Recursive multi-day ahead: same (seq_len, 5) window as training; roll with full pred vector ---
# Model is (batch, seq_length, nf) -> (batch, nf). We only use predicted return [0] for price path.
horizon = 60
last_idx = len(full_ds) - 1
window = full_ds[last_idx][0].numpy().copy()
price_curr = float(close.loc[r.index[last_idx + seq_length]])
recursive_prices = [price_curr]
recursive_returns = []
model.eval()
with torch.inference_mode():
    for _ in range(horizon):
        flat = window.reshape(-1, nf)
        X_scaled = scaler_X.transform(flat).reshape(seq_length, nf).astype(np.float32)
        xb = torch.tensor(X_scaled, device=device).unsqueeze(0)
        pred_scaled = model(xb)
        y_pred = scaler_y.inverse_transform(pred_scaled.cpu().numpy())[0]
        r_pred = float(y_pred[0])
        recursive_returns.append(r_pred)
        price_curr = price_curr * (1.0 + r_pred)
        recursive_prices.append(price_curr)
        window = np.roll(window, -1, axis=0)
        window[-1] = y_pred

recursive_prices = np.array(recursive_prices, dtype=np.float64)
print(
    f"Recursive {horizon}-day forecast (closes from last bar {r.index[last_idx + seq_length]}): "
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

# Save the model
model_path = Path.cwd() / "Stock Market Prediction" / "models"
model_path.mkdir(parents=True, exist_ok=True)
model_name = "stock_market_model_all_columns.pth"
model_save_path = model_path / model_name
torch.save(obj=model.state_dict(), f=model_save_path)