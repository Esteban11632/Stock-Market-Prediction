import numpy as np
import torch
from pathlib import Path
import yfinance as yf
from torch.utils.data import DataLoader, TensorDataset
from joblib import load
import pandas as pd

from cnn_lstm_attention import ConvLSTMAttentionStockModel
from config import get_config
from data_pipeline import StockMarketDataset
import utils


config = get_config()

ticker = config["ticker"]
seq_length = config["seq_length"]
test_start_date = config["test_start_date"]
wanted_features = config["wanted_features"]
engineering_features = config["engineering_features"]

df = yf.download(ticker, start=test_start_date)

device = "cuda" if torch.cuda.is_available() else "cpu"

directory = Path(__file__).resolve().parent.parent
models_dir = directory / "models"
scaler_dir = directory / "scalers"

filename = "stock_market_model.pth"

full_ds = StockMarketDataset(df=df, start_date=test_start_date, seq_length=seq_length, wanted_features=wanted_features, engineering_features=engineering_features, ticker=ticker)

X, y, anchor_prices, target_start_positions = utils.as_numpy_all_x(full_ds, df)

nf_in = X.shape[-1]
nf_out = len(full_ds.target_column_names)

horizons = [1, 5, 20]

scaler_X = load(scaler_dir / "scaler_X.joblib")
scaler_y = load(scaler_dir / "scaler_y.joblib")

X_scaled = scaler_X.transform(
    X.reshape(-1, nf_in)
).reshape(X.shape).astype("float32")

test_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_scaled)),
    batch_size=min(512, len(X_scaled)),
    shuffle=False,
)

model = ConvLSTMAttentionStockModel(
    input_dim=nf_in,
    output_dim=nf_out,
    num_lstm_layers=config["num_lstm_layers"]
).to(device)

state = torch.load(models_dir / filename, map_location=device)
model.load_state_dict(state)
model.eval()

preds_scaled = []

with torch.no_grad():
    for (xb,) in test_loader:
        xb = xb.to(device)
        preds_scaled.append(model(xb).cpu())

y_pred_scaled = torch.cat(preds_scaled, dim=0).numpy()
y_pred = scaler_y.inverse_transform(y_pred_scaled)


# Global RMSE across all targets
valid_mask = ~np.isnan(y)

rmse_log_returns_all = utils.get_rmse(
    y[valid_mask],
    y_pred[valid_mask]
)

true_prices_all = np.full_like(y, np.nan, dtype=np.float64)
predicted_prices_all = np.full_like(y_pred, np.nan, dtype=np.float64)

for index, H in enumerate(horizons):
    valid = ~np.isnan(y[:, index])

    true_prices_all[valid, index] = utils.log_returns_to_terminal_prices(
        anchor_prices[valid],
        y[valid][:, [index]]
    )[:, 0]

    predicted_prices_all[valid, index] = utils.log_returns_to_terminal_prices(
        anchor_prices[valid],
        y_pred[valid][:, [index]]
    )[:, 0]

valid_price_mask = ~np.isnan(true_prices_all)

rmse_prices_all = utils.get_rmse(
    true_prices_all[valid_price_mask],
    predicted_prices_all[valid_price_mask]
)

print(f"Global log-return RMSE: {rmse_log_returns_all:.6f}")
print(f"Global terminal-price RMSE: {rmse_prices_all:.6f}")

# Graphs
for index, H in enumerate(horizons):

    valid = ~np.isnan(y[:, index])

    terminal_dates = df.index[
        target_start_positions[valid] + H - 1
    ]

    true_prices_h = utils.log_returns_to_terminal_prices(
        anchor_prices[valid],
        y[valid][:, [index]]
    )

    predicted_prices_h = utils.log_returns_to_terminal_prices(
        anchor_prices[valid],
        y_pred[valid][:, [index]]
    )

    future_log_return = y_pred[-1, index]
    future_anchor_price = anchor_prices[-1]
    future_prediction = future_anchor_price * np.exp(future_log_return)

    future_date = df.index[-1] + pd.tseries.offsets.BDay(H)

    utils.plot_prediction_timeline(
        terminal_dates,
        true_prices_h,
        predicted_prices_h,
        [full_ds.target_column_names[index]],
        ticker,
        output_index=0,
        future_date=future_date,
        future_prediction=future_prediction
    )