import torch
from pathlib import Path
from sklearn.metrics import root_mean_squared_error
import yfinance as yf
from torch.utils.data import DataLoader, TensorDataset
from joblib import load
from cnn_lstm_attention import ConvLSTMAttentionStockModel
from data_pipeline import StockMarketDataset
import utils

ticker = "VOO"
seq_length = 10

df = yf.download(ticker, start="2024-01-01")

device = "cuda" if torch.cuda.is_available() else "cpu"
directory = Path(__file__).resolve().parent.parent
models_dir = directory / "models"
scaler_dir = directory / "scalers"
models_dir.mkdir(parents=True, exist_ok=True)
scaler_dir.mkdir(parents=True, exist_ok=True)

filename = "stock_market_model.pth"
full_ds = StockMarketDataset(df, seq_length)
X, y = full_ds.as_numpy()

nf_in = X.shape[-1]
nf_out = y.shape[-1]

scaler_X = load(scaler_dir / "scaler_X.joblib")
scaler_y = load(scaler_dir / "scaler_y.joblib")

X_scaled = scaler_X.transform(X.reshape(-1, nf_in)).reshape(X.shape).astype("float32")
y_scaled = scaler_y.transform(y).astype("float32")

test_loader = DataLoader(
    TensorDataset(
        torch.from_numpy(X_scaled),
        torch.from_numpy(y_scaled),
    ),
    batch_size=min(512, len(X_scaled)),
    shuffle=False,
)

model = ConvLSTMAttentionStockModel(
    input_dim=nf_in,
    output_dim=nf_out,
).to(device)
state = torch.load(models_dir / filename, map_location=device)
model.load_state_dict(state)

y_pred_scaled, y_true_scaled = utils.predict_loader(model, test_loader, device)
y_pred = scaler_y.inverse_transform(y_pred_scaled.numpy())
y_true = scaler_y.inverse_transform(y_true_scaled.numpy())
# Column 0 = daily return (StockMarketDataset.cols_y[0]).
_rmse_ret = root_mean_squared_error(y_true[:, 0], y_pred[:, 0])
print(f"RMSE (daily return): {_rmse_ret:.6f}")
indices = list(range(0, len(X_scaled)))
pred_prices, actual_prices, test_dates = utils.predicted_returns_to_prices(df, full_ds, indices, seq_length, y_pred)
horizon = 5
dates_plot, actual_prices_plot, pred_prices_plot, test_rmse_price, test_dates, horizon, actual_prices, pred_prices = utils.recursive_forecast(
    model,
    df,
    full_ds,
    seq_length,
    scaler_X,
    scaler_y,
    device,
    horizon,
    nf_in,
    pred_prices,
    test_dates,
    actual_prices,
    nf_out=nf_out,
)
utils.graph_predictions(dates_plot, actual_prices_plot, pred_prices_plot, test_rmse_price, test_dates, ticker, horizon, actual_prices, pred_prices)