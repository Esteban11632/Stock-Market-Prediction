import torch
from pathlib import Path
import yfinance as yf
from torch.utils.data import DataLoader, TensorDataset
from joblib import load

from model import StockMarketModel
from data_pipeline import StockMarketDataset
import utils

ticker = "MSFT"
seq_length = 30

df = yf.download(ticker, start="2020-01-01")

device = "cuda" if torch.cuda.is_available() else "cpu"
directory = Path(__file__).resolve().parent.parent
models_dir = directory / "models"
scaler_dir = directory / "scalers"
models_dir.mkdir(parents=True, exist_ok=True)
scaler_dir.mkdir(parents=True, exist_ok=True)

filename = "stock_market_model_all_columns.pth"
full_ds = StockMarketDataset(df, seq_length)
X, y = full_ds.as_numpy()

# Last dimension is the number of features
nf = X.shape[-1]

scaler_X = load(scaler_dir / "scaler_X_all_columns.joblib")
scaler_y = load(scaler_dir / "scaler_y_all_columns.joblib")

X_scaled = scaler_X.transform(X.reshape(-1, nf)).reshape(X.shape).astype("float32")
y_scaled = scaler_y.transform(y).astype("float32")

test_loader = DataLoader(
    TensorDataset(
        torch.from_numpy(X_scaled),
        torch.from_numpy(y_scaled),
    ),
    batch_size=min(512, len(X_scaled)),
    shuffle=False,
)

model = StockMarketModel(
    input_dim=nf,
    hidden_dim=16,
    num_layers=1,
    output_dim=nf,
    dropout=0.35,
).to(device)
state = torch.load(models_dir / filename, map_location=device)
model.load_state_dict(state)

y_pred_scaled, y_true_scaled = utils.predict_loader(model, test_loader, device)
y_pred = scaler_y.inverse_transform(y_pred_scaled.numpy())
y_true = scaler_y.inverse_transform(y_true_scaled.numpy())
indices = list(range(0, len(X_scaled)))
pred_prices, actual_prices, test_dates = utils.predicted_returns_to_prices(df, full_ds, indices, seq_length, y_pred)
horizon = 1
dates_plot, actual_prices_plot, pred_prices_plot, test_rmse_price, test_dates, horizon, actual_prices, pred_prices = utils.recursive_forecast(model, df, full_ds, seq_length, scaler_X, scaler_y, device, horizon, nf, pred_prices, test_dates, actual_prices)
utils.graph_predictions(dates_plot, actual_prices_plot, pred_prices_plot, test_rmse_price, test_dates, ticker, horizon, actual_prices, pred_prices)