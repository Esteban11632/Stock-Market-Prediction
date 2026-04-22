import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from pandas.tseries.offsets import BDay
from sklearn.metrics import root_mean_squared_error

def get_column_normalized_to_1d(df, column_name):
    column = df[column_name].squeeze()
    if isinstance(column, pd.DataFrame):
        column = column.iloc[:, 0]
    return column

def predict_loader(model, loader, device):
    model.eval()
    preds, ys = [], []
    with torch.inference_mode():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            preds.append(model(Xb).cpu())
            ys.append(yb)
    return torch.cat(preds, dim=0), torch.cat(ys, dim=0)

def predicted_returns_to_prices(df, full_ds, test_indices, seq_length, pred_returns):
    pred_ret_flat = np.asarray(pred_returns[:, 0]).ravel()
    assert len(pred_ret_flat) == len(test_indices)

    pred_prices = []
    actual_prices = []
    test_dates = []

    close = get_column_normalized_to_1d(df, "Close")

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

    return pred_prices, actual_prices, test_dates

def recursive_forecast(model, df, full_ds, seq_length, scaler_X, scaler_y, device, horizon, nf, pred_prices, test_dates, actual_prices):
    last_idx = len(full_ds) - 1
    window = full_ds[last_idx][0].numpy().copy()
    close = get_column_normalized_to_1d(df, "Close")
    r = full_ds.returns
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

    return dates_plot, actual_prices_plot, pred_prices_plot, test_rmse_price, test_dates, horizon, actual_prices, pred_prices

def graph_predictions(dates_plot, actual_prices_plot, pred_prices_plot, test_rmse_price, test_dates, ticker, horizon, actual_prices, pred_prices):
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