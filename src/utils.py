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


def predict_numpy_batches(model, X_np, device, batch_size=512):
    """Run inference on `(N, T, F)` float array; returns `(N, n_out)` on CPU numpy."""
    model.eval()
    n = len(X_np)
    X_t = torch.from_numpy(X_np.astype(np.float32, copy=False))
    outs = []
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            xb = X_t[start : start + batch_size].to(device)
            outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs, axis=0)


def permutation_feature_importance_mse(
    model,
    X_np,
    y_np,
    device,
    feature_names=None,
    batch_size=512,
    seed=0,
):
    """Shuffle one input channel across samples (break alignment with targets), measure MSE change.

    `X_np`/`y_np` must be on the **same scale the model sees** (e.g. scaler output). MSE matches
    `np.mean((pred - y)**2)` over **all outputs and samples** — same normalization as averaging
    `nn.MSELoss()` over disjoint batches whose total size is `(N * n_outputs)`.

    For each feature index ``i``, sets ``Xp[:, :, i] = X[perm, :, i]`` where ``perm`` is a random
    permutation of batch indices — the usual permutation-importance analogue for tensors shaped
    ``(samples, seq_len, channels)``.
    """
    if isinstance(X_np, torch.Tensor):
        X_np = X_np.detach().cpu().numpy()
    else:
        X_np = np.asarray(X_np)

    if isinstance(y_np, torch.Tensor):
        y_np = y_np.detach().cpu().numpy()
    else:
        y_np = np.asarray(y_np)

    rng = np.random.default_rng(seed)
    n, _, nfeat = X_np.shape

    names = (
        feature_names
        if feature_names is not None
        else [f"f{i}" for i in range(nfeat)]
    )
    if len(names) != nfeat:
        raise ValueError(f"Got {len(names)} names but X has {nfeat} channels")

    pred0 = predict_numpy_batches(model, X_np, device, batch_size=batch_size)
    base_mse = np.mean((pred0 - y_np) ** 2)

    deltas = []
    for i in range(nfeat):
        Xp = X_np.copy()
        perm = rng.permutation(n)
        Xp[:, :, i] = X_np[perm, :, i]
        pred = predict_numpy_batches(model, Xp, device, batch_size=batch_size)
        deltas.append(np.mean((pred - y_np) ** 2) - base_mse)

    return pd.DataFrame({"Feature": names, "mse_increase": deltas}).sort_values(
        "mse_increase", ascending=False
    )


def plot_permutation_feature_importance(
    importance_df,
    *,
    value_col="mse_increase",
    feature_col="Feature",
    title="Feature Importance (Permutation Method)",
    xlabel="Increase in MSE After Feature Permutation",
    show=True,
):
    """Horizontal bar chart of permutation ΔMSE (dark theme, strongest feature at top)."""
    df_plot = importance_df.sort_values(value_col, ascending=True)

    h = max(6.0, 0.35 * len(df_plot))
    fig, ax = plt.subplots(figsize=(10, h), facecolor="black")
    ax.set_facecolor("black")

    y_labels = df_plot[feature_col].astype(str)
    values = df_plot[value_col].to_numpy()

    ax.barh(y_labels, values, color="white", height=0.7, zorder=2)
    ax.axvline(0.0, color="#555555", linewidth=1.0, zorder=1)

    ax.set_title(title, color="white", fontsize=13)
    ax.set_xlabel(xlabel, color="white", fontsize=10)
    ax.set_ylabel("Feature", color="#00ff66", fontsize=10)
    ax.tick_params(axis="both", colors="white", which="both")
    ax.grid(True, axis="x", color="#333333", linestyle="-", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for lbl in ax.get_yticklabels():
        lbl.set_color("#00ff66")

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax
    pred_ret_flat = np.asarray(pred_returns[:, 0]).ravel()
    assert len(pred_ret_flat) == len(test_indices)

    pred_prices = []
    actual_prices = []
    test_dates = []

    close = get_column_normalized_to_1d(df, "Close")

    r = full_ds.returns

    # Align rows with test_loader order: sample j corresponds to dataset index test_indices[j].
    for j, k in enumerate(test_indices):
        p = k + full_ds._warmup + seq_length
        prev_c = float(close.loc[r.index[p - 1]])
        pred_prices.append(prev_c * (1.0 + float(pred_ret_flat[j])))
        actual_prices.append(float(close.loc[r.index[p]]))
        test_dates.append(r.index[p])

    pred_prices = np.array(pred_prices)
    actual_prices = np.array(actual_prices)
    test_dates = pd.DatetimeIndex(test_dates)

    return pred_prices, actual_prices, test_dates

def recursive_forecast(
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
    nf_out=None,
):
    if nf_out is None:
        nf_out = nf_in
    last_idx = len(full_ds) - 1
    window = full_ds[last_idx][0].numpy().copy()
    close = get_column_normalized_to_1d(df, "Close")
    r = full_ds.returns
    anchor_i = last_idx + full_ds._warmup + seq_length - 1
    forecast_i = last_idx + full_ds._warmup + seq_length
    price_curr = float(close.loc[r.index[anchor_i]])
    recursive_prices = [price_curr]
    recursive_returns = []
    model.eval()
    with torch.inference_mode():
        for _ in range(horizon):
            flat = window.reshape(-1, nf_in)
            X_scaled = scaler_X.transform(flat).reshape(seq_length, nf_in).astype(np.float32)
            xb = torch.tensor(X_scaled, device=device).unsqueeze(0)
            pred_scaled = model(xb)
            y_pred = scaler_y.inverse_transform(pred_scaled.cpu().numpy())[0].reshape(-1)
            r_pred = float(y_pred[0])
            recursive_returns.append(r_pred)
            price_curr = price_curr * (1.0 + r_pred)
            recursive_prices.append(price_curr)
            window = np.roll(window, -1, axis=0)
            y_full = np.zeros(nf_in, dtype=np.float64)
            y_full[: min(nf_out, len(y_pred))] = y_pred[:nf_out]
            if nf_out < nf_in:
                y_full[nf_out:] = window[-2, nf_out:nf_in]
            window[-1] = y_full

    recursive_prices = np.array(recursive_prices, dtype=np.float64)
    print(
        f"Recursive {horizon}-day forecast (anchor close {r.index[anchor_i]}; next bar {r.index[forecast_i]}): "
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