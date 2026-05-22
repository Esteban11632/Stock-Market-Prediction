import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
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


def get_shap_values(
    model,
    X_train,
    X_test,
    device,
    background_size=96,
    sample_size=24,
    output_index=None,
):
    """SHAP GradientExplainer on subsets of scaled ``(B, L, F)`` windows.

    ``shap`` is imported lazily (avoids numba/llvmlite at ``import utils``).

    CUDA + cuDNN LSTM requires **training mode** during ``shap_values`` backward;
    we set ``.train()`` only for that call, then restore prior mode.

    ``output_index``: if set, explains ``model(x)[:, output_index]`` only.
    """
    import shap

    prev_mode = model.training
    device = torch.device(device) if isinstance(device, str) else device

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    bg_n = max(8, min(int(background_size), len(X_train)))
    ex_n = max(1, min(int(sample_size), len(X_test)))

    background = torch.from_numpy(X_train[:bg_n]).to(device)
    samples = torch.from_numpy(X_test[:ex_n]).to(device)

    if output_index is not None:
        j = int(output_index)

        class _PickOutput(torch.nn.Module):
            def __init__(self, m, idx):
                super().__init__()
                self.m = m
                self.idx = idx

            def forward(self, x):
                return self.m(x)[:, self.idx : self.idx + 1]

        target_model = _PickOutput(model, j).to(device)
    else:
        target_model = model.to(device)

    explainer = shap.GradientExplainer(target_model, background)

    target_model.train()
    try:
        shap_values = explainer.shap_values(samples)
    finally:
        target_model.train(prev_mode)

    return shap_values, samples


def shap_time_heatmap(name, feature_names, sv):
    """
    sv shape:
    (samples, seq_length, features)
    """

    feature_day_importance = np.abs(sv).mean(axis=0)

    plt.figure(figsize=(10, 6))

    plt.imshow(
        feature_day_importance.T,
        aspect="auto"
    )

    plt.colorbar(label="Mean |SHAP value|")

    plt.xlabel("Day in window")
    plt.ylabel("Feature")

    plt.yticks(
        range(len(feature_names)),
        feature_names
    )

    seq_length = feature_day_importance.shape[0]

    plt.xticks(
        range(seq_length),
        [f"t-{seq_length-1-i}" for i in range(seq_length)]
    )

    plt.title(f"{name} SHAP Feature Importance Across Time")

    plt.show()

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
    target_output_index=None,
):
    """Shuffle one input channel across samples (break alignment with targets), measure MSE change.

    `X_np`/`y_np` must be on the **same scale the model sees** (e.g. scaler output).

    If ``target_output_index`` is ``None``, MSE is ``np.mean((pred - y)**2)`` over **all** outputs
    and samples (like global ``nn.MSELoss``). If set to an int (e.g. ``0`` for 1-day cumulative log
    return), only that output column is used: ``np.mean((pred[:, j] - y[:, j])**2)``.

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

    if target_output_index is not None:
        j = int(target_output_index)
        if j < 0 or j >= y_np.shape[-1]:
            raise ValueError(
                f"target_output_index={j} out of range for y shape {y_np.shape}"
            )

    def _output_mse(pred, y):
        if target_output_index is not None:
            j = target_output_index
            return np.mean((pred[:, j] - y[:, j]) ** 2)
        return np.mean((pred - y) ** 2)

    pred0 = predict_numpy_batches(model, X_np, device, batch_size=batch_size)
    base_mse = _output_mse(pred0, y_np)

    deltas = []
    for i in range(nfeat):
        Xp = X_np.copy()
        perm = rng.permutation(n)
        Xp[:, :, i] = X_np[perm, :, i]
        pred = predict_numpy_batches(model, Xp, device, batch_size=batch_size)
        deltas.append(_output_mse(pred, y_np) - base_mse)

    return pd.DataFrame({"Feature": names, "mse_increase": deltas}).sort_values(
        "mse_increase", ascending=False
    )


def window_scores_flat_to_grid(
    flat_scores: np.ndarray, seq_length: int, feature_names
) -> pd.DataFrame:
    """Flattened row-major `(time, feat)` matching ``X.reshape(n, -1)`` from ``X`` shaped ``(N, L, F)``.

    Row 0 of the grid is **oldest** timestep in the window; row ``seq_length-1`` is the **last**
    bar before the label (same stacking as NumPy ``(L, F)``.reshape(-1)).
    """
    flat_scores = np.asarray(flat_scores, dtype=float).ravel()
    nfeat = len(feature_names)
    if flat_scores.size != seq_length * nfeat:
        raise ValueError(
            f"flat size {flat_scores.size} != seq_length * n_features "
            f"({seq_length} * {nfeat})"
        )
    arr = flat_scores.reshape(seq_length, nfeat)
    ix = pd.Index(range(seq_length), name="timestep_in_window")
    return pd.DataFrame(arr, index=ix, columns=list(feature_names))


def plot_close_and_rsi(
    dates: pd.DatetimeIndex | pd.Index,
    close: pd.Series,
    rsi: pd.Series,
    ticker: str,
    *,
    period_label: str = "14",
    figsize: tuple[float, float] = (12, 5.5),
    show: bool = True,
):
    """Plot close (top) and RSI (bottom), aligned on ``dates``.

    RSI is expected to match ``data_pipeline.get_rsi`` / ``StockMarketDataset.rsi_*``.
    Draws heuristic 30/70 reference bands (not trading advice).
    """
    fig, (ax_price, ax_rsi) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        height_ratios=[1.25, 1],
    )
    ax_price.plot(
        dates, close, color="black", linewidth=0.85, alpha=0.92, label="close"
    )
    ax_price.set_ylabel("price (USD)")
    ax_price.set_title(
        f"{ticker} — close (top) and RSI({period_label}) (bottom, data_pipeline definition)"
    )
    ax_price.legend(loc="upper left", fontsize=8)
    ax_price.grid(True, alpha=0.3)

    ax_rsi.plot(
        dates,
        rsi,
        color="tab:purple",
        linewidth=0.9,
        label=f"RSI({period_label})",
    )
    ax_rsi.axhline(70, color="tab:red", linestyle="--", linewidth=0.7, alpha=0.85)
    ax_rsi.axhline(30, color="tab:green", linestyle="--", linewidth=0.7, alpha=0.85)
    ax_rsi.axhline(50, color="gray", linestyle=":", linewidth=0.5, alpha=0.7)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel(f"RSI({period_label})")
    ax_rsi.set_xlabel("Date")
    ax_rsi.set_title(
        "Heuristic reference: 30 / 70 bands (not a trading recommendation)"
    )
    ax_rsi.legend(loc="upper left", fontsize=8)
    ax_rsi.grid(True, alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, (ax_price, ax_rsi)


def plot_window_flat_scores(
    panels: dict[str, np.ndarray],
    seq_length: int,
    feature_names,
    *,
    zlabel: str = "score",
    suptitle: str | None = None,
    cmap: str = "magma",
    share_scale: bool = True,
    show: bool = True,
):
    """Heatmaps of per-(timestep x channel) scores, e.g. mutual information per flattened column."""
    names = tuple(feature_names)
    grids = {}
    vmin, vmax = None, None
    for key, flat in panels.items():
        grids[key] = window_scores_flat_to_grid(flat, seq_length, names)
        g = grids[key].to_numpy(dtype=float)
        finite = np.isfinite(g)
        if finite.any():
            lo, hi = float(np.nanmin(g[finite])), float(np.nanmax(g[finite]))
            vmin = lo if vmin is None else min(vmin, lo)
            vmax = hi if vmax is None else max(vmax, hi)

    n_p = len(panels)
    ncols = min(2, n_p)
    nrows = int(np.ceil(n_p / ncols))
    fig_w = max(11.0, 2.8 * ncols + 2.5)
    fig_h = max(4.8, 3.2 * nrows + 1.5)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )
    if not share_scale:
        vmin, vmax = None, None

    for ax, (title, _) in zip(axes.ravel(), panels.items()):
        grid = grids[title].to_numpy(dtype=float)
        im_kw = dict(
            aspect="auto",
            cmap=cmap,
            origin="upper",
            interpolation="nearest",
        )
        if vmin is not None and vmax is not None and vmin < vmax:
            im_kw["vmin"], im_kw["vmax"] = vmin, vmax
        im = ax.imshow(grid, **im_kw)
        ax.set_title(title)
        ax.set_xticks(np.arange(grid.shape[1]))
        ax.set_xticklabels(list(names), rotation=55, ha="right", fontsize=8)
        ax.set_yticks(np.arange(seq_length))
        ax.set_yticklabels(
            [
                (
                    "oldest"
                    if i == 0
                    else ("newest (= t-1)" if i == seq_length - 1 else str(i))
                )
                for i in range(seq_length)
            ],
            fontsize=8,
        )
        ax.set_xlabel("input channel")
        ax.set_ylabel("position in seq window")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=zlabel)

    for ax in axes.ravel()[len(panels) :]:
        ax.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, y=1.02, fontsize=12)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes, grids


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

def get_window_end_prices(full_ds, df):
    """
    Returns the close price at the end of each input window.
    Shape: (N,)
    """

    close = df["Close"].squeeze()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    idx = full_ds.returns.index
    close_on_returns = close.reindex(idx)

    returns = full_ds.returns.to_numpy(dtype=np.float64)

    cum = np.zeros(len(returns) + 1, dtype=np.float64)
    cum[1:] = np.cumsum(returns)

    bases = np.array(
        [full_ds._warmup + k * full_ds.stride for k in range(len(full_ds))],
        dtype=np.intp
    )

    seq_length = full_ds.seq_length

    cumulative_log_returns = cum[bases + seq_length] - cum[bases]

    prev_before_window = (
        close_on_returns
        .shift(1)
        .iloc[bases]
        .to_numpy(dtype=np.float64)
    )

    prices_end_of_window = prev_before_window * np.exp(cumulative_log_returns)

    actual_last_in_window = (
        close_on_returns
        .iloc[bases + seq_length - 1]
        .to_numpy(dtype=np.float64)
    )

    np.testing.assert_allclose(
        prices_end_of_window,
        actual_last_in_window,
        rtol=1e-8,
        atol=1e-8
    )

    return prices_end_of_window, bases

def log_returns_to_terminal_prices(anchor_prices, log_returns):
    """``anchor × exp(cumulative log return)`` per row / per output head.

    ``anchor_prices``: shape ``(N,)`` — one scalar anchor per sample (e.g. close before horizon).
    ``log_returns``: shape ``(N,)`` or ``(N, n_outputs)`` — inverse-transformed targets.

    Broadcasting: anchors need a trailing axis to multiply ``(N, K)``.
    """
    anchors = np.asarray(anchor_prices, dtype=np.float64).reshape(-1)
    logs = np.asarray(log_returns, dtype=np.float64)
    mult = np.exp(logs)
    if mult.ndim == 1:
        return anchors * mult
    return anchors[:, np.newaxis] * mult

def get_rmse(true_prices, predicted_prices):
    return root_mean_squared_error(true_prices, predicted_prices)

def as_numpy_all_x(ds, df):
    n_returns = len(ds.returns)
    n = n_returns - ds.seq_length - ds._warmup

    X_list = []
    y_list = []
    anchor_prices = []
    target_start_positions = []

    close = df["Close"].squeeze()

    for i in range(n):
        base = ds._warmup + i * ds.stride
        lbl = base + ds.seq_length

        X = np.stack(
            [col.iloc[base : base + ds.seq_length].values for col in ds.cols_x],
            axis=-1
        )

        y = np.array(
            [col.iloc[lbl] for col in ds.cols_y],
            dtype=np.float64
        )

        X_list.append(X)
        y_list.append(y)

        # df index has +1 because returns used diff().dropna()
        anchor_pos = 1 + base + ds.seq_length - 1
        target_start_pos = 1 + lbl

        anchor_prices.append(close.iloc[anchor_pos])
        target_start_positions.append(target_start_pos)

    return (
        np.stack(X_list),
        np.stack(y_list),
        np.asarray(anchor_prices),
        np.asarray(target_start_positions)
    )

def plot_prediction_timeline(
    dates,
    true_prices,
    predicted_prices,
    target_names,
    output_index=0,
    future_date=None,
    future_prediction=None
):
    plt.figure(figsize=(14, 6))

    plt.plot(
        dates,
        true_prices[:, output_index],
        label="Actual price",
        color="blue"
    )

    plt.plot(
        dates,
        predicted_prices[:, output_index],
        label="Predicted price",
        color="green"
    )

    # Add future prediction point
    if future_date is not None and future_prediction is not None:
        plt.scatter(
            future_date,
            future_prediction,
            color="red",
            s=120,
            label="Future forecast",
            zorder=5
        )

        plt.plot(
            [dates[-1], future_date],
            [
                predicted_prices[-1, output_index],
                future_prediction
            ],
            color="red",
            linestyle="--"
        )

    rmse = get_rmse(
        true_prices[:, output_index],
        predicted_prices[:, output_index]
    )

    plt.xlabel("Date")
    plt.ylabel("Price")

    plt.title(
        f"Prediction Timeline - "
        f"{target_names[output_index]} "
        f"- RMSE: {rmse:.6f}"
    )

    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)
    plt.show()