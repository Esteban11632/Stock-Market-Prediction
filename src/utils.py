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

    bg = torch.from_numpy(X_train[:bg_n]).to(device)
    smpl = torch.from_numpy(X_test[:ex_n]).to(device)

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

    explainer = shap.GradientExplainer(target_model, bg)

    target_model.train()
    try:
        shap_values = explainer.shap_values(smpl)
    finally:
        target_model.train(prev_mode)

    return shap_values, smpl


def shap_time_heatmap(feature_names, sv):
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

    plt.title("SHAP Feature Importance Across Time")

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

def predicted_returns_to_prices(df, full_ds, test_indices, seq_length, pred_returns):
    """Map predicted **daily log return** (column 0) to next close: ``prev * exp(pred)``."""
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
        pred_prices.append(prev_c * float(np.exp(pred_ret_flat[j])))
        actual_prices.append(float(close.loc[r.index[p]]))
        test_dates.append(r.index[p])

    pred_prices = np.array(pred_prices)
    actual_prices = np.array(actual_prices)
    test_dates = pd.DatetimeIndex(test_dates)

    return pred_prices, actual_prices, test_dates


def _cumulative_log_targets_to_terminal_closes(prev_close: float, cumulative_log_row):
    """Terminal close from anchor and each head's **cumulative log return** (sum of daily L)."""
    row = np.asarray(cumulative_log_row, dtype=np.float64).ravel()
    return prev_close * np.exp(row)

def forecast_price_path_from_last_sample(
    df, full_ds, seq_length, dataset_index_last, y_pred_last, y_true_last=None
):
    """From last dataset row: dates and closes at each forward horizon (same order as ``cols_y``)."""
    close = get_column_normalized_to_1d(df, "Close")
    r_ix = full_ds.returns.index
    p = dataset_index_last + full_ds._warmup + seq_length
    horizons = tuple(h for _, h in full_ds._fwd_target_specs)
    pred_row = np.asarray(y_pred_last, dtype=np.float64).ravel()
    if len(pred_row) != len(horizons):
        raise ValueError(
            f"y_pred row has {len(pred_row)} cols but _fwd_target_specs defines {len(horizons)} horizons"
        )
    # Compound H returns uses bars p..p+H-1; terminal close is end of day p+H-1.
    horizon_day_offsets = tuple(h - 1 for h in horizons)
    max_off = max(horizon_day_offsets)
    if p + max_off >= len(r_ix):
        raise ValueError(
            f"Not enough trailing bars (need index p+{max_off} < {len(r_ix)}); "
            "shorten longest horizon or use more price history."
        )
    prev_c = float(close.loc[r_ix[p - 1]])
    dates = pd.DatetimeIndex([r_ix[p + o] for o in horizon_day_offsets])
    pred_closes = _cumulative_log_targets_to_terminal_closes(prev_c, pred_row)
    realized_closes = None
    if y_true_last is not None:
        true_row = np.asarray(y_true_last, dtype=np.float64).ravel()
        if true_row.shape == pred_row.shape and np.all(np.isfinite(true_row)):
            realized_closes = _cumulative_log_targets_to_terminal_closes(prev_c, true_row)
    return dates, np.asarray(pred_closes, dtype=np.float64), realized_closes


def graph_predictions(
    ticker,
    test_dates,
    actual_prices,
    pred_prices,
    test_rmse_price=None,
    *,
    forward_forecast=None,
):
    """Actual vs one-step pred close; optional overlay of horizon price forecasts (last sample)."""
    actual_prices = np.asarray(actual_prices, dtype=np.float64).ravel()
    pred_prices = np.asarray(pred_prices, dtype=np.float64).ravel()
    if test_rmse_price is None:
        test_rmse_price = root_mean_squared_error(actual_prices, pred_prices)
        print(f"RMSE (price, one-step next-bar): {test_rmse_price:.4f}")

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(4, 1)
    ax1 = fig.add_subplot(gs[:3, 0])
    ax2 = fig.add_subplot(gs[3, 0])

    ax1.plot(test_dates, actual_prices, color="blue", label="Actual close")
    ax1.plot(
        test_dates,
        pred_prices,
        color="green",
        label="Pred. close (one-step)",
    )

    title = f"{ticker} — predicted vs actual close"

    if forward_forecast:
        fd = pd.DatetimeIndex(forward_forecast["forecast_dates"])
        fp = np.asarray(forward_forecast["pred_closes"], dtype=np.float64).ravel()
        ax1.plot(
            fd,
            fp,
            color="darkgreen",
            linestyle="--",
            marker="o",
            linewidth=2,
            markersize=7,
            label="Pred. closes (fwd 1d / 1w / 1m / 3m from last anchor)",
            zorder=5,
        )
        rc = forward_forecast.get("realized_closes")
        if rc is not None:
            rc = np.asarray(rc, dtype=np.float64).ravel()
            ok = np.isfinite(rc)
            if ok.any():
                ax1.scatter(
                    fd[ok],
                    rc[ok],
                    color="navy",
                    s=52,
                    marker="s",
                    zorder=6,
                    label="Realized closes at horizons (last sample)",
                )
        ax1.axvline(fd[0], color="gray", linestyle=":", linewidth=1, alpha=0.8)
        title = f"{ticker} — close + horizon price forecast (last sample)"

    ax1.set_title(title)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.grid(True)
    ax1.legend(loc="best", fontsize=8)

    err = np.abs(actual_prices - pred_prices)
    ax2.axhline(test_rmse_price, color="blue", linestyle="--", label="RMSE (price)")
    ax2.plot(test_dates, err, color="red", label="Absolute price error")
    ax2.set_title(f"{ticker} — price error (one-step)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Error")
    ax2.grid(True)
    ax2.legend()

    fig.tight_layout()
    plt.show()
