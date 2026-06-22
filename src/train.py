import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
import torch
import torch.optim as optim
from cnn_lstm_attention import ConvLSTMAttentionStockModel
from tqdm import tqdm
from pathlib import Path
from data_pipeline import StockMarketDataset
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import utils
from joblib import dump
from config import get_config
from purgeKFold import PurgedKFoldCustom as PurgedKFold
from tickers import tickers
import numpy as np
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

config = get_config()

seq_length = config["seq_length"]
train_start_date = config["train_start_date"]
train_end_date = config["train_end_date"]
batch_size = config["batch_size"]
max_epochs = config["max_epochs"]
patience = config["patience"]
wanted_features = config["wanted_features"]
engineering_features = config["engineering_features"]

def fit_scalers(X_train, y_train, nf_in):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    scaler_X.fit(X_train.reshape(-1, nf_in))
    scaler_y.fit(y_train)
    return scaler_X, scaler_y


def scale_xy(scaler_X, scaler_y, X, y, nf_in):
    X_scaled = scaler_X.transform(X.reshape(-1, nf_in)).reshape(X.shape)
    y_scaled = scaler_y.transform(y)
    return X_scaled, y_scaled


def make_loaders(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, batch_size):
    train_ds = TensorDataset(
        torch.from_numpy(X_train_scaled).float(),
        torch.from_numpy(y_train_scaled).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val_scaled).float(),
        torch.from_numpy(y_val_scaled).float(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(
        val_ds,
        batch_size=min(512, len(val_ds)),
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader


def make_train_loader(X_scaled, y_scaled, batch_size):
    train_ds = TensorDataset(
        torch.from_numpy(X_scaled).float(),
        torch.from_numpy(y_scaled).float(),
    )
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)


def new_model(nf_in, nf_out, device):
    return ConvLSTMAttentionStockModel(
        input_dim=nf_in,
        output_dim=nf_out,
        num_lstm_layers=config["num_lstm_layers"],
    ).to(device)


def train_model_cv(
    X_train,
    y_train,
    X_val,
    y_val,
    nf_in,
    nf_out,
    device,
    fold_label=None,
):
    """Train with early stopping; used inside PurgedKFold to pick best_epoch per fold."""
    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError(
            f"{fold_label or 'CV fold'} has empty train ({len(X_train)}) or val ({len(X_val)}) split. "
            "Check chronological sorting before PurgedKFold."
        )
    scaler_X, scaler_y = fit_scalers(X_train, y_train, nf_in)
    X_train_scaled, y_train_scaled = scale_xy(scaler_X, scaler_y, X_train, y_train, nf_in)
    X_val_scaled, y_val_scaled = scale_xy(scaler_X, scaler_y, X_val, y_val, nf_in)
    train_loader, val_loader = make_loaders(
        X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, batch_size
    )

    model = new_model(nf_in, nf_out, device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    best_val = float("inf")
    best_epoch = 1
    epochs_no_improve = 0
    best_state = None

    desc = f"Training {fold_label}" if fold_label else "Training"
    for epoch in tqdm(range(max_epochs), desc=desc):
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
            batch_elems = X_batch.size(0) * nf_out
            train_loss_sum += loss.item() * batch_elems
            n_elem_train += batch_elems
        train_loss = train_loss_sum / max(n_elem_train, 1)

        model.eval()
        val_loss = utils.mse_mean_over_batches(val_loader, model, criterion, device, nf_out)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epoch % 25 == 0:
            print(
                f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f} (best: {best_val:.4f} @ epoch {best_epoch})"
            )

        if epochs_no_improve >= patience:
            print(
                f"Early stopping at epoch {epoch + 1} "
                f"(no val improvement for {patience} epochs)."
            )
            last_epoch = epoch + 1
            break

    return best_val, last_epoch

def train_model_final(
    X,
    y,
    num_epochs,
    nf_in,
    nf_out,
    device,
    scaler_X,
    scaler_y,
):
    """Train on all trainval for a fixed number of epochs (from CV average)."""
    X_scaled, y_scaled = scale_xy(scaler_X, scaler_y, X, y, nf_in)
    train_loader = make_train_loader(X_scaled, y_scaled, batch_size)

    model = new_model(nf_in, nf_out, device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    for epoch in tqdm(range(num_epochs), desc=f"Final model ({num_epochs} epochs)"):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if epoch % 25 == 0 or epoch == num_epochs - 1:
            train_loss = utils.mse_mean_over_batches(
                train_loader, model, criterion, device, nf_out
            )
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")

    return model, X_scaled, y_scaled

all_X = []
all_y = []
all_samples_info_sets = []
first_ds = None

for ticker in tickers:
    print(f"\nBuilding dataset for {ticker}")

    df = yf.download(
        ticker,
        start=train_start_date,
        end=train_end_date,
        auto_adjust=False
    )

    if df.empty:
        print(f"Skipping {ticker}: no price data")
        continue

    ds = StockMarketDataset(
        df=df,
        start_date=train_start_date,
        end_date=train_end_date,
        seq_length=seq_length,
        wanted_features=wanted_features,
        engineering_features=engineering_features,
        ticker=ticker,
    )

    X_i, y_i = ds.as_numpy()

    all_X.append(X_i)
    all_y.append(y_i)
    all_samples_info_sets.append(ds.get_samples_info_sets())

    if first_ds is None:
        first_ds = ds

X = np.concatenate(all_X, axis=0)
y = np.concatenate(all_y, axis=0)

print("NaNs in X:", np.isnan(X).sum())
print("NaNs in y:", np.isnan(y).sum())
print("Inf in X:", np.isinf(X).sum())
print("Inf in y:", np.isinf(y).sum())

full_ds = first_ds
samples_info_sets = pd.concat(all_samples_info_sets, axis=0)

valid_mask = (
    np.isfinite(X).all(axis=(1, 2)) &
    np.isfinite(y).all(axis=1)
)

X = X[valid_mask]
y = y[valid_mask]
samples_info_sets = samples_info_sets.iloc[valid_mask]

# Sort by label-start time so PurgedKFold splits chronologically (required for multi-ticker pools).
sort_order = np.lexsort((np.arange(len(samples_info_sets)), samples_info_sets.index.values))
X = X[sort_order]
y = y[sort_order]
samples_info_sets = samples_info_sets.iloc[sort_order]

print(f"Total samples after NaN removal: {len(X):,}")
print(f"NaNs in X: {np.isnan(X).sum()}")
print(f"NaNs in y: {np.isnan(y).sum()}")

assert len(samples_info_sets) == len(X)

n = len(X)
test_size = int(0.10 * n)
trainval_end = n - test_size

X_trainval = X[:trainval_end]
y_trainval = y[:trainval_end]

X_test = X[trainval_end:]
y_test = y[trainval_end:]

samples_info_sets_trainval = samples_info_sets.iloc[:trainval_end]

nf_in = X.shape[-1]
nf_out = y.shape[-1]

# ============================================================
# Step 1: PurgedKFold — pick best_epoch per fold (leakage-safe validation)
# ============================================================

cv = PurgedKFold(
    n_splits=5,
    samples_info_sets=samples_info_sets_trainval,
    pct_embargo=0.01,
)

fold_scores = []
fold_last_epochs = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_trainval)):
    print(f"\n========== Fold {fold + 1} ==========")

    best_val, last_epoch = train_model_cv(
        X_trainval[train_idx],
        y_trainval[train_idx],
        X_trainval[val_idx],
        y_trainval[val_idx],
        nf_in,
        nf_out,
        device,
        fold_label=f"fold {fold + 1}",
    )
    fold_scores.append(best_val)
    fold_last_epochs.append(last_epoch)
    print(f"Fold {fold + 1} best val loss: {best_val:.6f} @ epoch {last_epoch}")

print("Fold scores:", fold_scores)
print(f"Mean CV val loss: {sum(fold_scores) / len(fold_scores):.6f}")
print(f"Last epochs per fold: {fold_last_epochs}")

final_epochs = max(1, round(sum(fold_last_epochs) / len(fold_last_epochs)))
print(f"Final training epochs (CV average): {final_epochs}")

# ============================================================
# Step 2: Retrain one model on ALL trainval for final_epochs
# ============================================================

print("\n========== Final model (100% trainval) ==========")

scaler_X, scaler_y = fit_scalers(X_trainval, y_trainval, nf_in)

model, X_trainval_scaled, y_trainval_scaled = train_model_final(
    X_trainval,
    y_trainval,
    final_epochs,
    nf_in,
    nf_out,
    device,
    scaler_X,
    scaler_y,
)

# ============================================================
# Step 3: Evaluate on untouched 10% holdout test
# ============================================================

X_test_scaled, y_test_scaled = scale_xy(scaler_X, scaler_y, X_test, y_test, nf_in)

test_ds = TensorDataset(
    torch.from_numpy(X_test_scaled).float(),
    torch.from_numpy(y_test_scaled).float(),
)
train_eval_ds = TensorDataset(
    torch.from_numpy(X_trainval_scaled).float(),
    torch.from_numpy(y_trainval_scaled).float(),
)

test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
eval_train_loader = DataLoader(train_eval_ds, batch_size=batch_size, shuffle=False, drop_last=False)

y_train_pred, y_train_scaled_out = utils.predict_loader(model, eval_train_loader, device)
y_test_pred, y_test_scaled_out = utils.predict_loader(model, test_loader, device)

y_train_preds = scaler_y.inverse_transform(y_train_pred.numpy())
y_train_inv = scaler_y.inverse_transform(y_train_scaled_out.numpy())
y_test_preds = scaler_y.inverse_transform(y_test_pred.numpy())
y_test_inv = scaler_y.inverse_transform(y_test_scaled_out.numpy())

train_rmse = root_mean_squared_error(y_train_inv, y_train_preds)
test_rmse = root_mean_squared_error(y_test_inv, y_test_preds)

_ret = 0
train_rmse_ret = root_mean_squared_error(y_train_inv[:, _ret], y_train_preds[:, _ret])
test_rmse_ret = root_mean_squared_error(y_test_inv[:, _ret], y_test_preds[:, _ret])

print(
    f"Train RMSE (avg over {nf_out} targets): {train_rmse:.6f} | "
    f"Test RMSE (avg over {nf_out} targets): {test_rmse:.6f}"
)
print(f"Train RMSE (1-day cumulative log return): {train_rmse_ret:.6f} | Test: {test_rmse_ret:.6f}")

feature_importance = utils.permutation_feature_importance_mse(
    model,
    X_test_scaled,
    y_test_scaled,
    device,
    feature_names=full_ds.input_feature_names,
    target_output_index=0,
)
print(feature_importance)
utils.plot_permutation_feature_importance(feature_importance)

for output_index, name in enumerate(full_ds.target_column_names):
    print(
        f"\nOutput index: {output_index} "
        f"({name}) SHAP values"
    )

    shap_values, _ = utils.get_shap_values(
        model,
        X_trainval_scaled,
        X_test_scaled,
        device,
        background_size=96,
        sample_size=24,
        output_index=output_index,
    )

    sv = shap_values[..., 0]

    utils.shap_time_heatmap(
        name,
        full_ds.input_feature_names,
        sv,
    )

directory = Path(__file__).resolve().parent.parent
model_dir = directory / "models"
scaler_dir = directory / "scalers"
model_dir.mkdir(parents=True, exist_ok=True)
scaler_dir.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), model_dir / "stock_market_model.pth")
dump(scaler_X, scaler_dir / "scaler_X.joblib")
dump(scaler_y, scaler_dir / "scaler_y.joblib")
