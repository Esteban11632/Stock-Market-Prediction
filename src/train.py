import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
import torch
import torch.optim as optim
from cnn_lstm_attention import ConvLSTMAttentionStockModel
from transformer import StockTransformer
from tqdm import tqdm
from pathlib import Path
from data_pipeline import StockMarketDataset, split_data
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import utils
from joblib import dump
from features import features

device = "cuda" if torch.cuda.is_available() else "cpu"

ticker = "VOO"

df = yf.download(ticker, start="2000-01-01", end="2023-12-31")

seq_length = 10

full_ds = StockMarketDataset(df, seq_length)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(full_ds)

start_test = len(X_train) + len(X_val)
test_indices = list(range(start_test, start_test + len(X_test)))

# Initialize scalers
scaler_X = StandardScaler()
scaler_y = StandardScaler()

nf_in = X_train.shape[-1]
nf_out = y_train.shape[-1]

# Reshape to (N, nf_in) for scaling
scaler_X.fit(X_train.reshape(-1, nf_in))
X_train_scaled = scaler_X.transform(X_train.reshape(-1, nf_in)).reshape(X_train.shape)
X_val_scaled = scaler_X.transform(X_val.reshape(-1, nf_in)).reshape(X_val.shape)
X_test_scaled = scaler_X.transform(X_test.reshape(-1, nf_in)).reshape(X_test.shape)

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

model = ConvLSTMAttentionStockModel(
    input_dim=nf_in,
    output_dim=nf_out
).to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

num_epochs = 190
patience = 60
best_val = float("inf")
epochs_no_improve = 0
best_state = None

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
        batch_elems = X_batch.size(0) * nf_out
        train_loss_sum += loss.item() * batch_elems
        n_elem_train += batch_elems
    train_loss = train_loss_sum / max(n_elem_train, 1)

    model.eval()
    val_loss = mse_mean_over_batches(val_loader, model, criterion, device, nf_out)

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
y_train_pred, y_train_scaled = utils.predict_loader(model, eval_train_loader, device)
y_test_pred, y_test_scaled = utils.predict_loader(model, test_loader, device)

y_train_preds = scaler_y.inverse_transform(y_train_pred.numpy())
y_train_inv = scaler_y.inverse_transform(y_train_scaled.numpy())
y_test_preds = scaler_y.inverse_transform(y_test_pred.numpy())
y_test_inv = scaler_y.inverse_transform(y_test_scaled.numpy())

# Multi-output; sklearn averages RMSE across outputs by default (see multioutput='uniform_average').
train_rmse = root_mean_squared_error(y_train_inv, y_train_preds)
test_rmse = root_mean_squared_error(y_test_inv, y_test_preds)

# Column 0 = daily return (first entry in StockMarketDataset.cols_y).
_ret = 0
train_rmse_ret = root_mean_squared_error(y_train_inv[:, _ret], y_train_preds[:, _ret])
test_rmse_ret = root_mean_squared_error(y_test_inv[:, _ret], y_test_preds[:, _ret])

print(
    f"Train RMSE (avg over {nf_out} targets): {train_rmse:.6f} | "
    f"Test RMSE (avg over {nf_out} targets): {test_rmse:.6f}"
)
print(f"Train RMSE (daily return): {train_rmse_ret:.6f} | Test RMSE (daily return): {test_rmse_ret:.6f}")

feature_importance = utils.permutation_feature_importance_mse(model, X_test_scaled, y_test_scaled, device, feature_names=features)
print(feature_importance)
utils.plot_permutation_feature_importance(feature_importance)

# Save the model and scalers
directory = Path(__file__).resolve().parent.parent
model_dir = directory / "models"
scaler_dir = directory / "scalers"
model_dir.mkdir(parents=True, exist_ok=True)
scaler_dir.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), model_dir / "stock_market_model.pth")
dump(scaler_X, scaler_dir / "scaler_X.joblib")
dump(scaler_y, scaler_dir / "scaler_y.joblib")