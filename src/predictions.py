import torch.nn as nn
from model import StockMarketModel
import torch
from data_pipeline import X_test, y_test

device = "cuda" if torch.cuda.is_available() else "cpu"

model = StockMarketModel(input_dim=1, hidden_dim=64, num_layers=2, output_dim=1).to(device)

