"""Conv1D → LSTM → additive attention → MLP head (multi-target regression).

Mirrors the user's Keras sketch (Conv → BN → dropout → LSTM with sequences →
Attention → dropout → Dense), adapted to this project: no sigmoid on the head
because targets are the next-step 5-vector (return, body, range, vol_chg, close_pos).
"""

import torch
import torch.nn as nn


class AdditiveAttentionPooling(nn.Module):
    """Weight LSTM time steps and sum to a fixed-size context vector."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, H)
        u = torch.tanh(self.proj(x))
        scores = self.score(u).squeeze(-1)  # (B, L)
        weights = torch.softmax(scores, dim=-1)
        ctx = (x * weights.unsqueeze(-1)).sum(dim=1)  # (B, H)
        return ctx


class ConvLSTMAttentionStockModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        conv_channels: int = 32,
        kernel_size: int = 3,
        lstm_hidden: int = 64,
        head_hidden: int = 32,
        dropout: float = 0.2,
        num_lstm_layers: int = 1
    ):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.num_lstm_layers = num_lstm_layers

        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn = nn.BatchNorm1d(conv_channels)
        self.drop = nn.Dropout(dropout)
        lstm_dropout = dropout if num_lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout
        )
        self.attn = AdditiveAttentionPooling(lstm_hidden)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # L: days in the window
        # F: features
        # x: (B, L, F)
        b = x.size(0)
        xc = x.transpose(1, 2)  # (B, F, L)
        xc = self.conv(xc)
        xc = self.bn(xc)
        xc = torch.relu(xc)
        xc = self.drop(xc)
        xl = xc.transpose(1, 2)  # (B, L, conv_out)

        h0 = torch.zeros(self.num_lstm_layers, b, self.lstm_hidden, device=x.device, dtype=x.dtype)
        c0 = torch.zeros(self.num_lstm_layers, b, self.lstm_hidden, device=x.device, dtype=x.dtype)
        seq, _ = self.lstm(xl, (h0, c0))
        ctx = self.attn(seq)
        ctx = self.drop(ctx)
        return self.head(ctx)