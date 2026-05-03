"""CNN front-end → token projection → positional encoding → Transformer encoder → MLP head.

Matches the CTTS-style stack (Conv over time → tokens + position → Transformer), with a
multi-target regression head instead of classification.
"""

import torch
import torch.nn as nn

class StockTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 64,
        conv_channels: int = 32,
        kernel_size: int = 2,
        stride: int | None = None,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        dim_feedforward: int = 128,
        max_seq_len: int = 512,
        head_hidden: int = 32
    ):
        super().__init__()

        # Patch-style conv: non-overlapping local windows -> tokens (CTTS-style)
        if stride is None:
            stride = kernel_size
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )
        self.bn = nn.BatchNorm1d(conv_channels)
        self.token_projection = nn.Linear(conv_channels, d_model)

        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,  # Norm → attention / FFN → residual (common in diagrams)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        xc = x.transpose(1, 2)  # (B, F, L)
        xc = self.conv(xc)       # (B, conv_channels, L_tokens)  with L_tokens = (L - k)//stride + 1
        xc = self.bn(xc)
        xc = torch.relu(xc)

        tokens = xc.transpose(1, 2)              # (B, L_tokens, conv_channels)
        x = self.token_projection(tokens)        # (B, L_tokens, d_model)

        n_tokens = x.size(1)
        x = x + self.positional_encoding[:, :n_tokens, :]

        x = self.transformer(x)                  # (B, L_tokens, d_model)

        pooled = x.mean(dim=1)                   # latent embedding of the time series
        return self.head(pooled)
