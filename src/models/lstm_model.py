from __future__ import annotations

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, out_dim: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])
