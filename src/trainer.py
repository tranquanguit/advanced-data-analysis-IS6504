from __future__ import annotations

import torch


def train_lstm(model, x_train, y_train, epochs: int = 30, lr: float = 1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    model.train()

    for _ in range(epochs):
        pred = model(x_train)
        loss = loss_fn(pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model
