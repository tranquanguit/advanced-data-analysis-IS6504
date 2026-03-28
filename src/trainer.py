from __future__ import annotations

import torch


def train_lstm(
    model,
    x_train,
    y_train,
    val_data=None,
    epochs: int = 30,
    lr: float = 1e-3,
    patience: int = 5,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    best_val = float("inf")
    wait = 0

    for _ in range(epochs):
        model.train()
        pred = model(x_train)
        loss = loss_fn(pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if val_data is None:
            continue
        model.eval()
        with torch.no_grad():
            val_pred = model(val_data[0])
            val_loss = loss_fn(val_pred, val_data[1]).item()
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait > patience:
                model.load_state_dict(best_state)
                break
    return model
