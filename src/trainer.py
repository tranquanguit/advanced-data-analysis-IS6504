from __future__ import annotations

import torch


def train_lstm(
    model,
    x_train,
    y_train,
    val_data=None,
    epochs: int = 60,
    lr: float = 1e-3,
    patience: int = 8,
    batch_size: int = 64,
    horizon_weights: list[float] | None = None,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    best_val = float("inf")
    wait = 0

    n = x_train.size(0)
    indices = torch.arange(n)

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = x_train[idx]
            yb = y_train[idx]
            pred = model(xb)
            
            if horizon_weights is not None:
                hw_tensor = torch.tensor(horizon_weights, dtype=torch.float32, device=pred.device)
                # xb has shape [B, ...], yb has shape [B, H], pred has shape [B, H]
                mse_per_horizon = torch.mean((pred - yb) ** 2, dim=0) # [H]
                loss = torch.sum(mse_per_horizon * hw_tensor) / torch.sum(hw_tensor) # normalized weighted MSE
            else:
                loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if val_data is None:
            continue
        model.eval()
        with torch.no_grad():
            val_pred = model(val_data[0])
            if horizon_weights is not None:
                hw_tensor = torch.tensor(horizon_weights, dtype=torch.float32, device=val_pred.device)
                mse_per_horizon = torch.mean((val_pred - val_data[1]) ** 2, dim=0)
                val_loss = (torch.sum(mse_per_horizon * hw_tensor) / torch.sum(hw_tensor)).item()
            else:
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
