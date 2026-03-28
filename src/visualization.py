from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_prediction(y_true: np.ndarray, y_pred: np.ndarray, out_file: Path, horizon_idx: int = 0):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(y_true[:, horizon_idx], label="actual", linewidth=1.8)
    plt.plot(y_pred[:, horizon_idx], label="pred", linewidth=1.2)
    plt.title(f"Prediction vs actual (horizon={horizon_idx + 1})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
