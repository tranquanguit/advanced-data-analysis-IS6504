from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    return float(pearsonr(x, y).statistic)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    return float(kendalltau(x, y).statistic)


def distance_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")

    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    n = x.shape[0]

    ax = squareform(pdist(x, metric="euclidean"))
    ay = squareform(pdist(y, metric="euclidean"))

    ax_centered = ax - ax.mean(axis=0)[None, :] - ax.mean(axis=1)[:, None] + ax.mean()
    ay_centered = ay - ay.mean(axis=0)[None, :] - ay.mean(axis=1)[:, None] + ay.mean()

    dcov2 = np.maximum((ax_centered * ay_centered).sum() / (n * n), 0.0)
    dvarx2 = np.maximum((ax_centered * ax_centered).sum() / (n * n), 0.0)
    dvary2 = np.maximum((ay_centered * ay_centered).sum() / (n * n), 0.0)

    if dvarx2 <= 0 or dvary2 <= 0:
        return float("nan")
    return float(np.sqrt(dcov2) / np.sqrt(np.sqrt(dvarx2 * dvary2)))


def mutual_info(x: np.ndarray, y: np.ndarray, random_state: int = 42) -> float:
    if len(x) < 3:
        return float("nan")
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    mi = mutual_info_regression(x, y, random_state=random_state)
    return float(mi[0])


def permutation_p_value(
    x: np.ndarray,
    y: np.ndarray,
    stat_fn,
    n_permutations: int = 200,
    random_state: int = 42,
) -> float:
    if len(x) < 3:
        return float("nan")

    observed = stat_fn(x, y)
    if np.isnan(observed):
        return float("nan")

    rng = np.random.default_rng(random_state)
    count = 0
    for _ in range(n_permutations):
        perm_stat = stat_fn(x, rng.permutation(y))
        if np.isnan(perm_stat):
            continue
        if abs(perm_stat) >= abs(observed):
            count += 1
    return float((count + 1) / (n_permutations + 1))
