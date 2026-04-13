from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.metrics import (
    distance_corr,
    kendall_tau,
    mutual_info,
    pearson_corr,
    permutation_p_value,
    spearman_corr,
)


@dataclass
class AnalysisParams:
    province_col: str
    disease_vars: list[str]
    climate_vars: list[str]
    social_vars: list[str]
    include_other_diseases_as_predictors: bool
    lags: list[int]
    min_samples_global: int
    min_samples_province: int
    min_provinces_global: int
    max_samples_for_distance: int
    max_samples_for_mutual_info: int
    max_missing_ratio_pair: float
    min_std_predictor: float
    min_std_target: float
    min_unique_predictor: int
    control_mode: str
    compute_permutation_p_value: bool
    n_permutations: int
    random_state: int


def build_params(cfg_analysis: dict) -> AnalysisParams:
    return AnalysisParams(
        province_col=cfg_analysis.get("province_col", "province"),
        disease_vars=cfg_analysis.get("disease_vars", []),
        climate_vars=cfg_analysis.get("climate_vars", []),
        social_vars=cfg_analysis.get("social_vars", []),
        include_other_diseases_as_predictors=cfg_analysis.get("include_other_diseases_as_predictors", True),
        lags=cfg_analysis.get("lags", [0, 1, 2, 3, 4, 5, 6]),
        min_samples_global=int(cfg_analysis.get("min_samples_global", 300)),
        min_samples_province=int(cfg_analysis.get("min_samples_province", 24)),
        min_provinces_global=int(cfg_analysis.get("min_provinces_global", 12)),
        max_samples_for_distance=int(cfg_analysis.get("max_samples_for_distance", 1500)),
        max_samples_for_mutual_info=int(cfg_analysis.get("max_samples_for_mutual_info", 5000)),
        max_missing_ratio_pair=float(cfg_analysis.get("max_missing_ratio_pair", 0.40)),
        min_std_predictor=float(cfg_analysis.get("min_std_predictor", 1e-9)),
        min_std_target=float(cfg_analysis.get("min_std_target", 1e-9)),
        min_unique_predictor=int(cfg_analysis.get("min_unique_predictor", 4)),
        control_mode=str(cfg_analysis.get("control_mode", "none")).strip().lower(),
        compute_permutation_p_value=bool(cfg_analysis.get("compute_permutation_p_value", False)),
        n_permutations=int(cfg_analysis.get("n_permutations", 200)),
        random_state=int(cfg_analysis.get("random_state", 42)),
    )


def _sample_arrays(x: np.ndarray, y: np.ndarray, max_samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if max_samples <= 0 or len(x) <= max_samples:
        return x, y
    idx = rng.choice(len(x), size=max_samples, replace=False)
    return x[idx], y[idx]


def _available(cols: Iterable[str], df_cols: set[str]) -> list[str]:
    return [c for c in cols if c in df_cols]


def _std_ok(series: pd.Series, min_std: float) -> bool:
    return float(series.std(ddof=0)) > min_std


def _apply_control_mode(
    pair_df: pd.DataFrame,
    control_mode: str,
    province_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    x = pair_df["predictor"].astype(float).copy()
    y = pair_df["target"].astype(float).copy()
    mode = (control_mode or "none").lower()

    if mode in {"month", "month_province"} and "month" in pair_df.columns:
        x = x - pair_df.groupby("month")["predictor"].transform("mean")
        y = y - pair_df.groupby("month")["target"].transform("mean")

    if mode in {"province", "month_province"}:
        x = x - pair_df.groupby(province_col)["predictor"].transform("mean")
        y = y - pair_df.groupby(province_col)["target"].transform("mean")

    return x.to_numpy(dtype=float), y.to_numpy(dtype=float)


def _province_control_mode(global_mode: str) -> str:
    mode = (global_mode or "none").lower()
    if mode == "month_province":
        return "month"
    if mode == "province":
        return "none"
    return mode


def analyze_global_dependencies(df: pd.DataFrame, params: AnalysisParams) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(params.random_state)
    df_cols = set(df.columns)

    base_predictors = _available([*params.climate_vars, *params.social_vars], df_cols)
    province_col = params.province_col
    total_rows = len(df)

    for target in params.disease_vars:
        if target not in df_cols:
            continue

        disease_predictors = []
        if params.include_other_diseases_as_predictors:
            disease_predictors = [d for d in params.disease_vars if d != target and d in df_cols]
        predictors = [*base_predictors, *disease_predictors]

        for predictor in predictors:
            grouped = df.groupby(province_col)[predictor]
            for lag in params.lags:
                shifted = grouped.shift(lag)
                pair = pd.DataFrame(
                    {
                        province_col: df[province_col],
                        "month": df["month"] if "month" in df.columns else np.nan,
                        "target": df[target],
                        "predictor": shifted,
                    }
                ).dropna()

                missing_ratio_pair = 1.0 - (len(pair) / total_rows if total_rows else 0.0)
                if missing_ratio_pair > params.max_missing_ratio_pair:
                    continue

                if len(pair) < params.min_samples_global:
                    continue
                if pair[province_col].nunique() < params.min_provinces_global:
                    continue
                if pair["predictor"].nunique() < params.min_unique_predictor:
                    continue
                if not _std_ok(pair["predictor"], params.min_std_predictor):
                    continue
                if not _std_ok(pair["target"], params.min_std_target):
                    continue

                x, y = _apply_control_mode(pair, params.control_mode, province_col)
                if np.std(x) <= params.min_std_predictor or np.std(y) <= params.min_std_target:
                    continue

                x_dc, y_dc = _sample_arrays(x, y, params.max_samples_for_distance, rng)
                x_mi, y_mi = _sample_arrays(x, y, params.max_samples_for_mutual_info, rng)

                spearman = spearman_corr(x, y)
                row = {
                    "target": target,
                    "predictor": predictor,
                    "lag": int(lag),
                    "n_samples": int(len(pair)),
                    "n_provinces": int(pair[province_col].nunique()),
                    "missing_ratio_pair": float(missing_ratio_pair),
                    "control_mode": params.control_mode,
                    "pearson_corr": pearson_corr(x, y),
                    "spearman_corr": spearman,
                    "kendall_tau": kendall_tau(x, y),
                    "distance_corr": distance_corr(x_dc, y_dc),
                    "mutual_info": mutual_info(x_mi, y_mi, random_state=params.random_state),
                }

                if params.compute_permutation_p_value:
                    row["spearman_perm_p"] = permutation_p_value(
                        x_dc,
                        y_dc,
                        stat_fn=spearman_corr,
                        n_permutations=params.n_permutations,
                        random_state=params.random_state,
                    )
                else:
                    row["spearman_perm_p"] = float("nan")

                rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_spearman"] = out["spearman_corr"].abs()
    out["abs_pearson"] = out["pearson_corr"].abs()
    return out


def rank_relationships(global_df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    if global_df.empty:
        return global_df

    out = global_df.copy()
    weight_abs_spearman = float(weights.get("abs_spearman", 0.35))
    weight_distance = float(weights.get("distance_corr", 0.40))
    weight_mi = float(weights.get("mutual_info", 0.25))

    for col in ["abs_spearman", "distance_corr", "mutual_info"]:
        series = out[col].fillna(0.0)
        min_v = float(series.min())
        max_v = float(series.max())
        if max_v == min_v:
            out[f"{col}_scaled"] = 0.0
        else:
            out[f"{col}_scaled"] = (series - min_v) / (max_v - min_v)

    out["composite_score"] = (
        weight_abs_spearman * out["abs_spearman_scaled"]
        + weight_distance * out["distance_corr_scaled"]
        + weight_mi * out["mutual_info_scaled"]
    )
    out = out.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return out


def analyze_province_for_top_relationships(
    df: pd.DataFrame,
    top_relationships: pd.DataFrame,
    params: AnalysisParams,
) -> pd.DataFrame:
    if top_relationships.empty:
        return pd.DataFrame()

    rows = []
    province_col = params.province_col
    province_mode = _province_control_mode(params.control_mode)

    for rank, rel in enumerate(top_relationships.itertuples(index=False), start=1):
        target = rel.target
        predictor = rel.predictor
        lag = int(rel.lag)
        relation_key = f"{target} <- {predictor} (lag {lag})"

        shifted = df.groupby(province_col)[predictor].shift(lag)
        pair = pd.DataFrame(
            {
                province_col: df[province_col],
                "target": df[target],
                "predictor": shifted,
            }
        ).dropna()

        for province, g in pair.groupby(province_col):
            if len(g) < params.min_samples_province:
                continue
            if g["predictor"].nunique() < params.min_unique_predictor:
                continue
            if not _std_ok(g["predictor"], params.min_std_predictor):
                continue
            if not _std_ok(g["target"], params.min_std_target):
                continue
            x, y = _apply_control_mode(g, province_mode, province_col)
            if np.std(x) <= params.min_std_predictor or np.std(y) <= params.min_std_target:
                continue
            rows.append(
                {
                    "relationship_rank": rank,
                    "relationship": relation_key,
                    "target": target,
                    "predictor": predictor,
                    "lag": lag,
                    "province": province,
                    "n_samples": int(len(g)),
                    "control_mode": province_mode,
                    "spearman_corr": spearman_corr(x, y),
                    "kendall_tau": kendall_tau(x, y),
                    "distance_corr": distance_corr(x, y),
                    "mutual_info": mutual_info(x, y, random_state=params.random_state),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_spearman"] = out["spearman_corr"].abs()
    return out
