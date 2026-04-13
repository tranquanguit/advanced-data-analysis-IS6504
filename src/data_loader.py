from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.config import PROVINCE_COL


def _ensure_columns(df: pd.DataFrame, required: list[str], file_path: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {file_path}")


def load_all_provinces(
    folder: Path,
    target_col: str,
    cases_col: str | None = None,
    compute_rate_per100k: bool = False,
) -> pd.DataFrame:
    files = sorted(folder.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No .xlsx files found in {folder}")

    frames: list[pd.DataFrame] = []
    for file in files:
        df = pd.read_excel(file)
        _ensure_columns(df, ["year", "month"], file)

        # Drop non-informative columns if present
        for col in ["Unnamed: 0", "year_month"]:
            if col in df.columns:
                df = df.drop(columns=col)

        province_name = file.stem
        df[PROVINCE_COL] = province_name
        df["date"] = pd.to_datetime(
            df["year"].astype(int).astype(str) + "-" + df["month"].astype(int).astype(str) + "-01"
        )

        # Ensure target column exists; optionally construct from cases + population
        if target_col not in df.columns:
            if compute_rate_per100k and cases_col and {"population_male", "population_female"}.issubset(df.columns):
                df["population_total"] = df["population_male"].fillna(0) + df["population_female"].fillna(0)
                denom = df["population_total"].replace(0, pd.NA)
                if cases_col in df.columns:
                    df[target_col] = (df[cases_col] / denom) * 1e5
            else:
                raise ValueError(f"Target column {target_col} not found and cannot be constructed in {file}")

        frames.append(df)

    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out.sort_values([PROVINCE_COL, "date"]).reset_index(drop=True)

    # Fill missing values within each province only to avoid cross-province leakage
    # and avoid pandas reset_index collisions on the grouping column.
    filled_frames: list[pd.DataFrame] = []
    for _, g in out.groupby(PROVINCE_COL, sort=False):
        filled_frames.append(g.ffill().bfill())
    out = pd.concat(filled_frames, axis=0, ignore_index=True)
    return out
