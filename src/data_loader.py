from __future__ import annotations

from pathlib import Path
import pandas as pd

PROVINCE_COL = "province"


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


def load_all_provinces_raw(
    folder: Path,
    province_col: str = "province",
) -> pd.DataFrame:
    """Load all province Excel files without target computation.

    This simplified loader is used by the non-linear correlation analysis
    pipeline, which needs raw data without rate construction.
    """
    files = sorted(folder.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No .xlsx files found in {folder}")

    frames: list[pd.DataFrame] = []
    for file in files:
        df = pd.read_excel(file)
        _ensure_columns(df, ["year", "month"], file)

        for col in ["Unnamed: 0", "year_month"]:
            if col in df.columns:
                df = df.drop(columns=col)

        df[province_col] = file.stem
        df["date"] = pd.to_datetime(
            df["year"].astype(int).astype(str) + "-" + df["month"].astype(int).astype(str) + "-01"
        )
        frames.append(df)

    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out.sort_values([province_col, "date"]).reset_index(drop=True)

    # Fill missing values within each province only (no cross-province leakage).
    filled_frames = []
    for _, g in out.groupby(province_col, sort=False):
        filled_frames.append(g.ffill().bfill())
    out = pd.concat(filled_frames, axis=0, ignore_index=True)
    return out


def build_quality_reports(
    df: pd.DataFrame,
    province_col: str,
    analysis_cols: list[str],
) -> dict[str, pd.DataFrame]:
    cols = [c for c in analysis_cols if c in df.columns]
    if not cols:
        empty = pd.DataFrame()
        return {
            "missing_by_col": empty,
            "missing_by_province": empty,
            "static_by_province_col": empty,
        }

    missing_by_col = (
        df[cols].isna().mean().rename("missing_ratio").reset_index().rename(columns={"index": "column"})
        .sort_values("missing_ratio", ascending=False)
        .reset_index(drop=True)
    )

    missing_by_province = (
        df.groupby(province_col)[cols]
        .apply(lambda g: g.isna().mean().mean())
        .rename("avg_missing_ratio")
        .reset_index()
        .sort_values("avg_missing_ratio", ascending=False)
        .reset_index(drop=True)
    )

    static_rows = []
    for province, g in df.groupby(province_col):
        for col in cols:
            s = g[col].dropna()
            if s.empty:
                static_rows.append(
                    {
                        province_col: province,
                        "column": col,
                        "n_unique": 0,
                        "std": 0.0,
                    }
                )
                continue
            static_rows.append(
                {
                    province_col: province,
                    "column": col,
                    "n_unique": int(s.nunique()),
                    "std": float(s.std(ddof=0)),
                }
            )

    static_df = pd.DataFrame(static_rows)
    if not static_df.empty:
        static_df = static_df.sort_values(["n_unique", "std", "column"], ascending=[True, True, True]).reset_index(drop=True)

    return {
        "missing_by_col": missing_by_col,
        "missing_by_province": missing_by_province,
        "static_by_province_col": static_df,
    }
