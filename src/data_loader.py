from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.config import PROVINCE_COL


def _ensure_columns(df: pd.DataFrame, required: list[str], file_path: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {file_path}")


def load_all_provinces(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No .xlsx files found in {folder}")

    frames: list[pd.DataFrame] = []
    for file in files:
        df = pd.read_excel(file)
        _ensure_columns(df, ["year", "month"], file)

        province_name = file.stem
        df[PROVINCE_COL] = province_name
        df["date"] = pd.to_datetime(
            df["year"].astype(int).astype(str) + "-" + df["month"].astype(int).astype(str) + "-01"
        )
        frames.append(df)

    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out.sort_values([PROVINCE_COL, "date"]).reset_index(drop=True)
    out = out.groupby(PROVINCE_COL, group_keys=False).apply(lambda g: g.ffill().bfill())
    out = out.reset_index(drop=True)
    return out
