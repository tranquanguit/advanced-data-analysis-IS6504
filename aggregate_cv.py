"""Aggregate rolling-CV detailed results from all results_cv/scenario* folders
into a single paper-ready summary (mean +/- std across folds per model)."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

CV_DIR = Path("results_cv")
rows = []
for det in sorted(CV_DIR.glob("scenario*/metrics/cv_results_detailed.csv")):
    sid = det.parent.parent.name
    df = pd.read_csv(det)
    for model, g in df.groupby("model"):
        row = {"scenario": sid, "model": model, "n_folds": int(len(g))}
        for h in range(1, 7):
            c = f"MAE@{h}"
            if c in g.columns:
                row[f"{c}_mean"] = round(float(g[c].mean()), 3)
                row[f"{c}_std"] = round(float(g[c].std()), 3)
        for m in ("precision", "recall"):
            if m in g.columns:
                row[f"{m}_mean"] = round(float(g[m].mean()), 3)
        rows.append(row)

out = pd.DataFrame(rows)
out.to_csv(CV_DIR / "cv_summary_all.csv", index=False)
print(f"Scenarios found: {sorted(out['scenario'].unique())}")
print(out[["scenario", "model", "MAE@1_mean", "MAE@1_std",
           "MAE@3_mean", "MAE@6_mean"]].to_string(index=False))
