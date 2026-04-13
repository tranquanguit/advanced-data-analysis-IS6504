# Non-Linear Correlation Analysis Sub-Project

This sub-project pivots the main project into a pure **non-linear correlation analysis** workflow.
It does not modify the forecasting code in the root project.

## Goal

Analyze lagged, non-linear dependencies between:

- Disease variables
- Climate variables
- Social variables

across all provinces and time points.

## What This Pipeline Produces

- Global lagged dependency table (`CSV`)
- Ranked top relationships (`CSV`)
- Province-level heterogeneity for top relationships (`CSV`)
- Heatmaps by target disease (`PNG`)
- Top relationship chart (`PNG`)
- Province variability chart (`PNG`)
- Auto-generated insight text (`TXT`)
- Report-ready summary (`MD`)

## Folder Layout

```text
non-linear-correlation-analysis/
├── configs/
│   └── default.yaml
├── src/
│   ├── analyzer.py
│   ├── data_loader.py
│   ├── metrics.py
│   ├── reporting.py
│   ├── runtime_config.py
│   └── visualization.py
├── outputs/
├── requirements.txt
└── run_analysis.py
```

## Setup

From this sub-project directory:

```bash
pip install -r requirements.txt
```

## Run

```bash
python run_analysis.py --config configs/default.yaml
```

## Default Data Source

`configs/default.yaml` points to:

- `../../data/raw` (root project raw data)

so this sub-project can reuse existing data without touching old code.
