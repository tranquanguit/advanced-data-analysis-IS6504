from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RuntimeConfig:
    raw: dict[str, Any]

    @property
    def paths(self) -> dict[str, Any]:
        return self.raw.get("paths", {})

    @property
    def experiment(self) -> dict[str, Any]:
        return self.raw.get("experiment", {})

    @property
    def model(self) -> dict[str, Any]:
        return self.raw.get("model", {})

    @property
    def run(self) -> dict[str, Any]:
        return self.raw.get("run", {})


def load_runtime_config(config_path: str | Path) -> RuntimeConfig:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return RuntimeConfig(raw=cfg)
