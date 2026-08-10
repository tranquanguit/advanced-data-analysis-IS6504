from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RuntimeConfig:
    raw: dict[str, Any]
    config_path: Path

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

    @property
    def analysis(self) -> dict[str, Any]:
        return self.raw.get("analysis", {})

    def resolve_path(self, value: str | Path) -> Path:
        """Resolve a path relative to the config file's directory."""
        p = Path(value)
        if p.is_absolute():
            return p
        return (self.config_path.parent / p).resolve()


def load_runtime_config(config_path: str | Path) -> RuntimeConfig:
    p = Path(config_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return RuntimeConfig(raw=cfg, config_path=p)
